import os
import sys
import torch
import numpy as np
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm, trange
from datetime import datetime
from tensorboardX import SummaryWriter

from .model import Frigate


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
        num_training_steps, last_epoch=-1):
    r"""
    https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/optimization.py#L75
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
                0.0, float(num_training_steps - current_step) / float(max(1,
                    num_training_steps - num_warmup_steps)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_tfrate_calculator(nepochs):
    def TFRate_calculator(epoch):
        a = -1.5
        b = -a - 1.3
        c = 1
        x = epoch / nepochs
        return max(min(a*x**x+b*x+c, 1), 0)
    return TFRate_calculator


def model_train(model_args,
        device,
        nepochs,
        dataloaders,
        edge_weight,
        loss_fn,
        run_num,
        logger,
        log_dir='outputs/tensorboard',
        ):
    model = Frigate(**model_args).to(device)
    opt = optim.AdamW(model.parameters(), lr=5e-4)
    num_training_steps = nepochs * len(dataloaders['train_loader']['dataloader']) * len(dataloaders['train_loader']['neighbor_loader'])
    lr_scheduler = get_linear_schedule_with_warmup(opt, 0, num_training_steps)
    best_val_loss = float('inf')
    patience = 5
    exhausted = 0
    writer = SummaryWriter(logdir=os.path.join(log_dir, f'run_{run_num}'))
    model_path = Path("outputs","models",f"run_{run_num}")
    model_path.mkdir(exist_ok=True)
    model_name = Path(model_path, f'model_best_loss_{datetime.now().strftime("%d%B%Y_%H_%M_%S")}.pt')
    TFRate_calculator = get_tfrate_calculator(nepochs)
    #
    for e in range(1, nepochs + 1):
        logger.info(f'epoch {e}')
        TFRate = TFRate_calculator(e - 1)
        train_loss = train_step(model, opt, lr_scheduler, dataloaders['train_loader'], device, logger, loss_fn, e, edge_weight, TFRate)
        val_loss = val_step(model, dataloaders['val_loader'], device, logger, e, edge_weight)
        writer.add_scalar('loss/train', train_loss, e)
        writer.add_scalar('loss/val', val_loss, e)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            exhausted = 0
            save_model(model, model_name)
            logger.info(f"Model saved at epoch {e} to {model_name} with loss {best_val_loss}")
        else:
            exhausted += 1
        if exhausted >= patience:
            logger.info(f"Early stopping at epoch: {e}")
            break


def train_step(model,
        opt,
        lr_scheduler,
        train_loader,
        device,
        logger,
        loss_fn,
        epoch,
        edge_weight,
        TFRate,
        ):
    model.train()
    accumulator = Accumulator()
    dataloader = train_loader['dataloader']
    nbrloader = train_loader['neighbor_loader']
    rnbrloader = train_loader['rev_loader']
    lipschitz = torch.tensor(dataloader.dataset.ls, dtype=torch.float32).to(device)
    mu = torch.tensor(dataloader.dataset.mu, dtype=torch.float32).to(device)
    sig = torch.tensor(dataloader.dataset.sig, dtype=torch.float32).to(device)
    nb = len(dataloader)
    updates = nb // 10 if nb > 10 else 1
    loop = tqdm(enumerate(dataloader), total=nb, unit='batch', file=sys.stdout)
    loop.set_description(f"Training epoch: {epoch}")
    for b, (xs, ys) in loop:
        xs, ys = xs.to(device), ys.to(device)
        xs, ys = xs.to(torch.float32), ys.to(torch.float32)
        for nf, nb in zip(nbrloader, rnbrloader):
            batch_size, n_ids, adjs = nf
            _, n_id_rev, adjs_rev = nb
            adjs = [adj.to(device) for adj in adjs]
            adjs_rev = [adj.to(device) for adj in adjs_rev]
            x_slice = xs[:, :, n_ids, :]
            x_slice_rev = xs[:, :, n_id_rev, :]
            y_slice = ys[:, :, n_ids[:batch_size], :1]
            y_hat = model(x_slice, x_slice_rev, adjs, adjs_rev, edge_weight, lipschitz, y_slice, TFRate=TFRate, mu=mu, std=sig)
            loss = loss_fn(y_hat, y_slice)
            accumulator(y_hat, y_slice)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
        if b % updates == 0:
            loop.set_postfix(loss=accumulator.get_score())
    return accumulator.get_score()


def val_step(model,
        val_loader,
        device,
        logger,
        epoch,
        edge_weight,
        ):
    model.eval()
    dataloader = val_loader['dataloader']
    nbrloader = val_loader['neighbor_loader']
    rnbrloader = val_loader['rev_loader']
    lipschitz = torch.tensor(dataloader.dataset.ls, dtype=torch.float32).to(device)
    mu = torch.tensor(dataloader.dataset.mu, dtype=torch.float32).to(device)
    sig = torch.tensor(dataloader.dataset.sig, dtype=torch.float32).to(device)
    accumulator = Accumulator()
    nb = len(dataloader)
    updates = 1#nb // 10 if nb > 10 else 1
    loop = tqdm(enumerate(dataloader), total=nb, unit='batch', file=sys.stdout)
    loop.set_description(f"Valid epoch: {epoch}")
    with torch.no_grad():
        for b, (xs, ys) in loop:
            xs, ys = xs.to(device), ys.to(device)
            xs, ys = xs.to(torch.float32), ys.to(torch.float32)
            for nf, nb in zip(nbrloader, rnbrloader):
                batch_size, n_ids, adjs = nf
                _, n_id_rev, adjs_rev = nb
                adjs = [adj.to(device) for adj in adjs]
                adjs_rev = [adj.to(device) for adj in adjs_rev]
                x_slice = xs[:, :, n_ids, :]
                x_slice_rev = xs[:, :, n_id_rev, :]
                y_slice = ys[:, :, n_ids[:batch_size], :1]
                y_hat = model(x_slice, x_slice_rev, adjs, adjs_rev, edge_weight, lipschitz, mu=mu, std=sig)
                accumulator(y_hat, y_slice)
            if b % updates == 0:
                loop.set_postfix(loss=accumulator.get_score())
    return accumulator.get_score()


class Accumulator:
    def __init__(this):
        this.score = 0
        this.n = 0
    def __call__(this, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy().reshape(-1)
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy().reshape(-1)
        n = y_true.shape[0]
        this.score *= this.n / (this.n + n)
        this.n += n
        this.score += np.sum(np.absolute(y_true - y_pred)) / this.n
    def get_score(this):
        return this.score


def save_model(model, model_file):
    torch.save(model.state_dict(), model_file)
