import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm, trange

from .model import Frigate


def model_test(model_args,
        device,
        dataloaders,
        edge_weight,
        run_num,
        model_name,
        n_nodes,
        ):
    model = Frigate(**model_args).to(device)
    model_path = Path("outputs","models",f"run_{run_num}",f"{model_name}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    prediction_path = Path("outputs","predictions",f"run_{run_num}")
    prediction_path.mkdir(exist_ok=True)
    prediction_name = Path(prediction_path, "pred_true.npz")
    #
    test_loss = test_step(model, dataloaders['test_loader'], device, edge_weight, prediction_name, n_nodes)


def test_step(model,
        test_loader,
        device,
        edge_weight,
        prediction_name,
        n_nodes,
        ):
    model.eval()
    dataloader = test_loader['dataloader']
    nbrloader = test_loader['neighbor_loader']
    rnbrloader = test_loader['rev_loader']
    lipschitz = torch.tensor(dataloader.dataset.ls, dtype=torch.float32).to(device)
    mu = torch.tensor(dataloader.dataset.mu, dtype=torch.float32).to(device)
    sig = torch.tensor(dataloader.dataset.sig, dtype=torch.float32).to(device)
    accumulator = Accumulator()
    nb = len(dataloader)
    updates = nb // 10 if nb > 10 else 1
    loop = tqdm(enumerate(dataloader), total=nb, unit='batch', file=sys.stdout)
    loop.set_description("Testing")
    prediction = np.ones((32*nb,12,n_nodes,1)) * np.inf
    truths     = np.ones((32*nb,12,n_nodes,1)) * np.inf
    with torch.no_grad():
        for b, (xs, ys) in loop:
            xs, ys = xs.to(device), ys.to(device)
            xs, ys = xs.to(torch.float32), ys.to(torch.float32)
            bs = xs.shape[0]
            for n1, n2 in zip(nbrloader, rnbrloader):
                batch_size, n_ids, adjs = n1
                adjs = [adj.to(device) for adj in adjs]
                _, n_id_rev, adjs_rev = n2
                adjs_rev = [adj.to(device) for adj in adjs_rev]
                x_slice = xs[:, :, n_ids, :]
                x_rev = xs[:, :, n_id_rev, :]
                y_slice = ys[:, :, n_ids[:batch_size], :1]
                y_hat = model(x_slice, x_rev, adjs, adjs_rev, edge_weight, lipschitz, mu=mu, std=sig)
                accumulator(y_hat, y_slice)
                prediction[b*32:b*32+bs,:,n_ids[:batch_size],:] = y_hat.detach().cpu().numpy()
                truths[b*32:b*32+bs,:,n_ids[:batch_size],:] = y_slice.detach().cpu().numpy()
            if b % updates == 0:
                loop.set_postfix(loss=accumulator.get_score())
    np.savez_compressed(prediction_name, predictions=prediction, truths=truths, ignore_val=np.inf)
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
