import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.nn.conv import MessagePassing


class FrigateConv(MessagePassing):
    def __init__(this, in_channels, out_channels):
        super(FrigateConv, this).__init__(aggr='add')
        this.lin = nn.Linear(in_channels, out_channels)
        this.lin_r = nn.Linear(in_channels, out_channels)
        this.lin_rout = nn.Linear(out_channels, out_channels)
        this.lin_ew = nn.Linear(1, 16)
        this.gate = nn.Sequential(
                nn.Linear(16 * 3, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
                nn.Sigmoid(),
                )
        #for p in this.lin_r.parameters():
        #    nn.init.constant_(p.data, 0.)
        #    p.requires_grad = False
    def forward(this, x, edge_index, edge_weight, lipschitz_embeddings):
        if isinstance(x, torch.Tensor):
            x_r = x
            x = this.lin(x)
            x = (x, x)
        else:
            x_r = this.lin_r(x[1])
            x_rest = this.lin(x[0])
            x = (x_rest, x_r)
        out = this.propagate(edge_index, x=x, edge_weight=edge_weight, lipschitz_embeddings=lipschitz_embeddings)
        #out += this.lin_rout(x_r)
        out = F.normalize(out, p=2., dim=-1)
        return out
    def message(this, x_j, edge_index_i, edge_index_j, edge_weight, lipschitz_embeddings):
        edge_weight_j = edge_weight.view(-1, 1)
        edge_weight_j = this.lin_ew(edge_weight_j)
        gating_input = torch.cat((edge_weight_j, lipschitz_embeddings[edge_index_i],
            lipschitz_embeddings[edge_index_j]), dim=1)
        gating = this.gate(gating_input)
        output = x_j * gating
        return output


class GNN(nn.Module):
    def __init__(this, input_dim, hidden_dim, output_dim, nlayers):
        super().__init__()
        this.nlayers = nlayers
        this.gc = nn.ModuleList()
        this.gc.append(FrigateConv(input_dim, hidden_dim))
        for _ in range(this.nlayers - 2):
            this.gc.append(FrigateConv(hidden_dim, hidden_dim))
        this.gc.append(FrigateConv(hidden_dim, output_dim))
        this.gs_sum = gnn.SAGEConv(1, 1, root_weight=False)
        this.reset_param(this.gs_sum)
        this.freezer(this.gs_sum)

    def reset_param(this, module):
        for n, p in module.named_parameters():
            if n.endswith('bias'):
                v = 0.
            elif n.endswith('weight'):
                v = 1.
            torch.nn.init.constant_(p, v)

    def freezer(this, module):
        for p in module.parameters():
            p.requires_grad = False

    def forward(this, xs, adjs, edge_weight, lipschitz, mu, std):
        last = len(adjs) - 1
        if mu is None or std is None:
            mu = 0.
            std = 1.
        x_org = xs.clone()[:, :, :, :1] * std + mu
        for i, (edge_index, e_id, size) in enumerate(adjs):
            xs_target = xs[:, :, :size[1], :]
            xs = this.gc[i]((xs, xs_target), edge_index, edge_weight[e_id],
                    lipschitz_embeddings=lipschitz)
            xs = F.relu(xs)
            if i == last:
                x_org_selected = x_org[:, :, :size[0], :]
                x_org_targ = x_org[:, :, :size[1], :]
                nz_org = (x_org_selected != 0).to(torch.float32)
                nz_targ = (x_org_targ != 0).to(torch.float32)
                x_sum = this.gs_sum((x_org_selected, x_org_targ), edge_index)
                count = this.gs_sum((nz_org, nz_targ), edge_index)
        return xs, x_sum, count


class Encoder(nn.Module):
    def __init__(this, input_dim, hidden_dim, enc_input_dim, enc_hidden_dim, nlayers):
        super().__init__()
        this.gnn = GNN(input_dim, hidden_dim, enc_input_dim//2, nlayers)
        this.rnn = nn.LSTM(enc_input_dim, enc_hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        this.h_0 = nn.Parameter(torch.randn(2, enc_hidden_dim))
        this.c_0 = nn.Parameter(torch.randn(2, enc_hidden_dim))
        this.reset_bias()
    def reset_bias(this):
        r"""https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/3
        """
        for names in this.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(this.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    def forward(this, xs, xrev, adjs, adjs_rev, edge_weight, lipschitz, mu, std):
        hs, xsum, count = this.gnn(xs, adjs, edge_weight, lipschitz, mu, std)
        mean = xsum / count
        mean[mean!=mean]=0.
        hs_rev, rev_sum, rev_count = this.gnn(xrev, adjs_rev, edge_weight, lipschitz, mu, std)
        mean2 = rev_sum / rev_count
        mean2[mean2!=mean2]=0.
        mean = (mean + mean2) / 2
        hs = torch.cat((hs,hs_rev), dim=-1)
        batch_size = hs.size(0)
        n_nodes = hs.size(2)
        h_0 = this.h_0.repeat(batch_size, 1, 1).permute(1, 0, 2).contiguous()
        c_0 = this.c_0.repeat(batch_size, 1, 1).permute(1, 0, 2).contiguous()
        rnn_outputs = []
        for n in range(n_nodes):
            out, (_, _) = this.rnn(hs[:, :, n, :], (h_0, c_0))
            rnn_outputs.append(out[:,-1:,:])
        rnn_outputs = torch.stack(rnn_outputs, dim=2)
        return rnn_outputs, mean


class Decoder(nn.Module):
    def __init__(this, enc_hidden_dim, dec_hidden_dim, dec_output_dim):
        super().__init__()
        this.rnn = nn.LSTM(enc_hidden_dim + dec_output_dim, dec_hidden_dim,
                proj_size=dec_output_dim, batch_first=True, num_layers=2, dropout=0.1)
        this.h_0 = nn.Parameter(torch.randn(2, dec_output_dim))
        this.c_0 = nn.Parameter(torch.randn(2, dec_hidden_dim))
        this.s   = nn.Parameter(torch.randn(1, dec_output_dim))
        this.reset_bias()
    def reset_bias(this):
        r"""https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/3
        """
        for names in this.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(this.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    def forward(this, enc_outputs, ys=None, mean=0, TFRate=1., future=12):
        TFRate = 0.
        if ys is None:
            TFRate = 0.
        decoder_outputs = []
        batch_size = enc_outputs.size(0)
        n_nodes = enc_outputs.size(2)
        last_output = this.s.repeat(batch_size, 1, 1)
        h_last = this.h_0.repeat(batch_size, 1, 1).permute(1, 0, 2).contiguous()
        c_last = this.c_0.repeat(batch_size, 1, 1).permute(1, 0, 2).contiguous()
        for n in range(n_nodes):
            enc_output = enc_outputs[:, :, n, :]
            y_slice = ys[:, :, n, :] if ys is not None else None
            decoder_outputs_per_node = []
            node_mean = torch.mean(mean[:, :, n, :], dim=1, keepdim=True) if isinstance(mean, torch.Tensor) and len(mean.shape) == 4 else mean
            for t in range(future):
                if torch.rand(1) >= TFRate:
                    loop = last_output + node_mean
                else:
                    if t == 0:
                        loop = last_output + node_mean
                    else:
                        loop = y_slice[:, t-1:t, :]
                input = torch.cat((enc_output, loop), dim=2)
                last_output, (h_last, c_last) = this.rnn(input, (h_last, c_last))
                decoder_outputs_per_node.append(last_output+node_mean)
            decoder_outputs_per_node = torch.cat(decoder_outputs_per_node, dim=1)
            decoder_outputs.append(decoder_outputs_per_node)
        decoder_outputs = torch.stack(decoder_outputs, dim=2)
        return decoder_outputs


class Frigate(nn.Module):
    def __init__(this, gnn_input_dim, gnn_hidden_dim,
            enc_input_dim, enc_hidden_dim, dec_hidden_dim,
            output_dim, nlayers):
        super().__init__()
        this.enc = Encoder(gnn_input_dim, gnn_hidden_dim, enc_input_dim,
                enc_hidden_dim, nlayers)
        this.dec = Decoder(enc_hidden_dim, dec_hidden_dim, output_dim)
    def forward(this, xs, xrev, adjs, adjs_rev, edge_weight, lipschitz, ys=None, TFRate=1., future=12, mu=None, std=None):
        enc_output, mean = this.enc(xs, xrev, adjs, adjs_rev, edge_weight, lipschitz, mu, std)
        dec_output = this.dec(enc_output, ys, mean, TFRate, future)
        return dec_output
