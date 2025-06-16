import argparse
import numpy as np
import pandas as pd


def F(embeddings):
    def inner(row):
        return embeddings[row]
    return inner


def G(node_mappings_used):
    def inner(osmid):
        return node_mappings_used[osmid]
    return inner


def H_inv(remap_df):
    remap_inv = dict()
    for osmid in remap_df.T:
        remap_inv.update({remap_df.T[osmid].values.item():osmid})
    def inner(m):
        return remap_inv[m]
    return inner


def main():
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", required=True,
            help="remap.csv")
    parser.add_argument("-l", required=True,
            help="lipschitz_raw.npz")
    pargs = parser.parse_args()
    # -------------------------------------------------------------
    l = np.load(pargs.l, allow_pickle=True)
    r = pd.read_csv(pargs.r, index_col=0)
    # f: {0, 1, ..., n-1} --> \mathbb{R}^{16}
    # g: {osmid} --> {0, 1, ..., n-1}
    # h: {osmid} --> {0, 1, ..., m-1}
    # m <= n
    #
    # L = f(g(h_inv(m))) : {0, 1, ..., m-1} --> \mathbb{R}^{16}
    #
    # Since h is a dict-like dataframe, h_inv can be easily calculated.
    f = F(l['embeddings'])
    g = G(l['node_mappings_used'].item())
    h_inv = H_inv(r)

    L = lambda m: f(g(h_inv(m)))
    # now just enumerate it to get the correctly ordered lipschitz embeddings
    lipschitz = np.stack(list(map(L, r.values.reshape(-1).tolist())))
    np.savez_compressed('lipschitz.npz', lipschitz=lipschitz)


if __name__=="__main__":
    main()
