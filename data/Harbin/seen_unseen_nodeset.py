import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_nodes", required=True, type=int)
    parser.add_argument("-percent", required=True, type=int)
    pargs = parser.parse_args()
    perm = np.random.RandomState(seed=0).permutation(pargs.n_nodes)
    seen_len = int(pargs.percent / 100 * pargs.n_nodes)
    seen = perm[:seen_len]
    unseen = perm[seen_len:]
    np.save(f"seen_{pargs.percent}.npy", seen)
    np.save(f"unseen_{pargs.percent}.npy", unseen)


if __name__=="__main__":
    main()
