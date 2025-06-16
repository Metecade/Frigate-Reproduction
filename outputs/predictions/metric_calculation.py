r"""This script is an interface between the predictions saved by Frigate
and metrics calculation.
The predictions file format:
    There should be a file named pred_true.npz in run_<run-number>.
    The npz file has 3 keys: 'truths', 'predictions', 'ignore_val'

    The shape of 'truths' and 'predictions' is the same and is equal to
    (batch_size, Delta, n_nodes, 1). Not all columns from 0 to n_nodes-1
    have valid entries. Only the nodes that weren't seen in training
    have predictions. The columns corresponding to seen nodes contain
    'ignore_val' which is np.inf for now.

This file takes the arguments: --pred_file </path/to/pred_true.npz>
Ex: if you want to calculate the metrics for run_1/pred_true.npz then run
    $ python3 metric_calculation.py --pred_file "run_1/pred_true.npz"
"""
import argparse
import numpy as np


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_file', required=True, type=str)
    # pargs = parser.parse_args()
    pred_dict = np.load("run_30/pred_true.npz")
    truths = pred_dict['truths']
    predictions = pred_dict['predictions']
    ignore_val = pred_dict['ignore_val']
    nodes, = np.where(truths[0, 0, :, 0] != ignore_val)
    batches, = np.where(truths[:, 0, nodes[0], 0] != ignore_val)
    print(f"Calculating MAE of predictions on {nodes}")
    truths = truths[batches, :, :, 0][:, :, nodes]
    predictions = predictions[batches, :, :, 0][:, :, nodes]
    MAE = np.mean(np.abs(truths - predictions))
    print(f"MAE = {MAE}")


if __name__=="__main__":
    main()
