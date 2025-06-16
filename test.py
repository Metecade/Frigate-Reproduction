import torch
import argparse
from io import StringIO
from pathlib import Path
from pprint import pprint

from utils.test_data_utils import get_dataloader_and_adj_mx
from model.tester import model_test


def main():
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--traffic_path', type=str, required=True,
            help='path to traffic data (pkl gz format)')
    parser.add_argument('--lipschitz_path', type=str, required=True,
            help='path to lipschitz data (npz)')
    parser.add_argument('--adj_path', type=str, required=True,
            help='path to adjacency data (pickle)')
    parser.add_argument('--seen_path', type=str, required=True,
            help='path to seen nodes index (npy)')
    parser.add_argument('--keep_tod', default=False, action='store_true',
            help='whether to keep time of day (boolean flag)')
    parser.add_argument('--future', type=int, default=12,
            help='how far in the future to predict')
    parser.add_argument('--past', type=int, default=12,
            help='how far in the past to look')
    parser.add_argument('--nlayers', type=int, default=10,
            help='number of layers used in the GNN')
    parser.add_argument('--gnn_input_dim', type=int, required=True,
            help='number of input dimensions taken by gnn')
    parser.add_argument('--gnn_hidden_dim', type=int, required=True,
            help='number of hidden dimensions of gnn')
    parser.add_argument('--enc_input_dim', type=int, required=True,
            help='number of input dimensions taken by lstm\'s encoder')
    parser.add_argument('--enc_hidden_dim', type=int, required=True,
            help='number of hidden dimensions of lstm encoder')
    parser.add_argument('--dec_hidden_dim', type=int, required=True,
            help='number of hidden dimensions of lstm decoder')
    parser.add_argument('--output_dim', type=int, required=True,
            help='number of output dimensions')
    parser.add_argument('--model_name', type=str, required=True,
            help='trained model\'s name that corresponds to given hyperparams')
    parser.add_argument('--run_num', type=int, required=True,
            help='used to find path to model, the run num of training')
    parser.print_usage = parser.print_help
    pargs = parser.parse_args()
    # ---------------------------------------------------
    model_args = {
        'gnn_input_dim':pargs.gnn_input_dim,
        'gnn_hidden_dim':pargs.gnn_hidden_dim,
        'enc_input_dim':pargs.enc_input_dim,
        'enc_hidden_dim':pargs.enc_hidden_dim,
        'dec_hidden_dim':pargs.dec_hidden_dim,
        'output_dim':pargs.output_dim,
        'nlayers':pargs.nlayers,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    dataloaders, edge_weight, n_nodes = get_dataloader_and_adj_mx(
        pargs.traffic_path,
        pargs.lipschitz_path,
        pargs.adj_path,
        pargs.seen_path,
        keep_tod=pargs.keep_tod,
        f=pargs.future,
        p=pargs.past,
        nlayers=pargs.nlayers,
    )
    with StringIO() as s:
        pprint(vars(pargs), stream=s, indent=4)
        print(s.getvalue())
    edge_weight = edge_weight.to(device).to(torch.float32)
    model_test(model_args,
               device,
               dataloaders,
               edge_weight,
               pargs.run_num,
               pargs.model_name,
               n_nodes
    )


if __name__=="__main__":
    main()
