dataset='Harbin'
future=12
past=12
nlayers=10
gnn_input_dim=18
gnn_hidden_dim=64
enc_input_dim=64
enc_hidden_dim=64
dec_hidden_dim=64
output_dim=1
run_num=1
model_name='model_best_loss_01February2023_03_49_14.pt'
gpu="${1}"
#
traffic_path="data/${dataset}/traffic_df.pkl.gz"
lipschitz_path="data/${dataset}/lipschitz.npz"
adj_path="data/${dataset}/adj_mx.pkl"
seen_path="data/${dataset}/seen_30.npy"

# Now running in theatres!
if echo $* | grep -e "--debug" -q
then
        CUDA_VISIBLE_DEVICES=$gpu python3 -m pdb test.py --traffic_path "${traffic_path}" --lipschitz_path "${lipschitz_path}" --adj_path "${adj_path}" --seen_path "${seen_path}" --keep_tod --future "${future}" --past "${past}" --nlayers "${nlayers}" --gnn_input_dim "${gnn_input_dim}" --gnn_hidden_dim "${gnn_hidden_dim}" --enc_input_dim "${enc_input_dim}" --enc_hidden_dim "${enc_hidden_dim}" --dec_hidden_dim "${dec_hidden_dim}" --output_dim "${output_dim}" --model_name "${model_name}" --run_num "${run_num}"
# debug
else
        CUDA_VISIBLE_DEVICES=$gpu python3 test.py --traffic_path "${traffic_path}" --lipschitz_path "${lipschitz_path}" --adj_path "${adj_path}" --seen_path "${seen_path}" --keep_tod --future "${future}" --past "${past}" --nlayers "${nlayers}" --gnn_input_dim "${gnn_input_dim}" --gnn_hidden_dim "${gnn_hidden_dim}" --enc_input_dim "${enc_input_dim}" --enc_hidden_dim "${enc_hidden_dim}" --dec_hidden_dim "${dec_hidden_dim}" --output_dim "${output_dim}" --model_name "${model_name}" --run_num "${run_num}"
fi
