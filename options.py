import os
import time
import argparse
import torch
from  Custom_environment_model_v1 import UAV_UGV_Env

env = UAV_UGV_Env()

def read_args_from_file(file_path):
    """
    Reads arguments from a text file and returns them as a dictionary.
    Each line in the file should be in the format: key=value
    """
    args = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                key, value = line.strip().split('=')
                args[key] = value
    except Exception as e:
        print(f"Error reading arguments from file: {e}")
    return args


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='mugdrl', help="The problem to solve, default 'tsp'")
    parser.add_argument('--uav_graph_size', type=int, default=45, help="The size of the uav mission points graph")
    parser.add_argument('--ugv_graph_size', type=int, default=15, help="The number of the charging stations")
    parser.add_argument('--unique_mission_points', type=float, default=None, help="The action space")
    parser.add_argument('--batch_size', type=int, default=256, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default= int(256*200), help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default= int(256*10),
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=1,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default= 0.5,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default='rollout',
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default='unif',
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=1, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='Tensborad_log', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')  #   default = 'outputs/mugdrl_15/try/epoch-15.pt',   default = 'default = 'model_checkpoints\policy_network_epoch_9.pt' C:/Users/mmonda4/OneDrive - University of Illinois Chicago/Documents/PhD-ARL/Imitation Learning/Evaluation_ss/outputs/mugdrl_15/run_20241111T145114/epoch-0.pt
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    
    parser.add_argument('--No_of_UAVs', default = 4, help='Uav number')
    parser.add_argument('--No_of_UGVs', default = 2, help='Ugv number')
    parser.add_argument('--DRL_model', default = 'proposed', help='DRL model')
    parser.add_argument('--arg_file', type=str, default = 'args.txt', help='Path to a file containing arguments')

    opts = parser.parse_args(args)

    # If an argument file is provided, override the defaults
    if opts.arg_file:
        file_args = read_args_from_file(opts.arg_file)
        if 'No_of_UAVs' in file_args:
            opts.No_of_UAVs = int(file_args['No_of_UAVs'])
        if 'No_of_UGVs' in file_args:
            opts.No_of_UGVs = int(file_args['No_of_UGVs'])
        if 'DRL_model' in file_args:
            opts.DRL_model = file_args['DRL_model']


    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.uav_graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts







