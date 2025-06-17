import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
from options import get_options
from termcolor import cprint


def load_problem(name):
    from problems import mugdrl
    problem = {
        'mugdrl': mugdrl
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, n_charging_station, epoch=None):
    from nets.attention_model_modified_v7 import AttentionModel
    from nets.pointer_network import PointerNetwork

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    #args = load_args(os.path.join(path, 'args.json'))
    opts = get_options()
    args = vars(opts)

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        n_charging_station=n_charging_station,
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    #a,b, c, d = input
    
    #a, b = do_batch_rep((a, b), batch_rep)
    #input = (a, b,  c, d)

    input = do_batch_rep(input, batch_rep)

    
    



    costs = []
    pis = []
    mission_times = []
    agents = []
    uav_indices = []
    chosen_uavs_for_recharge = []
    
    
    
    for i in range(iter_rep):
        
        cprint('Completed one run', 'red', attrs = ['bold'])
        _log_p, pi, cost, agent, uav_index, chosen_uav_for_recharge, _ = inner_func(input) 
        
        
        
        
        
        
        


        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))
        agents.append(agent.view(batch_rep, -1, agent.size(-1)).transpose(0, 1))
        uav_indices.append(uav_index.view(batch_rep, -1, agent.size(-1)).transpose(0, 1))
        chosen_uavs_for_recharge.append(chosen_uav_for_recharge.view(batch_rep, -1, agent.size(-1)).transpose(0, 1))
        #costs.append(m_t_t.view(batch_rep, -1).t())
    
    
    
    
    
    max_length = max(pi.size(-1) for pi in pis)
    padded_pis = []
    for pi in pis:
        current_length = pi.size(-1)
        pad_length = max_length - current_length
        if pad_length > 0:
            # Extract the last element along N and replicate it pad_length times
            last_elements = pi[:, :, -1:].expand(-1, -1, pad_length)
            # Concatenate pi with its replicated last elements
            padded_pi = torch.cat([pi, last_elements], dim=-1)
        else:
            padded_pi = pi
        padded_pis.append(padded_pi)
    
    # Concatenate all padded tensors along a new dimension
    pis = torch.cat(padded_pis, dim=1)

    
    costs = torch.cat(costs, 1)
    agents = torch.cat(
        [F.pad(agent, (0, max_length - agent.size(-1))) for agent in agents],
        1
    )
    
    
    
    
    uav_indices = torch.cat(
        [F.pad(uav_index, (0, max_length - uav_index.size(-1))) for uav_index in uav_indices],
        1
    )
    
    chosen_uavs_for_recharge = torch.cat(
        [F.pad(chosen_uav_for_recharge, (0, max_length - chosen_uav_for_recharge.size(-1))) for chosen_uav_for_recharge in chosen_uavs_for_recharge],
        1
    )
    
    
    
    
    
    

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)


    
    
    
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    agent = agents[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    

    chosen_uav_for_recharge = chosen_uavs_for_recharge [torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    uav_index = uav_indices[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]
    
    
    
    
    return minpis, mincosts, agent, uav_index , chosen_uav_for_recharge
