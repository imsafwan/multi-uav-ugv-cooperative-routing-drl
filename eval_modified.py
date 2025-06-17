# -*- coding: utf-8 -*-
"""
@author: safwan

"""

'''
Decoding : sampling with 1024 samples
 
'''

import csv
import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
mp = torch.multiprocessing.get_context('spawn')
from  Custom_environment_model_v4_v2 import UAV_UGV_Env
from termcolor import cprint
import pandas as pd

from options import get_options
opts = get_options()



env = UAV_UGV_Env()
torch.manual_seed(0)

def remove_consecutive_duplicates(a_list):
    if not a_list:  # Check if the list is empty
        return []

    # Initialize a new list with the first element of the original list
    new_list = [a_list[0]]

    # Iterate through the list, compare each element with the previous one
    for i in range(1, len(a_list)):
        if a_list[i] != a_list[i - 1]:
            new_list.append(a_list[i])

    return new_list

def get_best(sequences, cost, agents, uav_indices, chosen_uavs, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]
   

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result], [agents[i] if i>=0 else None for i in result], [uav_indices[i] if i>=0 else None for i in result], [chosen_uavs[i] if i>=0 else None for i in result]

def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)


def eval_dataset(dataset_path, width, softmax_temp, opts):
    
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model, n_charging_station=None)
    
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        
        device = torch.device("cuda:0" if use_cuda else "cpu")
        
        dataset = model.problem.make_dataset(
            uav_graph_size = opts.uav_graph_size, ugv_graph_size = opts.ugv_graph_size, num_samples=opts.val_size, filename=opts.val_dataset)
        
        
        
        results, batch_ids = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size
    
    
    costs, tours, durations = zip(*results)  # Not really costs since they should be negative
    
    
    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}{}".format(
            
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."
    
    
    save_dataset((results, parallelism), out_file)
    
    folder = r'./Tours_output'
    if opts.DRL_model == 'MF':
        full_path = os.path.join(folder,'actions_RL_sampling_{}_{}_ MF.csv'.format(opts.No_of_UAVs, opts.No_of_UGVs))
    if opts.DRL_model == 'proposed':
        full_path = os.path.join(folder,'actions_RL_sampling_{}_{}.csv'.format(opts.No_of_UAVs, opts.No_of_UGVs))
    if not os.path.exists(folder):
       os.makedirs(folder)
     
    
    batch_ids = pd.Series(batch_ids)
    tours_series = pd.Series(tours)
    costs_series = pd.Series(costs)
    #mission_time_series = pd.Series(mission_time)
   
   
    df = pd.DataFrame({'route': tours_series, 'Costs': costs_series})
    df.to_csv(full_path, index=False)
    
    
    '''folder = r'./Tours_output'
    full_path = os.path.join(folder,'actions_RL.csv')
    if not os.path.exists(folder):
       os.makedirs(folder)
       
    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['route'])
        for tour in tours:
          tour_string = str(tour)
          writer.writerow([tour_string])'''
           
    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size, shuffle = False)
    
    folder = r'./Tours_output'
    full_path = os.path.join(folder,'scenarios.csv')
    
    if not os.path.exists(folder):
       os.makedirs(folder)
    if os.path.isfile(full_path):
      os.remove(full_path)
       
 
    results = []
    batch_ids = []
    for batch_id, batch in enumerate (tqdm(dataloader, disable=opts.no_progress_bar)) :

        
        
         
        depot = batch['depot'].cpu().numpy().tolist()
        uav_loc = batch['uav loc'].cpu().numpy().tolist()
        ugv_loc = batch['ugv loc'].cpu().numpy().tolist()
       


        df = pd.DataFrame({
                'Depot': depot,
                'UAV loc': uav_loc,
                'UGV loc': ugv_loc
                 })

        file_exists = os.path.isfile(full_path)
        non_empty_file = file_exists and os.path.getsize(full_path) > 0
        df.to_csv(full_path, mode='a', index=False, header=not non_empty_file)
        
        
        batch_ids.append(batch)
        batch = move_to(batch, device)
        
        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                
                sequences, costs, agents, uav_indices, chosen_uavs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                cprint('Completed one batch', 'yellow', attrs = ['bold'])
                
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            
            sequences, costs, agents, uav_indices, chosen_uavs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(), agents.cpu().numpy(), uav_indices.cpu().numpy(), chosen_uavs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
            
        duration = time.time() - start
        
        for seq, cost, agent, uav_index, chosen_uav  in zip(sequences, costs, agents, uav_indices, chosen_uavs):
            
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("op", "pctsp"):
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            elif model.problem.NAME == "mugdrl":
                seq = seq.tolist()
                agent = agent.tolist()
                chosen_uav = chosen_uav.tolist()
                uav_index = uav_index.tolist()
                
                action_with_agent = remove_consecutive_duplicates( list(zip(seq, agent, uav_index, chosen_uav)) )[:-1]
                
               
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, action_with_agent, duration))

    return results, batch_ids



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='mugdrl', help="The problem to solve, default 'tsp'")
    parser.add_argument('--val_dataset', type=str, default='data/mugdrl/mugdrl20_test_seed1234.pkl', help='Dataset file to use for validation')
    parser.add_argument('--uav_graph_size', type=int, default=15, help="The size of the uav mission points graph")
    parser.add_argument('--ugv_graph_size', type=int, default=5, help="The size of the ugv mission points graph")
    parser.add_argument('--station_size', type=int, default= env.ugv_mission_points.size(0), help="The number of the charging stations")
    parser.add_argument('--unique_mission_points', type=float, default=env.encoder_mission_space[0], help="The action space")
    parser.add_argument("--datasets", nargs='+', default=['data/mugdrl/mugdrl20_test_seed1234.pkl'], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, default = [256*4], nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '    # default = [1280],
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default = 'sample',
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    if opts.DRL_model == 'MF':
        model = 'MF'
        parser.add_argument('--model', type=str,  default=r'outputs/mugdrl_15/mp_20_{}_uav_{}_ugv_MF/epoch-99.pt'.format(opts.No_of_UAVs, opts.No_of_UGVs)) 
    if opts.DRL_model == 'proposed':
            parser.add_argument('--model', type=str,  default=r'outputs/mugdrl_15/mp_20_{}_uav_{}_ugv/epoch-99.pt'.format(opts.No_of_UAVs, opts.No_of_UGVs)) 
    
      
    
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=256, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--DRL_model', type=str, default=opts.DRL_model, help="The model to use for the DRL agent")
    parser.add_argument('--No_of_UAVs', type=int, default=opts.No_of_UAVs, help="The number of the UAVs")
    parser.add_argument('--No_of_UGVs', type=int, default=opts.No_of_UGVs, help="The number of the UGVs")

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]
    
    
    for width in widths:
        for dataset_path in opts.datasets:
            
            eval_dataset(dataset_path, width, opts.softmax_temperature, opts)


