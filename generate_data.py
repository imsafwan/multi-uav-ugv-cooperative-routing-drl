import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset
from  Custom_environment_model_v1 import UAV_UGV_Env
from scenario_generator_v2 import scenario_gen
env = UAV_UGV_Env()
import  torch

def generate_mcsrp_data(uav_graph_size, ugv_graph_size,  num_samples):
    
    
    data = []
    for i in range(num_samples):
        
        uav_loc, ugv_loc = scenario_gen(uav_graph_size, ugv_graph_size)
        
        
        uav_mission_points = torch.from_numpy(uav_loc).float() # tensor [N_uav, 2]
        ugv_mission_points = torch.from_numpy(ugv_loc).float() # tensor [N_ugv, 2]
        
        all_mission_points = torch.cat([ugv_mission_points, uav_mission_points], dim=0)      # tensor [N_all, 2] 
        unique_mission_points = torch.cat([ugv_mission_points,uav_mission_points], dim=0) 
        #unique_mission_points = torch.tensor(pd.DataFrame(unique_mission_points.cpu().numpy()).drop_duplicates(keep='first').values, device=unique_mission_points.device, dtype=unique_mission_points.dtype)
        # [ N_unique, all ]

        ones = torch.ones(unique_mission_points.size(0), 1, device = uav_mission_points.device )
        zeros = torch.zeros(ugv_mission_points.size(0), 1, device = ugv_mission_points.device)
        unique_mission_points_with_ones = torch.cat([unique_mission_points, ones], -1)  # [ N_unique, 3 ]
        ugv_mission_points_with_zeros = torch.cat([ugv_mission_points, zeros], -1)  # [ N_ugv, 3]

        encoder_mission_space = torch.cat([ugv_mission_points_with_zeros, unique_mission_points_with_ones], 0)  # [ N_all, 3]
       

        depot_ix = torch.randint(0, ugv_mission_points.size(0), (1,), device=ugv_mission_points.device)   # [ 1]    
        
        data.append(
            {
                'depot': depot_ix,
                'uav loc': uav_mission_points,
                'ugv loc': ugv_mission_points,
                'encoder space': encoder_mission_space,
                'unique loc':  unique_mission_points
                
                
            }
            
        )

        
        
    
    return data
    
    
def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_op_data(dataset_size, op_size, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[op_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='mcsrp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--uav_graph_size', type=int, default= 15, help="The size of the problem graph")
    parser.add_argument('--ugv_graph_size', type=int, default= 5, help="No of charging station")
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--station_size', type=int, default= env.ugv_mission_points.size(0), help="The number of the charging stations")
    parser.add_argument('--unique_mission_points', type=float, default=env.encoder_mission_space[0], help="The action space")
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, default = 'test', help="Name to identify dataset")
    #parser.add_argument("--problem", type=str, default='all',
    #                    help="Problem, 'tsp', 'vrp', 'pctsp' or 'op_const', 'op_unif' or 'op_dist'"
    #                         " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=25600, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'vrp': [None],
        'pctsp': [None],
        'mcsrp':[None],
        'op': ['const', 'unif', 'dist']
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }
    
    
    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes: 

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'vrp':
                    dataset = generate_vrp_data(
                        opts.dataset_size, graph_size)
                elif problem == 'pctsp':
                    dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                elif problem == "op":
                    dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution)
                elif problem == 'mcsrp':
                    dataset = generate_mcsrp_data(opts.uav_graph_size,  opts.ugv_graph_size,  opts.dataset_size)
                    
                else:
                    assert False, "Unknown problem: {}".format(problem)

                              
                print(dataset[0])

                save_dataset(dataset, filename)
