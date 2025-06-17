from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.mugdrl.state_mugdrl import Statemugdrl
from .scenario_generator_v2 import scenario_gen
from utils.beam_search import beam_search
import numpy as np
import pandas as pd



 
class mugdrl(object):
    NAME = 'mugdrl'  

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #         torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #         pi.data.sort(1)[0]
        # ).all(), "Invalid tour"

        # Gather dataset in order of tour
        loc = dataset['loc']
        d = loc.gather(1, pi[..., None].expand(*pi.size(), loc.size(-1)))
       
        # Check that tours are valid
        rolled_d = d.roll(dims=1, shifts=1)

        lengths = ((d-rolled_d)**2).sum(2).sqrt()

        cum_length = torch.cumsum(lengths, dim=1)

        
        idx = (pi >= dataset['p_size'][:, None])

        cum_length[idx] = 0

        sorted_cum_length, _ = cum_length.sort(axis=1)

        rolled_sorted_cum_length = sorted_cum_length.roll(dims=1, shifts=1)
        diff_mat = sorted_cum_length - rolled_sorted_cum_length
        diff_mat[diff_mat < 0] = 0
        makespans, _ = torch.max(diff_mat, dim=1)

        assert (makespans <= dataset['max_length']).all(), print(makespans[makespans > dataset['max_length']])
        # cost = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - loc[:, 0]).norm(p=2, dim=1)

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
                                                                                                                                                                                                                                                                                                                                                                                      
        return cum_length[:, -1], None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return mugdrlDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return Statemugdrl.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = mugdrl.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(uav_graph_size, ugv_graph_size):
    # size: the number of targets
    # p_size: the number of the charging stations including the depot

    MAX_LENGTHS = {
        20: 3.,
        50: 3.,
        100: 3.
    }
    

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
    return {
        'depot': depot_ix,
        'uav loc': uav_mission_points,
        'ugv loc': ugv_mission_points,
        'encoder space': encoder_mission_space,
        'unique loc':  unique_mission_points
        
        
    }


class mugdrlDataset(Dataset):

    def __init__(self, filename=None, uav_graph_size=50, ugv_graph_size = 10, num_samples=1000000, offset=0):
        super(mugdrlDataset, self).__init__()
        
        self.data_set = []
        if filename is not None:
            
            assert os.path.splitext(filename)[1] == '.pkl'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
                self.data = [
                    {
                        'depot': data_ix['depot'],
                        'uav loc': data_ix['uav loc'],
                        'ugv loc': data_ix['ugv loc'],
                        'encoder space': data_ix['encoder space'],
                        'unique loc':  data_ix['unique loc']
                        
                        
                    }
                    for data_ix in (data[offset:offset + num_samples])
                ]
                
                
        else:
            self.data = [
                generate_instance(uav_graph_size, ugv_graph_size)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
