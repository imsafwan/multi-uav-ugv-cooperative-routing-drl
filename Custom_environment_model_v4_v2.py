# -*- coding: utf-8 -*-


'''
Environment model : V4

v4: one agent at every time step approach

'''


import os
import gym
from gym import spaces
import numpy as np
import networkx as nx
import importlib.util
import pandas as pd
import time
import math
import random
import gym
from gym import spaces
from torch.autograd.profiler import profile, record_function
import torch
import csv
from termcolor import cprint
from options import get_options

opts = get_options()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class scenario(): 

     def __init__(self, csv):
         
         self.df = pd.read_csv(csv)
         self.ugv_data_points = [(self.df['UGV_X'][i], self.df['UGV_Y'][i]) for i in range(len(self.df['UGV_X']))]
         self.uav_fuel = 287700.0
         self.batch_size = 256
         
         #------------- road network graph------------------#
         G = nx.Graph()
         for coord in self.ugv_data_points:
                 G.add_node(coord)
         # Add edges with weights
         for i in range(0,6):
                  weight = int(np.sqrt((self.ugv_data_points[i+1][0] - self.ugv_data_points[i][0])**2 + (self.ugv_data_points[i+1][1] - self.ugv_data_points[i][1])**2)*5280)
                  G.add_edge(self.ugv_data_points[i], self.ugv_data_points[i+1], weight=weight)
         for i in range(7,18):
                  weight = int(np.sqrt((self.ugv_data_points[i+1][0] - self.ugv_data_points[i][0])**2 + (self.ugv_data_points[i+1][1] - self.ugv_data_points[i][1])**2)*5280)
                  G.add_edge(self.ugv_data_points[i], self.ugv_data_points[i+1], weight=weight)
         for i in range(19,29):
                  weight = int(np.sqrt((self.ugv_data_points[i+1][0] - self.ugv_data_points[i][0])**2 + (self.ugv_data_points[i+1][1] - self.ugv_data_points[i][1])**2)*5280)
                  G.add_edge(self.ugv_data_points[i], self.ugv_data_points[i+1], weight=weight)
                  
         nodes_list = list(G.nodes())
         dist_matrix = np.zeros((len(nodes_list), len(nodes_list)))

         # Fill the distance matrix
         for i, node1 in enumerate(nodes_list):
             for j, node2 in enumerate(nodes_list):
                 try:
                     dist_matrix[i][j] = nx.shortest_path_length(G, source=node1, target=node2, weight='weight')
                 except nx.NetworkXNoPath:
                     dist_matrix[i][j] = float('inf')  # or some large number
         
         
         self.dist_matrix = dist_matrix
         self.nodes_list = nodes_list
         


     def dis_along_graph(self, node1, node2): # dis along road
         
        cu_dis, A = self.dist_matrix, self.nodes_list
        A_tensor = torch.tensor(A, dtype=torch.float32).unsqueeze(0).repeat(node1.size(0), 1, 1).to(device)  # [B, N, d]
        match1 = (A_tensor == node1.unsqueeze(1)).all(-1)  # [B, N]
        valid1, ix1 = match1.max(1)  # [B]
        match2 = (A_tensor == node2.unsqueeze(1)).all(-1)  # [B, N]
        valid2, ix2 = match2.max(1)  # [B]
        cu_dis_tensor = torch.tensor(cu_dis, dtype=torch.float32).to(device)  # [N, N]
        distances_tensor = torch.zeros(node1.size(0), dtype=torch.float32, device=device)
        valid_indices = valid1 & valid2
        distances_tensor[valid_indices] = cu_dis_tensor[ix1[valid_indices], ix2[valid_indices]]
        return distances_tensor
                       
                        
                  
                        
                       

class UAV_UGV_Env(gym.Env):
    
    
    def __init__(self):
        
         '''-------------------------------- parameter initilaization -----------------------------------''' 
        
         self.scene = scenario('ugv_road.csv')
         self.batch_size = self.scene.batch_size
         self.uav_speed = 33    # ft/sec
         self.ugv_speed = 15    # ft/sec
         self.prv_action = [None] * self.batch_size                            #initialize
         self.uav_fuel_constant = 287700.0 # J
         
         '''------------------------------ random initialization -------------------------------------'''
          
         self.uav_mission_points = torch.rand( 100, 2, device = device)                                                     # tensor [N, d]
         self.ugv_mission_points = torch.rand( 100, 2, device = device)                                                     # tensor [N, d] 
         self.all_mission_points = torch.cat([self.uav_mission_points, self.ugv_mission_points], dim=0)                     # tensor [N, d] 
         self.unique_mission_points = torch.cat([self.uav_mission_points, self.ugv_mission_points], dim=0)                  # tensor [1, N_unique, d]
         self.encoder_mission_space = torch.rand(self.batch_size, 100, 3, device = device) 
         self.uav_fuel_limit = torch.full((self.batch_size,), 287700.0, dtype=torch.float32, device = device)               # tensor [B]
         self.uav_fuel = torch.full((self.batch_size,), 287700.0, dtype=torch.float32, device = device)                     # tensor [B]
         self.uav_pos_coord = torch.stack([self.uav_mission_points[-1]] * self.batch_size)                                  # tensor [B, d]
         self.ugv_pos_coord = torch.stack([self.ugv_mission_points[0]] * self.batch_size)                                   # tensor [B, d]
         self.refuel_stop = torch.stack([self.ugv_mission_points[0]] * self.batch_size)                                     # [B,d]
         self.depot_index = torch.stack([self.uav_mission_points[-1]] * self.batch_size)   
         self.mission_status = torch.zeros(self.batch_size, len(self.unique_mission_points[0]), device = device)            # tensor [B, N]
         self.mission_time_elapsed = torch.zeros(self.batch_size, device = device)                                          # tensor [B, N] 
         self.infeasibility = torch.zeros(self.batch_size, device = device).bool()                                          # tensor [B]
         self.recharging_location = torch.zeros(self.batch_size, device = device).bool()                                    # tensor [B]
         self.refuel_stop_time = torch.zeros(self.batch_size, device = device)                                              # tensor [B]
         
         

         self.No_of_UAVs = opts.No_of_UAVs
         self.No_of_UGVs = opts.No_of_UGVs
         
    
         '''-------------------------------------------------------------------'''
         
         
         

    def calculate_distances(self, p1, p2):
         return torch.norm((p1 - p2), dim=-1)*5280 
    
    def ecuclidean_dis(self, a, b):
       a = a.unsqueeze(1)
       b = b.unsqueeze(1)
       diff = a - b
       distance = torch.norm(diff, dim=2) * 5280
       return distance.permute(1,0)[0]
   
    
        
    def step(self, actions, step):

        batch_size = self.batch_size
        batch_indices = torch.arange(batch_size, device = 'cuda') 
        assert batch_size == actions.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        action_indices = actions
        self.prv_action = actions

        #region
        #------------ extract the action coordinates from the action index ---------- #
        
        is_refuel_stop = action_indices < len(self.ugv_mission_points[0])       # action : recharging action
        is_refuel_stop = is_refuel_stop.unsqueeze(-1)
        is_visit_point = (action_indices >= len(self.ugv_mission_points[0]))    # action: visiting mission points
        is_visit_point = is_visit_point.unsqueeze(-1)


        if self.encoder_mission_space.shape[0] != batch_size:
               self.encoder_mission_space = self.encoder_mission_space.repeat(batch_size, 1, 1)
                  
        temp_refuel_points = self.encoder_mission_space[:, :, :2].clone() 
        refuel_coords = torch.where(is_refuel_stop, temp_refuel_points[batch_indices, action_indices], torch.zeros(action_indices.shape[0], 2, device = device))
        visit_coords = torch.where(is_visit_point, temp_refuel_points[batch_indices, action_indices], torch.zeros(action_indices.shape[0], 2, device = device))
        action_coords = refuel_coords + visit_coords                            # action coordinates [B, 2]
        
        # ------- unique action indices ------ #
        
        expanded_action_coords = action_coords.unsqueeze(1)
        matches = torch.all(self.unique_mission_points == expanded_action_coords, dim=-1)  # Shape: [B, N]
        matches_int = matches.int()
        unique_action_indices = torch.argmax(matches_int, dim=-1)                          # indices of action shape [B]
        assert batch_size == len(unique_action_indices)
        
        
        is_refuel_stop = is_refuel_stop.any(dim=1)
        is_visit_point = is_visit_point.any(dim=1)
        #endregion
        
        
        # ------- effect of actions --------- #
        #region
        
        #                --------------> effect on those instances where uav is the agent -------------------- >
        uav_agent_mask = (self.agent_type == 0)  # Shape [B]
        fuel_depleted_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.uav_fuel.device)  # Shape: [B]
        
        if uav_agent_mask.any():  # Apply only to instances where UAV is the agent # If any instance has UAV as the agent
        
            uav_batch_indices = batch_indices[uav_agent_mask]
            uav_agent_indices = self.uav_agent_index[uav_agent_mask]
            
            self.uav_index_last_action_taken[uav_batch_indices] = self.uav_agent_index[uav_agent_mask].long()
            
            # Calculate step elapsed times only for UAV instances
            step_elapsed_times = torch.norm((action_coords[uav_agent_mask] - self.uav_pos_coord[uav_batch_indices, uav_agent_indices]), dim=1) * 5280 / self.uav_speed  # [num_uav_instances]
            
            # Update UAV fuel only for the selected UAVs in the filtered batch instances
            self.uav_fuel[uav_batch_indices, uav_agent_indices] -= (step_elapsed_times * 198).int()  # [num_uav_instances]
            
            fuel_depleted_mask[uav_agent_mask] = (self.uav_fuel[uav_agent_mask] < 0).any(dim=1)  # Shape: [num_uav_instances] 
            
            
            
            self.mission_time_elapsed_local_uav[uav_batch_indices, uav_agent_indices] += step_elapsed_times     # [ B, N_uav]           # time elapsed in uav's local time frame    
            self.uav_pos_indices[uav_batch_indices, uav_agent_indices]  = actions[uav_batch_indices]    
            self.uav_pos_coord[uav_batch_indices, uav_agent_indices] = action_coords[uav_batch_indices]  # [ B, N_uav, 2]

            recharge_mask = is_refuel_stop[uav_agent_mask]           # Shape: [num_uav_instances]
            masked_batch_indices = uav_batch_indices[recharge_mask]  # Shape: [num_recharging_instances]
            masked_agent_indices = uav_agent_indices[recharge_mask]  # Shape: [num_recharging_instances]
            
            # Create a tensor filled with the full fuel level value
            full_fuel = torch.full((masked_batch_indices.size(0),), 287700.0, dtype=torch.float32, device=device)
            
            # Update the fuel to full capacity for the specific UAVs that are refueling
            self.uav_fuel[masked_batch_indices, masked_agent_indices] = full_fuel   # [B, N_uav]     #   in those instances where after recharging fuel level is full  
            
            # Update recharging locations for UAVs that visited recharge points
            self.recharging_location[uav_batch_indices, uav_agent_indices] = torch.zeros(
                uav_batch_indices.size(0), dtype=torch.bool, device=device
            )
            
            # Update recharging locations for UAVs that are currently refueling
            self.recharging_location[masked_batch_indices, masked_agent_indices] = torch.ones(
                masked_batch_indices.size(0), dtype=torch.bool, device=device
            )
        
        #          ----------------> effect on those instances where ugv is the agent -------------------- >
        ugv_agent_mask = (self.agent_type == 1)  # Shape [B]
        if ugv_agent_mask.any(): 
        
                # Filtered batch indices where UGV is the agent
                ugv_batch_indices = batch_indices[ugv_agent_mask]  # Shape: [num_ugv_instances]
                
                
                fuel_depleted_mask[ugv_agent_mask] = (self.uav_fuel[ugv_agent_mask] < 0).any(dim=1)  # Shape: [num_ugv_instances]
                

                # Calculate distance-based time for UGV only for filtered instances
                masked_distance_time = (self.scene.dis_along_graph(self.ugv_pos_coord[ugv_agent_mask, self.ugv_agent_index[ugv_agent_mask], :], action_coords[ugv_agent_mask]) / self.ugv_speed)  #Shape: [num_recharging_instances]
                
                self.ugv_local_time[ugv_batch_indices, self.ugv_agent_index[ugv_batch_indices]] += masked_distance_time # Shape: [num_recharging_instances]
                    
                    
                chosen_uav_index = self.select_uav_for_recharge[ugv_agent_mask,self.ugv_agent_index[ugv_agent_mask].long()].long()  # Shape: [num_ugv_instances]  
                
                uav_landing_time = self.uavs_landing_times[ugv_batch_indices, chosen_uav_index]  # Shape: [num_ugv_instances]
            
                # Compute waiting time for UGV instances
                waiting_time = torch.where(
                    is_refuel_stop[ugv_batch_indices],  # Condition: Shape [num_ugv_instances]
                    (self.ugv_local_time[ugv_batch_indices, self.ugv_agent_index[ugv_batch_indices]] - uav_landing_time),  # Shape: [num_ugv_instances]
                    torch.zeros_like(uav_landing_time, device=device)  # Shape: [num_ugv_instances]
                )
                waiting_time = torch.clamp(waiting_time, min=0)  # Ensure no negative waiting time; Shape: [num_ugv_instances]
                
                
                
               
                zero_waiting_time_mask = (waiting_time == 0)  # Boolean mask for instances with waiting_time == 0
                if zero_waiting_time_mask.any():
                        self.ugv_local_time[
                            ugv_batch_indices[zero_waiting_time_mask],
                            self.ugv_agent_index[ugv_batch_indices[zero_waiting_time_mask]]
                        ] = self.uavs_landing_times[
                            ugv_batch_indices[zero_waiting_time_mask],
                            chosen_uav_index[zero_waiting_time_mask]
                        ]  
                
                # Calculate recharge time based on recharging condition only for UGV agent instances
                recharge_time = torch.where(
                    is_refuel_stop[ugv_batch_indices],  # Shape: [num_ugv_instances]
                    torch.tensor(900.0, device=device),  # Scalar recharge time
                    torch.zeros_like(ugv_batch_indices).float()  # Shape: [num_ugv_instances]
                )
                
                
                self.ugv_local_time[ugv_batch_indices,self.ugv_agent_index[ugv_batch_indices]] += recharge_time[is_refuel_stop[ugv_batch_indices]]
                
                self.mission_time_elapsed_local_uav[ugv_batch_indices, chosen_uav_index] += waiting_time + recharge_time
                
                self.uavs_recharge_complete_flag[ugv_batch_indices, chosen_uav_index] = 1.0

                self.ugv_pos_coord[ugv_batch_indices,self.ugv_agent_index[ugv_batch_indices],:] = action_coords[ugv_batch_indices]

        
        #endregion
        
        # Assign infeasible penalties for batch instances
        infeasible_penalties = torch.where(
            fuel_depleted_mask, 
            torch.ones(batch_size, device=device), 
            torch.zeros(batch_size, device=device)
        )  # Shape: [B]
  
         
        # ----------- update state --------- #
        
        self.mission_status[batch_indices, unique_action_indices] = 1                              
        
        
        # <------------------------------------------ Change of agent ----------------------------------------> #
        #region

        #if uav is current agent
        if uav_agent_mask.any():  # If any instance in the batch is UAV
            recharge_mask = uav_agent_mask & is_refuel_stop   # Boolean tensor [B], True where recharging is needed and agent is uav
            if recharge_mask.any():
                batch_indices = torch.arange(self.batch_size, device=device)[recharge_mask]  # [num_recharges]
                uav_indices = self.uav_agent_index[recharge_mask]  # UAV indices for batches needing recharging
                

                self.uavs_landing_locs_node[batch_indices, uav_indices] = actions[batch_indices].float()
                self.uavs_landing_locs[batch_indices, uav_indices] = action_coords[batch_indices]
                self.uavs_landing_times[batch_indices, uav_indices] = self.mission_time_elapsed_local_uav[batch_indices, uav_indices]
                self.uavs_landing_complete_flag[batch_indices, uav_indices] = True
                

            all_uavs_landed = self.uavs_landing_complete_flag.all(dim=1)  # Shape [B], True if all UAVs have landed per batch instance
            

            self.agent_type[uav_agent_mask & all_uavs_landed] = 1


            #---------- allocation of UAvs to UGV -----------------#

            allocation_mask = uav_agent_mask & all_uavs_landed  # [B]
           
            filtered_ugv_local_time = self.ugv_local_time[allocation_mask]  # [num_instances, N_ugv]
            filtered_ugv_local_coord = self.ugv_pos_coord[allocation_mask]  # [num_instances, N_ugv, d]
            filtered_uav_local_time = self.uavs_landing_times[allocation_mask]  # [num_instances, N_uav]
            filtered_uav_local_coord = self.uav_pos_coord[allocation_mask]  # [num_instances, N_uav, d]

            
            coord_diff = filtered_ugv_local_coord.unsqueeze(2) - filtered_uav_local_coord.unsqueeze(1)  # [num_instances, N_ugv, N_uav, d]
            euclidean_distances = torch.norm(coord_diff, dim=-1)  # [num_instances, N_ugv, N_uav]

            
            time_based_distances = filtered_ugv_local_time.unsqueeze(2) + filtered_uav_local_time.unsqueeze(1)  # [num_instances, N_ugv, N_uav]
            combined_metric = euclidean_distances + time_based_distances  # [num_instances, N_ugv, N_uav]

            
            assigned_ugvs = torch.argmin(combined_metric, dim=1)  # [num_instances, N_uav]
            
            allocation_tensor = torch.zeros_like(combined_metric, dtype=torch.bool)  # [num_instances, N_ugv, N_uav]
            allocation_tensor.scatter_(dim=1, index=assigned_ugvs.unsqueeze(1).expand(-1, filtered_ugv_local_time.size(1), -1), value=True)

            
            self.allocation_tensor[allocation_mask] = allocation_tensor  # [B, N_ugv, N_uav]

            
            # -----------------> ugv swap order #

            batch_size, N_ugv = self.ugv_local_time.shape  # [B, N_ugv]
            device = self.ugv_local_time.device
            allocated_uavs_mask = self.allocation_tensor.sum(dim=-1) > 0  # [B, N_ugv], True if UGV has at least one allocated UAV
            masked_local_time = self.ugv_local_time.clone()
            masked_local_time[~allocated_uavs_mask] = float('inf')  # [B, N_ugv]
            sorted_local_times, sorted_indices = torch.sort(masked_local_time, dim=-1)  # [B, N_ugv]
            batch_indices = torch.nonzero(allocation_mask, as_tuple=False).squeeze(-1)  # Indices where allocation_mask is true
            if batch_indices.numel() > 0:  # Ensure there are valid instances
                self.switch_map[batch_indices] = sorted_indices[batch_indices]  # [Valid Batches, N_ugv]
                self.ugv_agent_index[batch_indices] = self.switch_map[batch_indices, 0]  # First element in switch map

            #---------------------------------------------------#

            not_all_landed_mask = uav_agent_mask & ~all_uavs_landed  # Select instances where agent is UAV and not all UAVs have landed

            if not_all_landed_mask.any():
                
                valid_uav_mask = ~self.uavs_landing_complete_flag.bool()  # Shape: [B, N_uav]
                next_uav_indices = (self.uav_index_last_action_taken + 1) % self.uavs_landing_complete_flag.size(1)  # Shape: [B]
                candidate_indices = (next_uav_indices.unsqueeze(1) + torch.arange(self.uavs_landing_complete_flag.size(1), device=valid_uav_mask.device)) % self.uavs_landing_complete_flag.size(1)  # Shape: [B, N_uav]
                valid_candidates = valid_uav_mask.gather(1, candidate_indices)  # Shape: [B, N_uav]
                valid_first_idx = torch.argmax(valid_candidates.float(), dim=-1)  # Shape: [B] (cast to float)
                chosen_uav_indices = candidate_indices[torch.arange(candidate_indices.size(0), device=candidate_indices.device), valid_first_idx]  # Shape: [B]
                self.uav_agent_index[not_all_landed_mask] = chosen_uav_indices[not_all_landed_mask]


            
            

        #if ugv is current agent
        if ugv_agent_mask.any():
            
            all_uavs_recharged = self.uavs_recharge_complete_flag.all(dim=1)  # Shape [B], True if all UAVs have recharged per batch instance
            
            
            exceeded_indices = torch.where(torch.all(self.mission_status == 1, dim=1))[0]  # Shape: [num_exceeded_instances]
            exceeded_mask = torch.zeros_like(all_uavs_recharged, dtype=torch.bool)  # Shape: [B]
            exceeded_mask[exceeded_indices] = True  # Mark exceeded indices as True
            
              
            # -------- Identify instances where the mission is not completed ---------#
            
            combined_mask = ugv_agent_mask & all_uavs_recharged & ~exceeded_mask

            self.agent_type[combined_mask] = 0
            recharge_mask1 = (is_refuel_stop == True) & combined_mask   # Select instances needing recharging with UAVs still in flight
            if recharge_mask1.any():
                min_time_indices = torch.argmin(self.mission_time_elapsed_local_uav, dim=1)  # Shape [B]
                self.uav_agent_index[recharge_mask1] = min_time_indices[recharge_mask1]
                
                
            self.uavs_recharge_complete_flag[combined_mask] = torch.zeros((self.batch_size, self.uavs_recharge_complete_flag.size(1)), device = device)[combined_mask]  # tensor [B, N_uav] 
            self.uavs_landing_complete_flag[combined_mask] = torch.zeros((self.batch_size, self.uavs_recharge_complete_flag.size(1)), device = device)[combined_mask]  # tensor [B, N_uav] 
            
            # ----- Instances where UGV agent would be changed ----- #

            
            allocated_uavs = self.allocation_tensor[torch.arange(batch_size, device=self.allocation_tensor.device), self.ugv_agent_index]  # [B, N_uav]

            
            recharged_allocated_uavs = allocated_uavs & self.uavs_recharge_complete_flag.bool()  # [B, N_uav]

            
            allocated_uavs_recharged = torch.where(
                allocated_uavs.any(dim=-1),  # If there are any allocated UAVs in a batch
                recharged_allocated_uavs.all(dim=-1),  # Check if all allocated UAVs in the batch are recharged
                torch.tensor(True, device=allocated_uavs.device)  # Default to True if no UAVs are allocated
            )  # [B]

            ugv_agent_change_mask = ugv_agent_mask & ~all_uavs_recharged & allocated_uavs_recharged  # [B]
            if ugv_agent_change_mask.any():
                current_ugv_indices = self.ugv_agent_index[ugv_agent_change_mask]  # [num_ugv_changes]
                next_ugv_indices = self.switch_map[
                    torch.arange(batch_size, device=self.ugv_agent_index.device)[ugv_agent_change_mask],  # Batch indices
                    current_ugv_indices  # Current UGV indices
                ]
                print(self.switch_map[9])
                
                self.ugv_agent_index[ugv_agent_change_mask] = next_ugv_indices
                print(self.ugv_agent_index[9])

            # -------- Identify instances where the mission is complete ---------#
            mission_complete_mask = ugv_agent_mask & all_uavs_recharged & exceeded_mask
            
            newly_completed_instances = mission_complete_mask & (self.mission_complete_flag == 0.0)
            
            self.mission_complete_flag[mission_complete_mask] = 1.0
            
            if newly_completed_instances.any():
                self.total_mission_time[newly_completed_instances] = self.mission_time_elapsed_local_uav.max(dim=1)[0][newly_completed_instances]
                
        #endregion

        # -------- call state ------- #
        #region
        
        states = self.get_observation() 
        exceeded_indices = torch.where(self.mission_complete_flag == 1)[0]
        
        if exceeded_indices.numel() > 0:
            shrink = exceeded_indices.shape[0] 
        else :
            shrink = 0
        
        
        step_condition = torch.tensor([step > 250], dtype=torch.bool, device = device)
        shrink_condition = torch.tensor([shrink > int(batch_size*0.90)], dtype=torch.bool, device = device)
        dones = torch.logical_or((self.mission_complete_flag ==1), self.infeasibility)
        dones = torch.logical_or(dones, step_condition)
        dones = dones.to(device)
        
        if self.mission_complete_flag[9] == 1:
               cprint('<-----------------All mission points visited for the first batch instance--------------------------->', 'yellow', attrs = ['bold'])
               
               
        if dones.all():
           self.mission_time_elapsed , _ = self.mission_time_elapsed_local_uav.max(dim=1)
           unfinished_batch_indices = torch.where(self.mission_complete_flag != 1)[0]
           self.total_mission_time[unfinished_batch_indices] = self.mission_time_elapsed_local_uav.max(dim=1)[0][unfinished_batch_indices] + (500*60) # this acts like a penalty for those cases where all mission points are not visited
           
        info = {} 
        #endregion 

        
        return states, dones, info
    


    def reset(self, input):
        
            No_of_UAVs = self.No_of_UAVs
            No_of_UGVs = self.No_of_UGVs
           
        
            self.uav_mission_points =  input['uav loc']          # tensor [B, N, d]
            self.batch_size = self.uav_mission_points.size(0)
            self.ugv_mission_points=   input ['ugv loc']           # tensor [B, N, d]
            self.unique_mission_points = input['unique loc']     # tensor [B, N, d]
            self.encoder_mission_space = input['encoder space']  # tensor [B, N, d]

            depot_ix = input['depot']              # [B, 1]
            self.depot_index = depot_ix
            self.uav_pos_coord = (input['ugv loc'][torch.arange(self.batch_size).unsqueeze(1),depot_ix].squeeze(1)).unsqueeze(1).repeat(1, No_of_UAVs, 1)      # tensor [B, N_uav, d]
            
            self.ugv_pos_coord = input['ugv loc'][torch.arange(self.batch_size).unsqueeze(1),depot_ix].squeeze(1).unsqueeze(1).repeat(1, No_of_UGVs, 1)       # tensor [B, N_ugv, d]
            
            self.uav_pos_indices = depot_ix.clone().repeat(1, No_of_UAVs)               # tensor [B,1] 
            self.uav_fuel_limit = torch.full((self.batch_size, No_of_UAVs), 287700.0, dtype=torch.float32, device = device)               # tensor [B, N_uav] 
            self.uav_fuel = torch.full((self.batch_size, No_of_UAVs), 287700.0, dtype=torch.float32, device = device)                     # tensor [B, N_uav] 
            
    
            self.mission_status = torch.zeros_like(self.unique_mission_points, device = device)[:,:,0]  # tensor [B, N]
            self.mission_time_elapsed = torch.zeros(self.batch_size, device = device)                   # tensor [B] 
            self.infeasibility = torch.zeros(self.batch_size, device = device).bool()                   # tensor [B]
            self.recharging_location = torch.zeros((self.batch_size, No_of_UAVs), device = device).bool()             # tensor [B, N_uav]
            self.last_score = torch.zeros_like(self.unique_mission_points, device = device)[:,:,0]         # tensor [B, N]
            self.prv_action = [None] * self.batch_size                                               #initialize
            
            self.prv_action_UGV = torch.full((self.batch_size,), -1, dtype=torch.long, device="cuda")

            
            
            self.mission_time_elapsed_local_uav = torch.zeros((self.batch_size, No_of_UAVs), device = device)   # tensor [B, N_uav] 
            
            
            self.uav_agent_index = torch.zeros(self.batch_size, dtype=torch.long, device = device)       # tensor [B]
            self.ugv_agent_index = torch.zeros(self.batch_size, dtype=torch.long, device = device)       # tensor [B]
            
            self.uavs_landing_complete_flag = torch.zeros((self.batch_size, No_of_UAVs), device = device)  # tensor [B, N_uav] , this flag indicates that the uav has landed and we can use this to change the uav agent selection
            self.uavs_recharge_complete_flag = torch.zeros((self.batch_size, No_of_UAVs), device = device)  # tensor [B, N_uav] , this flag indicates that the uav's recharge is done and we can use this for the ugv to choose next uav to select for recharging
            
            self.agent_type = torch.zeros(self.batch_size, device = device)                   # tensor [B] 
            
            
            self.uavs_landing_locs = torch.zeros((self.batch_size, No_of_UAVs, 2), device = device)       #initillay placeholder            # tensor [B, N_uav, 128(d)]
            self.uavs_landing_locs_node = torch.zeros((self.batch_size, No_of_UAVs), device = device)       #initillay placeholder            # tensor [B, N_uav]
            self.uavs_landing_times = torch.zeros((self.batch_size, No_of_UAVs), device = device)           #initially placeholder        # tensor [B, N_uav]
            
            self.ugv_local_time = torch.zeros((self.batch_size, No_of_UGVs), device = device)                   # tensor [B, N_ugv] 

            

            # selected_uav now contains the UAV corresponding to the UGV index for each batch

            
            self.select_uav_for_recharge = torch.zeros((self.batch_size, No_of_UGVs), dtype=torch.float32, device = device)       #initillay placeholder            # tensor [B, N_ugv]
            
            
            self.mission_complete_flag = torch.zeros((self.batch_size), dtype=torch.float32, device = device)       #initillay placeholder            # tensor [B]
            self.total_mission_time = torch.zeros((self.batch_size), dtype=torch.float32, device = device)       #initillay placeholder            # tensor [B]
            
            
            self.allocation_tensor = torch.zeros((self.batch_size, No_of_UGVs, No_of_UAVs), dtype=torch.bool, device=device)  # [B, N_ugv, N_uav]
            
            self.switch_map = torch.arange(No_of_UGVs, device=device).roll(-1).repeat(self.batch_size, 1)  # Shape: [B, N_ugv]
            self.switch_map[:, -1] = -1

            self.uav_index_last_action_taken = torch.zeros((self.batch_size), device = device).long()       #initillay placeholder            # tensor [B]
             
             
            state = self.get_observation()
            
            return state
        
    
    
    
    def get_observation(self):
       return (self.uav_pos_indices, self.uav_fuel, self.mission_status)

            
        
    
    def feasible_action(self):
           # filter out invalid action points
           
           
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           batch_size = self.batch_size
           N = self.encoder_mission_space.shape[1]
           assert batch_size == self.uav_pos_coord.size(0)
    
           if self.ugv_mission_points.dim() == 2:
                     self.ugv_mission_points = self.ugv_mission_points.unsqueeze(0).repeat(batch_size, 1, 1)   # [B, N, d ]
                     
           if self.uav_mission_points.dim() == 2:
                  self.uav_mission_points = self.uav_mission_points.unsqueeze(0).repeat(batch_size, 1, 1)      # [B, N, d ]
           
           
           
           
           batch_indices = torch.arange(self.uav_pos_coord.size(0), dtype=torch.long, device=self.uav_pos_coord.device)
    
           ugv_distances = torch.stack([self.ecuclidean_dis(self.uav_pos_coord[batch_indices, self.uav_agent_index], point) for point in self.ugv_mission_points.permute(1, 0, 2)])
           ugv_distances = ugv_distances.permute(1, 0) / self.uav_speed  # Shape: [B, N]
    
           uav_distances = torch.stack([self.ecuclidean_dis(self.uav_pos_coord[batch_indices, self.uav_agent_index], point) for point in self.unique_mission_points.permute(1, 0, 2)])  
           uav_distances = uav_distances.permute(1,0) / self.uav_speed  # [B, N]
           
           
    
           #--------feasible region-----------#
           
           

           
           action_positions = self.encoder_mission_space[:,:,:2]
           expanded_uav_pos = self.uav_pos_coord[batch_indices, self.uav_agent_index].unsqueeze(1).expand(-1, N, -1)
           dist_to_uav_nodes = self.calculate_distances(expanded_uav_pos, action_positions)
           action_positions_expanded = action_positions.unsqueeze(2)      # Shape: [B, N, 1, 2]
           ugv_positions_expanded = self.ugv_mission_points.unsqueeze(1)  # Shape: [B, 1, N, 2]
           dist_to_ugv_nodes = self.calculate_distances(action_positions_expanded, ugv_positions_expanded)  # Shape: [B, N, N]
           min_dist_to_ugv,  min_ugv_indices = torch.min(dist_to_ugv_nodes, dim=2) #[0]  # Shape: [B, N]
           total_distance = dist_to_uav_nodes + min_dist_to_ugv
           return_feasible_region = (((total_distance/ self.uav_speed)*198).int() <= (self.uav_fuel[batch_indices, self.uav_agent_index].unsqueeze(1)).int()).to(device) 
           
           # return feasible region : those mission points where if the uav goes to visit will not be able to come to any refule stop for recharging
           
           rows_with_all_zeros = ~torch.any(return_feasible_region, dim=1)
           
           if torch.any(rows_with_all_zeros):
                rows_with_all_zeros_indices = torch.where(rows_with_all_zeros)[0]
                print("Indices of rows with all zeros:", rows_with_all_zeros_indices.tolist())
                raise RuntimeError("Execution stopped because there are rows with all zeros.")
                
           
           repeated_feasible_region = (self.mission_status != 1).to(device).int()       
           tensor1 = torch.ones([batch_size, len(self.ugv_mission_points[0])], device=device)
           repeated_feasible_region = torch.cat((tensor1, repeated_feasible_region), dim=1)  # [ B, N]

           # return feasible region : those mission points that are already visited
           
           
           # -------- create mask -----------#
                                                                         
        
           feasible_mask = return_feasible_region * repeated_feasible_region
           assert torch.any(feasible_mask, dim=1).all(), "A row with all zeros (False) found in the mask."
           a = feasible_mask.clone()
           
           
           ugv_indices = torch.where(~self.recharging_location)[0].to(device)
           ugv_slice = slice(0, len(self.ugv_mission_points[0]))
           feasible_mask[ugv_indices, ugv_slice] = torch.zeros(batch_size, len(self.ugv_mission_points[0]), device = device)[ugv_indices]   # makes sure after recharging visiting action is performed
           
           rows_with_all_zeros = ~torch.any(feasible_mask, dim=1)
           
           if torch.any(rows_with_all_zeros):
                 rows_with_all_zeros_indices = torch.where(rows_with_all_zeros)[0]
                 feasible_mask[rows_with_all_zeros_indices] = a.clone()[rows_with_all_zeros_indices]
            
            # if there is no mission points available to go, then visit already visited points
    
           assert torch.any(feasible_mask, dim=1).all(), "A row with all zeros (False) found in the mask."
           
           
           
           
           if self.prv_action != [None]* self.batch_size :
               f_mask = self.prv_action < len(self.ugv_mission_points[0])                       # same refuel action = same visit action is avoided
               indices = torch.arange(self.batch_size, device=device)[f_mask]
               feasible_mask[indices, self.prv_action[indices] + len(self.ugv_mission_points[0])] = 0
           
           exceeded_indices = torch.where(torch.all(self.mission_status == 1, dim=1))[0]
           if exceeded_indices.numel() > 0:
             feasible_mask[exceeded_indices, :] = 0 
             feasible_mask[exceeded_indices, 0:len(self.ugv_mission_points[0])] = 1
       
           assert torch.any(feasible_mask, dim=1).all(), "A row with all zeros (False) found in the mask."


           
             
           return feasible_mask

