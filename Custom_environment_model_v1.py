# -*- coding: utf-8 -*-


'''

Environment model : V1

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class scenario(): 
    
     
         
         
     def __init__(self, csv):
         
         self.df = pd.read_csv(csv)
         self.ugv_data_points = [(self.df['UGV_X'][i], self.df['UGV_Y'][i]) for i in range(len(self.df['UGV_X']))]
         self.uav_fuel = 287700
         self.batch_size = 256
         
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
         


     def dis_along_graph(self, node1, node2):
            
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
        
         '''-------------------------------------------------------------------''' 
        
         self.scene = scenario('ugv_road.csv')
         self.batch_size = self.scene.batch_size
         self.uav_speed = 33  
         self.ugv_speed = 15   
         self.max_planning_horizon = 250*60                                    #constant
         self.prv_action = [None] * self.batch_size                            #initialize
         self.uav_fuel_constant = 287700
         
         '''-------------------------------------------------------------------'''
          
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
         self.last_score = torch.zeros(self.batch_size, len(self.unique_mission_points[0]), device = device)                # tensor [B, N]
         self.refuel_stop_time = torch.zeros(self.batch_size, device = device)                                              # tensor [B]
         
         self.buffer_state = []
         
    
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
        
       
        if step == 0:
            assert (self.refuel_stop_time == 0).all()
        
        
        action_indices = actions
        self.prv_action = actions

        #infeasible_penalties = torch.zeros(batch_size, device = device)                            #initialize infeasible penalties 

        #------------ extract the action coordinates from the action index ---------- #
        
        is_refuel_stop = action_indices < len(self.ugv_mission_points[0])       # action : recharge
        is_refuel_stop = is_refuel_stop.unsqueeze(-1)
        is_visit_point = (action_indices >= len(self.ugv_mission_points[0]))    # action: visiting mission points
        is_visit_point = is_visit_point.unsqueeze(-1)


        if self.encoder_mission_space.shape[0] != batch_size:
               self.encoder_mission_space = self.encoder_mission_space.repeat(batch_size, 1, 1)
                  
        temp_refuel_points = self.encoder_mission_space[:, :, :2].clone() 
        refuel_coords = torch.where(is_refuel_stop, temp_refuel_points[torch.arange(batch_size, device = device), action_indices], torch.zeros(action_indices.shape[0], 2, device = device))
        visit_coords = torch.where(is_visit_point, temp_refuel_points[torch.arange(batch_size, device = device), action_indices], torch.zeros(action_indices.shape[0], 2, device = device))
        
        
        action_coords = refuel_coords + visit_coords                            # action coordinates [B, 2]
        

        # ------- unique action indices ------ #
        
        expanded_action_coords = action_coords.unsqueeze(1)
        matches = torch.all(self.unique_mission_points == expanded_action_coords, dim=-1)  # Shape: [B, N]
        matches_int = matches.int()
        unique_action_indices = torch.argmax(matches_int, dim=-1)                          # indices of action shape [B]
        assert batch_size == len(unique_action_indices)
        
        
        is_refuel_stop = is_refuel_stop.any(dim=1)
        is_visit_point = is_visit_point.any(dim=1)
        
        
        # ------- effect of actions --------- #
        
        # Mask to select only instances where the UAV is the agent
        uav_agent_mask = (self.agent_type == 0)  # Shape [B]
        
        # Apply only to instances where UAV is the agent
        if uav_agent_mask.any():  # If any instance has UAV as the agent
            # Filtered batch indices where UAV is the agent
            uav_batch_indices = batch_indices[uav_agent_mask]
            uav_agent_indices = self.uav_agent_index[uav_agent_mask]
            
            # Calculate step elapsed times only for UAV instances
            step_elapsed_times = torch.norm((action_coords[uav_agent_mask] - self.uav_pos_coord[uav_batch_indices, uav_agent_indices]), dim=1) * 5280 / self.uav_speed  # [num_uav_instances]
            
            # Update UAV fuel only for the selected UAVs in the filtered batch instances
            self.uav_fuel[uav_batch_indices, uav_agent_indices] -= (step_elapsed_times * 198).int()  # [num_uav_instances]
            kkk
        
            # Create a mask for fuel depletion check, specific to UAV instances
            fuel_depleted_mask = self.uav_fuel < 0  # Boolean mask for all UAVs
            
            
            
        # if at any ins uav is agent , then this steps will be effective:  
        step_elapsed_times = torch.norm((action_coords - self.uav_pos_coord[torch.arange(batch_size), self.uav_agent_index]  ), dim=1)* 5280 / self.uav_speed   # [ B ]
        
        self.uav_fuel[batch_indices, self.uav_agent_index] -= (step_elapsed_times * 198).int()  # [ B, N_uav]
        
        fuel_depleted_mask = self.uav_fuel < 0 # Check for fuel levels less than zero for each batch instance
        #infeasible_penalties = torch.where(fuel_depleted_mask, torch.ones(batch_size, device = device), torch.zeros(batch_size, device = device))
  
        # if at any ins ugv is agent , then this steps will be effective:       
        recharge_time = torch.where(is_refuel_stop, torch.tensor(900).float().to(device), torch.zeros_like(action_indices).float().to(device))  # [ B]
        
        
        mask = (is_refuel_stop == True)
        masked_batch_indices = batch_indices[mask]         # Select batch elements where recharging occurs
        masked_agent_indices = self.uav_agent_index[mask]  # Select corresponding agent indices
        masked_distance_time = (self.scene.dis_along_graph(self.ugv_pos_coord, action_coords)/self.ugv_speed)[mask]  # Select distance-based times for those elements
        
        
        if masked_batch_indices.numel() > 0:
              self.ugv_local_time[masked_batch_indices, masked_agent_indices] += masked_distance_time  # [ B, N_uav]
              
        step_elapsed_times_expanded = step_elapsed_times.unsqueeze(1).expand(-1, self.refuel_stop_time.size(1))  # Shape [B, N]
        
        
        refuel_stop_time_selected = self.refuel_stop_time[batch_indices, self.uav_agent_index]
        
        
        adjusted_mission_time = (self.mission_time_elapsed_local_uav + step_elapsed_times_expanded)[batch_indices, self.uav_agent_index]
        # Use torch.where with matched dimensions
        waiting_time = torch.where(
            is_refuel_stop,
            refuel_stop_time_selected - adjusted_mission_time,
            torch.zeros_like(refuel_stop_time_selected, device=device)
        )
        waiting_time = waiting_time   # [ B]
        waiting_time = torch.clamp(waiting_time, min=0) # if UGV arrives earlier UAV will have no waiting time # [ B]
        
        
        exceeded_indices =torch.where(torch.all(self.mission_status == 1, dim=1))[0]
        if exceeded_indices.numel() > 0:
               step_elapsed_times[exceeded_indices] = 0
               
    
        # -------- reward collection -------- #
        
        
        reward_collect = step_elapsed_times/60 
        
            
        # ----------- update state --------- #
        
        
        self.mission_status[torch.arange(self.batch_size), unique_action_indices] = 1                                                         # updates age period of visited mission points as 0
        
        self.mission_time_elapsed = self.mission_time_elapsed + step_elapsed_times   # [B]                                                    # mission time elapsed in global time frame
        
        self.mission_time_elapsed_local_uav[torch.arange(batch_size), self.uav_agent_index] += step_elapsed_times     # [ B, N_uav]           # time elapsed in uav's local time frame
        
        self.uav_pos_indices[torch.arange(batch_size), self.uav_agent_index]  = actions 
        self.uav_pos_coord[torch.arange(batch_size), self.uav_agent_index] = action_coords # [ B, N_uav, 2]
        self.ugv_pos_coord = action_coords
        
        masked_batch_indices = batch_indices[is_refuel_stop]  # Select batch indices where refuel stop occurs
        masked_agent_indices = self.uav_agent_index[is_refuel_stop]  # Corresponding UAV indices
        # Create a tensor filled with the full fuel level value
        full_fuel = torch.full((masked_batch_indices.size(0),), 287700.0, dtype=torch.float32, device=device)
        
        # Update the fuel to full capacity for the specific UAVs that are refueling
        self.uav_fuel[masked_batch_indices, masked_agent_indices] = full_fuel    # [B, N_uav]     #   in those instances where after recharging fuel level is full     
        
        self.recharging_location[is_visit_point] = torch.ones(self.batch_size).bool().to(device)[is_visit_point]    #   in those instances where after visiting self.recharging_location becomes false
        self.recharging_location[is_refuel_stop] = torch.zeros(self.batch_size).bool().to(device)[is_refuel_stop]   #   in those instances where after recharging self.recharging_location becomes false
        
        
        
        masked_action_coords = action_coords[is_refuel_stop]  # Corresponding action coordinates
        self.refuel_stop[masked_batch_indices, masked_agent_indices] = masked_action_coords
        self.refuel_stop_time[masked_batch_indices, masked_agent_indices] = self.mission_time_elapsed.clone()[is_refuel_stop]
        
        
        #----------------------- buffer state --------------------#
        
        local_time_of_planning_uav = self.mission_time_elapsed_local_uav[torch.arange(batch_size), self.uav_agent_index].clone()
        buffer_entry = [local_time_of_planning_uav, self.mission_status.clone()]
        self.buffer_state.append(buffer_entry)
        
        
        
        
        # <------------------------------------------ Change of uav agent ----------------------------------------> #
        
        
        
        
        # Boolean mask for instances where the agent type is UAV (0)
        uav_mask = (self.agent_type == 0)  # Shape [B]
        
        if uav_mask.any():  # If any instance in the batch is UAV
        
            # Continue processing recharging actions for instances where not all UAVs have landed
            recharge_mask = uav_mask & (is_refuel_stop == True)  # Boolean tensor [B], True where recharging is needed and agent is uav
            if recharge_mask.any():
                batch_indices = torch.arange(self.batch_size, device=device)[recharge_mask]  # [num_recharges]
                uav_indices = self.uav_agent_index[recharge_mask]  # UAV indices for batches needing recharging
                
                # Set landing complete flag for each (batch, UAV) pair
                self.uavs_landing_loc[batch_indices, uav_indices] = action_coords[batch_indices, uav_indices]
                self.uavs_landing_complete_flag[batch_indices, uav_indices] = True
                
                
            # Check if all UAVs have landed for instances with agent type UAV
            all_uavs_landed = self.uavs_landing_complete_flag.all(dim=1)  # Shape [B], True if all UAVs have landed per batch instance
        
            # Update agent_type to UGV (1) for batches where all UAVs have landed
            self.agent_type[uav_mask & all_uavs_landed] = 1
        
            # Handle cases where not all UAVs have landed
            not_all_landed_mask = uav_mask & ~all_uavs_landed  # Select instances where agent is UAV and not all UAVs have landed
        
            if not_all_landed_mask.any():
                # For each batch instance that requires recharging, select the UAV agent with the smallest time value
                recharge_mask = (is_refuel_stop == True) & not_all_landed_mask  # Select instances needing recharging with UAVs still in flight
                if recharge_mask.any():
                    # Get indices of the minimum time values along the UAV dimension for each batch instance
                    min_time_indices = torch.argmin(self.mission_time_elapsed_local_uav, dim=1)  # Shape [B]
        
                    # Update `self.uav_agent_index` only for the batch instances where recharging is required
                    self.uav_agent_index[recharge_mask] = min_time_indices[recharge_mask]
                    
                    
        ugv_mask = (self.agent_type == 1)  # Shape [B]            
        
        if ugv_mask.any():
            # Check if all UAVs have completed recharging for instances with agent type UGV
            all_uavs_recharged = self.uavs_recharge_complete_flag.all(dim=1)  # Shape [B], True if all UAVs have recharged per batch instance
        
            # Reset agent_type to UAV (0) for batches where all UAVs have recharged
            self.agent_type[ugv_mask & all_uavs_recharged] = 0

            
        
        
        
        
        print('UAV agent--->', self.uav_agent_index[0])
        
        
        
        

        # --------- reward from step ---------- #
        
        exceeded_indices = torch.where(torch.all(self.mission_status == 1, dim=1))[0]  # [B]
        if exceeded_indices.numel() > 0:
               reward_collect[exceeded_indices] = 0
            
        
        rewards = reward_collect 

        
        # -------- call state ------- #
        
        states = self.get_observation() 
        exceeded_indices = torch.where(torch.all(self.mission_status == 1, dim=1))[0]
        
        if exceeded_indices.numel() > 0:
            shrink = exceeded_indices.shape[0] 
        else :
            shrink = 0
        
        
        step_condition = torch.tensor([step > 40], dtype=torch.bool, device = device)
        shrink_condition = torch.tensor([shrink > int(batch_size*0.90)], dtype=torch.bool, device = device)
        
        dones = torch.logical_or(torch.all(self.mission_status == 1, dim=1), self.infeasibility)
        dones = torch.logical_or(dones, step_condition)
        dones = torch.logical_or(dones, shrink_condition)
        dones = dones.to(device)
        
        rewards = rewards.to(device)
        if dones.all():
            
           unfinished_batch_indices = torch.where(~torch.all(self.mission_status == 1, dim=1))[0]
           rewards[unfinished_batch_indices] = 300 # this acts like a penalty for those cases where all mission points are not visited
           #print("Unfinished batch indices:", unfinished_batch_indices)
           
           for i, time1 in enumerate(self.mission_time_elapsed):
               if time1 == 0:
                  cprint(f"Instance {i} has a value of zero", 'cyan', attrs = ['bold'] )
                  print(self.encoder_mission_space[i])
                  cprint(self.depot_index[i], 'cyan', attrs = ['bold'])
        
        info = {}  
        
        
        #print('A------->', states[2][0])
        
        
        
        return states, rewards, dones, info





    def reset(self, input):
        
         No_of_UAVs = 2
         No_of_UGVs = 1
       
    
         self.uav_mission_points =  input['uav loc']          # tensor [B, N, d]
         self.batch_size = self.uav_mission_points.size(0)
         self.ugv_mission_points=   input ['ugv loc']           # tensor [B, N, d]
         self.unique_mission_points = input['unique loc']     # tensor [B, N, d]
         self.encoder_mission_space = input['encoder space']  # tensor [B, N, d]

         depot_ix = input['depot']              # [B, 1]
         self.depot_index = depot_ix
         self.uav_pos_coord = (input['ugv loc'][torch.arange(self.batch_size).unsqueeze(1),depot_ix].squeeze(1)).unsqueeze(1).repeat(1, No_of_UAVs, 1)      # tensor [B, N_uav, d]
         
         self.ugv_pos_coord = input['ugv loc'][torch.arange(self.batch_size).unsqueeze(1),depot_ix].squeeze(1).unsqueeze(1).repeat(1, No_of_UAVs, 1)      # tensor [B, N_ugv, d]
         self.refuel_stop =   input['ugv loc'][torch.arange(self.batch_size).unsqueeze(1),depot_ix].squeeze(1).unsqueeze(1).repeat(1, No_of_UAVs, 1)      # tensor [B, N_ugv, d]   

         self.uav_pos_indices = depot_ix.clone().repeat(1, No_of_UAVs)               # tensor [B,1] 
         self.uav_fuel_limit = torch.full((self.batch_size, No_of_UAVs), 287700.0, dtype=torch.float32, device = device)               # tensor [B, N_uav] 
         self.uav_fuel = torch.full((self.batch_size, No_of_UAVs), 287700.0, dtype=torch.float32, device = device)                     # tensor [B, N_uav] 
         self.refuel_stop_time = torch.zeros(self.batch_size, device = device).unsqueeze(1).repeat(1, No_of_UAVs)       
        
  
         self.mission_status = torch.zeros_like(self.unique_mission_points, device = device)[:,:,0]  # tensor [B, N]
         self.mission_time_elapsed = torch.zeros(self.batch_size, device = device)                   # tensor [B] 
         self.infeasibility = torch.zeros(self.batch_size, device = device).bool()                   # tensor [B]
         self.recharging_location = torch.zeros(self.batch_size, device = device).bool()             # tensor [B]
         self.last_score = torch.zeros_like(self.unique_mission_points, device = device)[:,:,0]         # tensor [B, N]
         self.prv_action = [None] * self.batch_size                                               #initialize
         self.checking1 = [None] * self.batch_size 
         
         self.buffer_state = []
         self.mission_time_elapsed_local_uav = torch.zeros((self.batch_size, No_of_UAVs), device = device)                   # tensor [B, N_uav] 
         self.mission_time_elapsed_local_ugv = torch.zeros((self.batch_size, No_of_UGVs), device = device)                   # tensor [B, N_ugv] 
         
         self.uav_agent_index = torch.zeros(self.batch_size, dtype=torch.long, device = device)       # tensor [B]
         self.ugv_agent_index = torch.zeros(self.batch_size, device = device)       # tensor [B]
         
         
         
         self.uavs_landing_loc = torch.zeros((self.batch_size, No_of_UAVs, 2), device = device)  # tensor [B, N_uav, 2] , uav's landing loc
         self.uavs_landing_time = torch.zeros((self.batch_size, No_of_UAVs), device = device)  # tensor [B, N_uav] , uav's landing time
         
         
         self.uavs_landing_complete_flag = torch.zeros((self.batch_size, No_of_UAVs), device = device)  # tensor [B, N_uav] , this flag indicates that the uav has landed and we can use this to change the uav agent selection
         self.uavs_recharge_complete_flag = torch.zeros((self.batch_size, No_of_UAVs), device = device)  # tensor [B, N_uav] , this flag indicates that the uav's recharge is done and we can use this for the ugv to choose next uav to select for recharging
         
         self.agent_type = torch.zeros(self.batch_size, device = device)                   # tensor [B] 
         
         
         state = self.get_observation()
         
         
         
         
         
         return state
    
    
    
    
    def get_observation(self):
       return (self.uav_pos_indices, self.uav_fuel, self.mission_status)

            
        
    
    def feasible_action(self):
       
       
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
       
       uav_fuel_consumption = 198 * ugv_distances
       ugv_feasible_region = (uav_fuel_consumption <= self.uav_fuel[batch_indices, self.uav_agent_index].unsqueeze(1)).float().to(device)
       
       uav_fuel_consumption = 198 * uav_distances
       uav_feasible_region = (uav_fuel_consumption <= self.uav_fuel[batch_indices, self.uav_agent_index].unsqueeze(1)).float().to(device)
       
       
       action_positions = self.encoder_mission_space[:,:,:2]
       expanded_uav_pos = self.uav_pos_coord[batch_indices, self.uav_agent_index].unsqueeze(1).expand(-1, N, -1)
       dist_to_uav_nodes = self.calculate_distances(expanded_uav_pos, action_positions)
       action_positions_expanded = action_positions.unsqueeze(2)      # Shape: [B, N, 1, 2]
       ugv_positions_expanded = self.ugv_mission_points.unsqueeze(1)  # Shape: [B, 1, N, 2]
       dist_to_ugv_nodes = self.calculate_distances(action_positions_expanded, ugv_positions_expanded)  # Shape: [B, N, N]
       min_dist_to_ugv,  min_ugv_indices = torch.min(dist_to_ugv_nodes, dim=2) #[0]  # Shape: [B, N]
       total_distance = dist_to_uav_nodes + min_dist_to_ugv
       return_feasible_region = (((total_distance/ self.uav_speed)*198).int() <= (self.uav_fuel[batch_indices, self.uav_agent_index].unsqueeze(1)).int()).to(device) 
       
       
       rows_with_all_zeros = ~torch.any(return_feasible_region, dim=1)
       
       if torch.any(rows_with_all_zeros):
            rows_with_all_zeros_indices = torch.where(rows_with_all_zeros)[0]
            print("Indices of rows with all zeros:", rows_with_all_zeros_indices.tolist())
            kk
            
       
       repeated_feasible_region = (self.mission_status != 1).to(device).int()       
       tensor1 = torch.ones([batch_size, len(self.ugv_mission_points[0])], device=device)
       repeated_feasible_region = torch.cat((tensor1, repeated_feasible_region), dim=1)  # [ B, N]
       
       
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

       assert torch.any(feasible_mask, dim=1).all(), "A row with all zeros (False) found in the mask."
       
       
       
       
       if self.prv_action != [None]* self.batch_size :
           f_mask = self.prv_action < len(self.ugv_mission_points[0])                       # same refuel action = same visit action is avoided
           indices = torch.arange(self.batch_size, device=device)[f_mask]
           feasible_mask[indices, self.prv_action[indices] + len(self.ugv_mission_points[0])] = 0
       
       exceeded_indices = torch.where(torch.all(self.mission_status == 1, dim=1))[0]
       if exceeded_indices.numel() > 0:
         feasible_mask[exceeded_indices, :] = 0 
         feasible_mask[exceeded_indices, self.prv_action[exceeded_indices]] = 1
       
       
            
       assert torch.any(feasible_mask, dim=1).all(), "A row with all zeros (False) found in the mask."
       
       

       self.checking1 = (((total_distance/ self.uav_speed)*198).clone()).int()
       
       
       
       
       return feasible_mask



    
    
    
               

    











































