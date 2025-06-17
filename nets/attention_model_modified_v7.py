''' Transformer model with attention layers '''


# v7 adds the aspect of ugv action state #

from options import get_options
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
import GPUtil
from nets.graph_encoder import GraphAttentionEncoder as  GraphAttentionEncoder1
from torch.nn import DataParallel
from utils.functions import sample_many
import time

from termcolor import colored, cprint

#from random_point_gen import random_points_generator
opts = get_options()
if opts.DRL_model == 'MF':
   from  Custom_environment_model_v4_v2 import UAV_UGV_Env
if opts.DRL_model == 'proposed':
    from  Custom_environment_model_v3_v2 import UAV_UGV_Env
    

env = UAV_UGV_Env()


torch.set_printoptions(sci_mode=False, precision=3)

index_of_interest = 0 # what you want to print


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 n_charging_station=None,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_mugdrl = problem.NAME == 'mugdrl'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize / distin
                # (distinguish the current node is target node or charging node）

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        elif self.is_mugdrl:
            node_dim = 3
            step_context_dim = embedding_dim + 1
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.embedder1 =  GraphAttentionEncoder1(n_heads=8,embed_dim=128,n_layers=3,normalization='batch')
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        #self.age_proj = nn.Linear(opts.ugv_graph_size + opts.ugv_graph_size + opts.uav_graph_size, embedding_dim)
        self.age_embed = nn.Linear(embedding_dim+1, embedding_dim)
        #self.age_proj = nn.Linear(1, embedding_dim)
        #self.age_mean_proj = nn.Linear(embedding_dim, embedding_dim)
        #self.embed_proj = nn.Linear(embedding_dim, embedding_dim)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.time_proj = nn.Linear(1, embedding_dim, bias=False)
        self.current_time_proj = nn.Linear(1, embedding_dim, bias=False)
        self.com_proj = nn.Linear(embedding_dim*2, embedding_dim, bias=False)
        #self.score_layer = nn.Linear(embedding_dim, 1)
        
        

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            
            
            
            all_mission_points = self._init_embed(input).clone()
            all_mission_points[:,:,:2] = all_mission_points[:, :, :2] / 12
            
            
            st = time.time()
            embeddings, _ = self.embedder1(all_mission_points)
            print('\n Encoder time----->', time.time()-st)
        
        st = time.time()
        _log_p, pi, cost, agents, _, _, unfinished_batches = self._inner(input, embeddings) #_log_p = [ B, number of action step, N]   #pi = [B, number of action step] #cost = [B]
        print('\n Decoder time----->', time.time()-st)
        
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask = None)
        #print('log-likelihood --->', ll)

        if return_pi:
            return cost, ll, pi
        
        return cost, ll, unfinished_batches
        
        

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]
    
    
    def find_local_mission_status(self, buffer_state, target_time):
        
        #print('Buffer state:', buffer_state, '\n')
        
        all_mission_times = torch.cat([entry[0].unsqueeze(0) for entry in buffer_state], dim=0)  # Shape [T, B]
        all_mission_statuses = torch.cat([entry[1].unsqueeze(0) for entry in buffer_state], dim=0)  # Shape [T, B, N]
        
        #print(all_mission_times)
        
        #print(all_mission_statuses)
        
        

        # Step 1: Create a mask where times are less than or equal to target_time for each element in the batch
        mask = all_mission_times <= target_time.unsqueeze(0)  # Shape [T, B], broadcasting target_time
        
        # Step 2: Set times that exceed target_time to -inf so they are ignored in the max operation
        masked_times = all_mission_times.clone()
        masked_times[~mask] = float('-inf')  # Shape [T, B]
        
        # Step 3: Find the closest time ≤ target time for each element in the batch
        closest_times, closest_indices = masked_times.max(dim=0)  # closest_times: [B], closest_indices: [B]
        
        
        # Step 4: Use indices to gather the corresponding mission statuses
        local_state_at_closest_time = all_mission_statuses[closest_indices, torch.arange(all_mission_statuses.size(1))]  # Shape [B, N]
        
        return local_state_at_closest_time, closest_times
        
        
        

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        #print('Log     prob------>', log_p[9])

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        #print('log p of acctions---->', log_p[9])

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        elif self.is_mugdrl:
            return input['encoder space']
        
        else:
            return self.init_embed(input)

    def _inner(self, input, embeddings) :#, specific_input, replanning):


        
        
        batch_size = env.batch_size 
        outputs = []
        sequences = []
        agents = []
        agent_index = []
        chosen_uav_for_recharge = []

        # if replanning:
        #     state = env.reset_to_specific_state(input, specific_input)

        # else:
        state = env.reset(input)
         
        

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        j = 0
        
        
        while True: 

            decoder_ele_start_time = time.time()
            
            # Separate indices for UAV and UGV based on agent type
            uav_mask = (env.agent_type == 0)  # Boolean mask for UAV instances
            ugv_mask = (env.agent_type == 1)  # Boolean mask for UGV instances
            
            # Placeholder for the selected actions
            selected = torch.zeros(env.agent_type.size(0), device=env.agent_type.device, dtype=torch.long)  # Initialize with zeros
            
            # Placeholder for the log probabilities
            log_p_combined = torch.zeros((env.agent_type.size(0), 1, env.encoder_mission_space.size(1)), device=env.agent_type.device)  # Initialize for combined log_p

            #print('decoder time 1 --->', time.time()- decoder_ele_start_time)
            
            if uav_mask.any():
                
                
                log_p_uav, mask = self._get_log_p(embeddings, fixed, state) 
                selected_by_uav = self._select_node(log_p_uav.exp()[:, 0, :], mask[:, 0, :])  # Selected: [num_uav_instances]

                

                selected[uav_mask] = selected_by_uav[uav_mask]
                log_p_combined[uav_mask] = log_p_uav[uav_mask]

                
                

                env.prv_action_UGV[uav_mask] = torch.full((batch_size,), -1, dtype=torch.long, device="cuda")[uav_mask]

            
            
            if ugv_mask.any():
                
                 
                 current_ugv_time= env.ugv_local_time[torch.arange(batch_size, device=env.ugv_local_time.device), env.ugv_agent_index].clone()
                 selected_by_ugv, log_p_ugv, chosen_uav_index  = self.ugv_action(state, embeddings, env.uavs_landing_locs_node, env.uavs_landing_times.clone(), current_ugv_time) # Selected: [num_ugv_instances]
                 env.select_uav_for_recharge[ugv_mask, env.ugv_agent_index[ugv_mask].long()] = chosen_uav_index[ugv_mask].float()   # tensor [num_ugv_instances]
                 
                 
                 selected[ugv_mask] = selected_by_ugv[ugv_mask]

                 env.prv_action_UGV[ugv_mask] = selected_by_ugv[ugv_mask]
                 
                   
                 log_p_combined[ugv_mask] = log_p_ugv[ugv_mask]
                 
            
            
            log_p = log_p_combined
            
            
            if env.agent_type[index_of_interest] == 1:
                agent_type = 'UGV_{:02d}'.format(env.ugv_agent_index[index_of_interest].int().item() + 1)
                selected_uav_for_recharging = env.select_uav_for_recharge[index_of_interest]
            

            elif env.agent_type[index_of_interest] == 0:
                agent_type = 'UAV_{:02d}'.format(env.uav_agent_index[index_of_interest].int().item() + 1)
                selected_uav_for_recharging = env.select_uav_for_recharge[index_of_interest]

            

                
            if self.decode_type == "greedy":
                cprint('Action -------> {} by agent {}, selected uav for recharge {}'.format(selected[index_of_interest], agent_type, selected_uav_for_recharging ), 'red', attrs = ['bold'])
            else:
                cprint('Action -------> {} by agent {}, selected uav for recharge {}'.format(selected[index_of_interest], agent_type, selected_uav_for_recharging ), 'cyan', attrs = ['bold'])
            

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)
            agents.append( env.agent_type.clone())
            agent_index_tensor = torch.where(
                env.agent_type == 0,  # Condition for UAV
                env.uav_agent_index,  # Select uav_agent_index for UAVs
                env.ugv_agent_index   # Select ugv_agent_index for UGVs
            ).clone() 
            agent_index.append(agent_index_tensor)

            # Gather the selected UAV for the UGV index in each batch
            selected_uav_for_recharge = torch.gather(
            env.select_uav_for_recharge,  # [B, N_ugv]
            dim=1,
            index=env.ugv_agent_index.unsqueeze(1)  # [B, 1] - Convert to 2D for gather
        ).squeeze(1).clone()  # Remove the extra dimension to get shape [B]
            
            chosen_uav_for_recharge.append(selected_uav_for_recharge)

             
 
            # Step execution
            state, done, _ = env.step(selected, j)
    
            #print('Buffer state:', env.buffer_state)
            #if j > 25:
                
                  #random_tensor = torch.randint(low=10, high=50, size=(batch_size,)).to('cuda')
                  #local_state, closest_time = self.find_local_state(env.buffer_state, random_tensor)
                  
            
            j += 1
            
            if done.all():
                mission_terminal_time = env.mission_time_elapsed  # just to balance between unequal length
                break
        
        cost_unified = env.total_mission_time/60 
        unfinished_batch_indices = torch.where(env.mission_complete_flag != 1)[0]
        unfinished_batches = unfinished_batch_indices


        print('index of interest cost --->', cost_unified[index_of_interest])

        print('cost--->', cost_unified.mean())
        
        return torch.stack(outputs, 1), torch.stack(sequences, 1), cost_unified, torch.stack(agents, 1), torch.stack(agent_index, 1),  torch.stack(chosen_uav_for_recharge, 1), unfinished_batches
        
    
    def ugv_action(self, state, embeddings, landing_loc_node, time_of_landing, current_time):

        

        batch_size, N_uav = landing_loc_node.size()  # [B, N_uav]
        B, graph_size, d_k = embeddings.size()       # [B, graph_size, d_k]

        landing_loc_node = landing_loc_node.long()  # Convert to int64 if not already

        # Step 1: Gather embeddings for landing locations
        landing_location_embed = torch.gather(
            embeddings,  # Shape: [B, graph_size, d_k]
            1,  # Gather along the node dimension
            landing_loc_node.unsqueeze(-1).expand(batch_size, N_uav, d_k)  # Shape: [B, N_uav, d_k]
        )

        # Step 2: Process temporal information
        time_of_landing_embed = F.leaky_relu(self.time_proj(time_of_landing.unsqueeze(-1)))  # [B, N_uav, time_dim]
        current_time_embed = F.relu(self.current_time_proj(current_time.unsqueeze(-1))).unsqueeze(1)  # [B, 1, time_dim]
        current_time_embed = current_time_embed.expand(-1, N_uav, -1)  # [B, N_uav, time_dim]

        # Step 3: Combine embeddings and temporal information
        combined_input = torch.cat([landing_location_embed, time_of_landing_embed + current_time_embed], dim=-1)  # [B, N_uav, d_k + time_dim]
        combined_input = F.relu(self.com_proj(combined_input))  # [B, N_uav, hidden_dim]

        # Step 4: Compute affinities
        affinities = torch.matmul(combined_input, embeddings.transpose(-1, -2))  # [B, N_uav, graph_size]

        

        # Step 5: Create a mask for valid indices
        
        mask = torch.full((batch_size, graph_size), -float('inf'), device=affinities.device)  # [B, graph_size]
        valid_indices = landing_loc_node.masked_fill(env.uavs_recharge_complete_flag.bool(), -1)  # Exclude completed UAVs
        batch_idx = torch.arange(batch_size, device=landing_loc_node.device).unsqueeze(-1).expand(-1, N_uav)  # [B, N_uav]
        valid_indices_flat = valid_indices[valid_indices != -1]
        batch_idx_flat = batch_idx[valid_indices != -1]
        mask[batch_idx_flat, valid_indices_flat] = 0  # Set valid indices to 0
        #mask = mask.unsqueeze(1)  # [B, 1, graph_size]

        
        current_ugv_index = env.ugv_agent_index  # [B]
        current_allocation = env.allocation_tensor[torch.arange(batch_size, device=env.allocation_tensor.device), current_ugv_index, :]  # [B, N_uav]
        valid_allocations = current_allocation.bool() & (valid_indices != -1)  # Combine valid UAVs and allocated UAVs
        batch_idx_alloc = torch.arange(batch_size, device=landing_loc_node.device).unsqueeze(-1).expand(-1, N_uav)  # [B, N_uav]
        
        valid_alloc_indices = valid_indices[valid_allocations]  # Indices of UAVs that are valid and allocated
        batch_idx_alloc_flat = batch_idx_alloc[valid_allocations]  # Corresponding batch indices
        

        mask[batch_idx_alloc_flat, valid_alloc_indices] = 0  # Set allocated UAVs to 0 in the mask
        mask = mask.unsqueeze(1).expand(-1, N_uav, -1)  # Final shape: [B, N_uav, graph_size]




        # Apply the mask to affinities
        masked_affinities = affinities + mask  # Shape: [B, N_uav, graph_size]


        

        # Step 6: Handle mission completion
        aggregated_affinities = masked_affinities.mean(dim=1, keepdim=True)  # [B, 1, graph_size]
        exceeded_indices = torch.where(env.mission_complete_flag == 1)[0]  # [B_exceeded]
        if exceeded_indices.numel() > 0:
            # Mask all locations except the previous UGV action for completed missions
            aggregated_affinities[exceeded_indices] = -float('inf')  # Mask everything
            aggregated_affinities[exceeded_indices, :, env.prv_action_UGV[exceeded_indices]] = 0  # Allow only previous action

        # Step 7: Compute log probabilities
        log_probabilities = F.log_softmax(aggregated_affinities, dim=-1)  # [B, 1, graph_size]
        probabilities = log_probabilities.exp()  # Convert to probabilities for sampling
        actions = torch.multinomial(probabilities.squeeze(1), num_samples=1).squeeze(-1)  # [B]

        

        # Step 8: Determine which UAV is chosen based on the action
        chosen_uav = torch.full((batch_size,), -1, dtype=torch.long, device=actions.device)  # Initialize [B]
        eligible_uavs_mask = env.uavs_recharge_complete_flag == 0  # [B, N_uav]
        matching_uavs_mask = landing_loc_node == actions.unsqueeze(-1)  # [B, N_uav]
        valid_uavs_mask = eligible_uavs_mask & matching_uavs_mask  # [B, N_uav]

        # Find UAVs with single and multiple matches
        single_match = valid_uavs_mask.sum(dim=-1) == 1  # [B]
        if single_match.any():
            chosen_uav[single_match] = torch.argmax(valid_uavs_mask[single_match].float(), dim=-1)  # Single match UAVs


        multiple_match = valid_uavs_mask.sum(dim=-1) > 1  # [B]
        
    
        if multiple_match.any():
            batch_indices, uav_indices = torch.nonzero(valid_uavs_mask, as_tuple=True)  # [num_matches]
            valid_landing_times = time_of_landing[batch_indices, uav_indices]  # [num_matches]
            sorted_indices = torch.argsort(valid_landing_times)  # Sort by earliest landing time
            selected_uav = uav_indices[sorted_indices]  # UAV indices sorted by time
            selected_batch = batch_indices[sorted_indices]  # Batch indices sorted by time
            sorted_batch, sorted_indices = torch.sort(selected_batch)
            is_first_occurrence = torch.ones_like(sorted_batch, dtype=torch.bool)
            is_first_occurrence[1:] = sorted_batch[1:] != sorted_batch[:-1]  # True for first occurrence of each batch
            first_occurrence_indices = sorted_indices[is_first_occurrence]
            unique_batches_b = sorted_batch[is_first_occurrence]  # Unique batch values
            chosen_uav[unique_batches_b] = selected_uav[first_occurrence_indices]  # Assign UAVs with earliest landing times

            
            
            




        

        return actions, log_probabilities, chosen_uav


    
        
        
    

    def do_batch_rep(self, v, n):
        
        if isinstance(v, dict):
            return {k: self.do_batch_rep(v_, n) for k, v_ in v.items()}
        elif isinstance(v, list):
            return [self.do_batch_rep(v_, n) for v_ in v]
        elif isinstance(v, tuple):
            return tuple(self.do_batch_rep(v_, n) for v_ in v)

        return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])
    
    
    

    def sample_many(self, input, batch_rep=1, iter_rep=1): #self, input, specific_input, replanning, batch_rep=1, iter_rep=1)
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        
        all_mission_points = self._init_embed(input).clone()
        all_mission_points[:,:,:2] = all_mission_points[:, :, :2] / 12
        
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder1(all_mission_points)[0]) ,#, specific_input, replanning), # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        
        #print('Probability--->',probs[0])
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            #cprint('Sampling', 'green')
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        
        #print("requires_grad:" , selected.requires_grad)
        return selected

    def _precompute(self, embeddings, num_steps=1):
        
        
        
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
        
        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )   #(n_heads, batch_size, num_steps, graph_size, head_dim)
        
        
            
        
        return AttentionModelFixed(embeddings,  *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self,embeddings, fixed, state, normalize=True):
        
        assert not torch.isnan(embeddings).any()


        age_period_unique = state[2]  #/env.max_planning_horizon
        #age_period = torch.cat((age_period, age_period), dim=1)  # To Do: helpful if we have repeating point
        age_period_ugv = age_period_unique.clone()[:, :env.ugv_mission_points.size(1)]

        age_period = torch.cat((age_period_ugv, age_period_unique ), dim = 1)
        
        h_g = torch.cat((embeddings, age_period.unsqueeze(2)), dim=2)
        age_period = self.age_embed(h_g)
        
        age_period_mean = age_period.mean(1)[:, None, :]

        

        '''age_period = (age_period.unsqueeze(-1)/env.max_planning_horizon)
        age_period = self.age_embed(age_period)
        age_period_mean = age_period.mean(1)[:, None, :]'''
        
        # Compute query = context node embedding
        query  = age_period_mean+ \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
                #_get_parallel_step_context adds current node and reamaining length to form [ B, 1(num_steps), dk]   and adds them with graph embedding to create query
                #query shape : [ B, 1, N_k]

        
        '''age_period_unique = state[2]  
        age_period_ugv = age_period_unique.clone()[:, :env.ugv_mission_points.size(1)]
        
        age_period = torch.cat((age_period_ugv, age_period_unique ), dim = 1).unsqueeze(2) # [ B, N, 1]
        
        
        
        #age_period_mean = age_period.mean(1)[:, None, :]
        age_period_proj = self.age_proj(age_period)   #[ B, N, 128]

        age_period_mean = age_period_proj.mean(1)[:, None, :] #[ B, 1, 128]

        age_period_mean_proj = self.age_mean_proj(age_period_mean) # [ B, 1 , 128]

        
        
        embedding_mean = embeddings.mean(1)[:, None, :]
        embedding_mean_proj = self.embed_proj(embedding_mean)
        
        
        assert not torch.isnan(fixed.node_embeddings).any()
        


        # Compute query = context node embedding
        query  = age_period_mean_proj + embedding_mean_proj + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
                #_get_parallel_step_context adds current node and reamaining length to form [ B, 1(num_steps), dk]   and adds them with graph embedding to create query
                #query shape : [ B, 1, N_k]'''
                
       
                
        #assert not torch.isnan(self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))).any()
        #assert not torch.isnan(age_period_mean).any()
        
        assert not torch.isnan(query).any()
        
        
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        
        
       
        
        #print('Mask 3 time ------>', time.time()-st1)
        

        # Compute the mask
        
        
        mask = (env.feasible_action().unsqueeze(1)==0).bool() #state.get_mask() mask : [ B , 1 , N ]
       
        
        #print('Mask 4 time ------>', time.time()-st1)
        
        #print(mask[0])
        #print(query[0])
        #print(glimpse_K[0])
        #print(glimpse_V[0])
        #st1 = time.time()
        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        #print('Mask 5 time ------>', time.time()-st1)
        
        assert not torch.isnan(log_p).any()
        prv_log_p = log_p
        #print('Log p ------>', log_p[0])
        
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        
        
        if torch.isnan(log_p).any():
             nan_indices = torch.where(torch.isnan(log_p))
             unique_batch_indices = torch.unique(nan_indices[0])
            
             print('Indices where nan is present--->', nan_indices )
             print('Previous log_p--->', prv_log_p[unique_batch_indices[0]])
             print('log_p--->', log_p[unique_batch_indices[0]])
            
             
             
             large_negative_value = -1e6
             log_p = torch.where(torch.isnan(log_p), torch.tensor(large_negative_value, device=log_p.device), log_p)
             assert not torch.isnan(log_p).any(), "NaN values are still present in log_probs after replacement"
        
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        
        # Filtered batch indices for UAV instances
        
        batch_indices = torch.arange(env.uav_pos_coord.size(0), device=env.uav_pos_coord.device)
        
        current_node  = state[0][
                         batch_indices,  # Batch indices
                         env.uav_agent_index  # UAV agent indices
                         ].unsqueeze(1) # Shape: [batch_instances, 1] 
        
        batch_size, num_steps = current_node.size()
       
        
       
        
       
        if self.is_mugdrl:
           
           uav_fuel = ((state[1][batch_indices, env.uav_agent_index])/(env.uav_fuel_constant)).unsqueeze(1)
           
           
           
           
           
           
           return torch.cat(
               
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)) ,
                    
                    uav_fuel[:, :, None]),-1)
            
            

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        
        
        

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        #print('Glimpse query ------>', glimpse_Q[0])
        
        
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        
        
        #print(compatibility[0])
        if self.mask_inner:
            #print('-------------------------')
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        
        assert not torch.isnan(logits).any()
        
        nan_mask = torch.isnan(logits)

        if nan_mask.any():
            cprint('<----------- Having Nan values ----------->', 'yellow', attrs = ['bold'])
            nan_batch_indices = torch.any(nan_mask, dim=-1)
            for batch_idx in torch.where(nan_batch_indices)[0]:
                logits[batch_idx] = torch.full(logits[batch_idx].shape, 1.0 / logits.size(-1), device=logits.device)
                
        '''if torch.isnan(logits).any():
            print(key_size, val_size)
            print(final_Q.size(-1))
            print(glimpse_Q.size(-1))
            kkk'''
            
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            
        assert not torch.isnan(logits).any()
        
        if self.mask_logits:
            logits[mask] = -math.inf
        
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        
        

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
