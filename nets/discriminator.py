# -*- coding: utf-8 -*-
"""
Created on Fri Nov 8 14:00:11 2024

@author: mmonda4
"""

''' Discriminator network '''
import torch
import torch.nn as nn
from nets.graph_encoder1 import GraphAttentionEncoder as GraphAttentionEncoder1

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, embedding_dim=128,):
        super(Discriminator, self).__init__()

        # Graph Attention Encoder
        self.embedder1 = GraphAttentionEncoder1(n_heads=8, embed_dim=128, n_layers=3, normalization='batch')
        
        # Additional embedding layer for combined state and contextual information
        self.age_embed = nn.Linear(embedding_dim + 1, embedding_dim)

        

        # Layers to process the context from raw state information
        self.project_step_context = nn.Linear(embedding_dim+1, embedding_dim, bias=False)

        # Dropout layer applied after combining context vector and action embeddings
        self.dropout1 = nn.Dropout(0.3)
        
        # Classifier with an additional dropout layer
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Second dropout layer within the classifier
            nn.Linear(128, 1)
        )

        # Sigmoid layer for final probability output
        self.sigmoid = nn.Sigmoid()

    def _init_embed(self, input):
        return input['encoder space']

    def _get_parallel_step_context(self, embeddings, state):
        # Extract context features from the state
        current_node = state[:, :, 0].long()  # Shape: [B, T]
        uav_fuel = (state[:, :, 1] / 287700).unsqueeze(-1)  # Normalize fuel level, shape: [B, T, 1]
    
        batch_size, num_steps = current_node.size()
    
        # Gather embeddings based on current position index
        # We need to match the dimensions to [B, T, 128] by gathering along N' based on current_node
        mission_embeddings = torch.gather(
            embeddings,
            2,  # Gather along the mission points dimension (N')
            current_node.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_steps, 1, embeddings.size(-1))
        ).squeeze(2)  # Resulting shape after squeeze: [B, T, 128]
    
        # Concatenate mission embeddings and uav_fuel along the last dimension
        return torch.cat((mission_embeddings, uav_fuel), -1)  # Resulting shape: [B, T, 128 + 1]

    def forward(self, state, action, input):
        
        
        #--- one time ---#
        
        # Initialize embeddings for all mission points
        all_mission_points = self._init_embed(input).clone() 
        all_mission_points[:, :, :2] = all_mission_points[:, :, :2] / 12  # Scale coordinates

        # Apply graph attention encoder to get embeddings
        embeddings, _ = self.embedder1(all_mission_points)  #[ B, N', 128]
        
        
        
        

        # Process age period information and concatenate with embeddings
        age_period_unique = state[:, :, 2:]  #[ B, T, N]
        age_period_ugv = age_period_unique.clone()[:, :, :5]
        
        age_period = torch.cat((age_period_ugv, age_period_unique), dim=2)  #[ B, T, N']
        
        T = age_period.size(1)
        embeddings = embeddings.unsqueeze(1).repeat(1, T, 1, 1)  #[ B, T, N', 128]
        
        # Concatenate age period information with embeddings and apply additional embedding
        h_g = torch.cat((embeddings, age_period.unsqueeze(3)), dim=3)  #[B, T, N', 129]
        
        
        
        
        age_period_embedded = self.age_embed(h_g)  #[B, T, N', 128]
        
        
        
        # Calculate the mean of age_period embeddings for the context vector
        age_period_mean = age_period_embedded.mean(2)[:, :, None, :] #[B, T, 1, 128]
        
        
        
        age_period_mean = age_period_mean.squeeze(2)  # [B, T, 128]
        
        
        
        # Project step context by combining age-period embeddings and current step context
        context_vector = age_period_mean + self.project_step_context(self._get_parallel_step_context(embeddings, state)) #[ B, T, 128]
        
    
        
        # Ensure action has the correct number of dimensions
        action_expanded = action.unsqueeze(-1)  # Shape: [B, T, 1, 1]
        
        # Expand action to match the embedding shape along the last dimension
        action_expanded = action_expanded.expand(-1, -1, -1, embeddings.size(-1))  # Shape: [B, T, 1, 128]
        
        # Use torch.gather to select the embeddings based on the indices specified in action
        action_embeddings = torch.gather(
            embeddings,  # Source tensor
            2,           # Dimension along which to gather (N' dimension)
            action_expanded
        ).squeeze(2)  # Shape: [B, T, 128]

        
        # Combine context vector with action embeddings along the last dimension
        combined_input = torch.cat((context_vector, action_embeddings), dim=-1)  # Shape: [B, T, 256]
        
        #Apply the first dropout layer
        combined_input = self.dropout1(combined_input)
           
        # Pass combined input through classifier
        logits = self.classifier(combined_input)  # Shape: [B, T, 1]
        
        # Squeeze the last dimension to get logits in shape [B, T]
        logits = logits.squeeze(-1)  # Shape: [B, T]
        
        # Convert logits to probability
        probs = self.sigmoid(logits)  # Shape: [B, T]
        
        return probs
