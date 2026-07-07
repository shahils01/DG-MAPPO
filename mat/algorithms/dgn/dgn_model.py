import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGNRelationLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("dgn_hidden_dim must be divisible by dgn_num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, x, adj):
        batch_size, num_agents, _ = x.shape
        q = self.query(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        eye = torch.eye(num_agents, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        mask = (adj + eye).clamp(max=1.0).unsqueeze(1) > 0
        scores = scores.masked_fill(~mask, -1e9)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v).transpose(1, 2).contiguous()
        out = out.view(batch_size, num_agents, self.num_heads * self.head_dim)
        return self.out(out), attention


class DGNBackbone(nn.Module):
    def __init__(self, obs_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [DGNRelationLayer(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_dim = hidden_dim * (num_layers + 1)

    def forward(self, obs, adj):
        h = self.encoder(obs)
        features = [h]
        attentions = []
        for layer in self.layers:
            h, attention = layer(h, adj)
            features.append(h)
            attentions.append(attention)
        return torch.cat(features, dim=-1), attentions


class DGNQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.backbone = DGNBackbone(obs_dim, hidden_dim, num_layers, num_heads)
        self.q_head = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, adj):
        features, attentions = self.backbone(obs, adj)
        return self.q_head(features), attentions


class DGNActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.backbone = DGNBackbone(obs_dim, hidden_dim, num_layers, num_heads)
        self.actor_head = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs, adj):
        features, attentions = self.backbone(obs, adj)
        return self.actor_head(features), attentions


class DGNCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.backbone = DGNBackbone(obs_dim + action_dim, hidden_dim, num_layers, num_heads)
        self.q_head = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, actions, adj):
        features, attentions = self.backbone(torch.cat([obs, actions], dim=-1), adj)
        return self.q_head(features), attentions
