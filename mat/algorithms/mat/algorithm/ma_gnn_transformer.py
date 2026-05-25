import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from mat.algorithms.utils.variationalPolicyEncoder import PolicyVAE
from mat.algorithms.mat.algorithm.aero_gnn import AERO_GNN_Model as gnn
from mat.algorithms.mat.algorithm.aero_gnn import GATv2MultiHop as gat

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class Encoder(nn.Module):

    def __init__(self, args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        '''self.ln_ = nn.ModuleList()
        for n in range(n_agent):
            ln = nn.LayerNorm(obs_dim+2*n_embd)
            self.ln_.append(ln)

        self.attn_ = nn.ModuleList()
        for n in range(n_agent):
            attn = SelfAttention(obs_dim+2*n_embd, n_head, n_agent, masked=False)
            self.attn_.append(attn)'''
        
        self.head_ = nn.ModuleList()
        for n in range(n_agent):
            critic = nn.Sequential(nn.LayerNorm(obs_dim+2*n_embd),
                                init_(nn.Linear(obs_dim+2*n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)                
        v_loc = []
        rep = []
        for n in range(self.n_agent):
            x = obs[:, n, :]
            x = x.unsqueeze(1)
            #rep_n = self.ln_[n](x)# + self.attn_[n](x, x, x))
            rep_n = x
            v_loc_n = self.head_[n](rep_n[:,0,:])
            v_loc.append(v_loc_n)
            rep.append(rep_n)
        v_loc = torch.stack(v_loc, dim=1)
        rep = torch.stack(rep, dim=1)

        v_loc, _ = torch.sort(v_loc, dim=-1)
            
        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
                        
        print('n_agent = ', n_agent)
        print('action_dim = ', action_dim)
        
        '''self.ln_ = nn.ModuleList()
        for n in range(n_agent):
            ln = nn.LayerNorm(obs_dim+2*n_embd)
            self.ln_.append(ln)

        self.attn_ = nn.ModuleList()
        for n in range(n_agent):
            attn = SelfAttention(obs_dim+2*n_embd, n_head, n_agent, masked=False)
            self.attn_.append(attn)'''
        
        self.mlp_ = nn.ModuleList()
        for n in range(n_agent):
            actor = nn.Sequential(nn.LayerNorm(obs_dim+2*n_embd),
                                init_(nn.Linear(obs_dim+2*n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

        '''self.actor_trunk = nn.Sequential(nn.LayerNorm(obs_dim+2*n_embd),
                                init_(nn.Linear(obs_dim+2*n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd))

        self.mlp_ = nn.ModuleList()
        for n in range(n_agent):
            actor = nn.Sequential(init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)'''

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        #x = obs
        #x = F.gelu(x)

        #obs = self.actor_trunk(obs)

        logit = []
        for n in range(self.n_agent):
            x = obs[:, n, :]
            x = x.unsqueeze(1)
            #x = self.ln_[n](x)# + self.attn_[n](x, x, x))
            logit_n = self.mlp_[n](x[:,0,:])
            logit.append(logit_n)
        logit = torch.stack(logit, dim=1)
            
        return logit


class PolicyCNN1D(nn.Module):
    def __init__(self, hid_dim, act_dim, out_channels=32, kernel_size=3):
        super().__init__()
        # Treat `act_dim` as input channels, `hid_dim` as sequence length
        self.conv1d = nn.Conv1d(
            in_channels=hid_dim,      # Each "channel" is an action dimension
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # To maintain sequence length
        )
        # Optional pooling or additional layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces hid_dim to 1

    def forward(self, policy):
        # Input: [batch_size, act_dim, hid_dim]
        # Permute to [batch_size, hid_dim (channels),  act_dim(sequence)]
        policy = policy.permute(0, 2, 1)  
        conv_out = self.conv1d(policy)  # [batch_size, out_channels, act_dim]
        pooled = self.pool(conv_out)    # [batch_size, out_channels, 1]
        return pooled.squeeze(-1)       # [batch_size, out_channels]


class MultiAgentGnnTransformer(nn.Module):

    def __init__(self, args, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, num_quants=50):
        super(MultiAgentGnnTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self._use_CNN_for_pi = args.use_CNN_for_pi
        self._use_VAE_for_pi = args.use_VAE_for_pi

        # state unused
        state_dim = 37
    
        # Edge Index
        self.edge_index = self.ring_edge_index(n_agent)
        #self.edge_index = np.array([[0,0,0,1,1,1,2,2,3,3,3,4,4,4,5,5],
        #                            [0,1,3,0,1,2,1,2,0,3,4,3,4,5,4,5]])
        self.edge_index = torch.from_numpy(self.edge_index).long().to(device)

        self.obs_embedding = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        
        # GNN
        self.obs_encoder = gnn(args, obs_dim, args.hid_dim, n_embd, n_agent)

        #self.obs_encoder = gat(obs_dim*args.n_rollout_threads, n_embd, n_embd, args.num_heads, n_agent)
        #self.policy_encoder = gat(n_embd*args.n_rollout_threads, n_embd, n_embd, args.num_heads, n_agent)

        if self._use_CNN_for_pi:
            self.policy_encoder = gnn(args, args.out_channels+n_agent, args.hid_dim, n_embd, n_agent)
            self.policy_processor = PolicyCNN1D(args.hid_dim, action_dim, args.out_channels, args.kernel_size)
        elif self._use_VAE_for_pi:
            self.vae = PolicyVAE(obs_dim, action_dim, n_embd, n_agent,latent_dim=512)
            self.policy_encoder = gnn(args, 512+n_agent, args.hid_dim, n_embd, n_agent)
        else:
            #self.policy_encoder = gnn(args, n_embd+n_agent, args.hid_dim, n_embd, n_agent)
            self.policy_encoder = gnn(args, action_dim*n_embd+n_agent, args.hid_dim, n_embd, n_agent)
            #self.policy_encoder = gnn(args, 2, args.hid_dim, n_embd, n_agent)
        
        self.encoder = Encoder(args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device)
        self.decoder = Decoder(args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                                   self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        
        #self.edge_index = np.array([[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
        #                            [1,9,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7,9,8,0]])
        #self.edge_index = np.array([[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
        #                            [0,1,4,0,1,2,1,2,3,2,3,4,3,0,4]])
            
        self.to(device)
        
    def ring_edge_index(self, n_agents):
        edge_index = []
        for i in range(n_agents):
            # Only connect to right neighbor to avoid duplicates
            j = np.clip((i + 1),0,n_agents) % n_agents
            #edge_index.append([i, i])
            edge_index.append([i, j])
            edge_index.append([j, i])
            
        edge_index = np.roll(edge_index, shift=1, axis=0)
    
        return np.array(edge_index).T

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        # print('v_loc shape = ', v_loc.shape)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim,
                                                                             self.tpdv, deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot



