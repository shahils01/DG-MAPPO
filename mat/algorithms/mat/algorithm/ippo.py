import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act, discrete_decentralized_act, continuous_decentralized_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act


def _run_grus(grus, inputs, hidden_states, masks, sequence_length, shared):
    """Run either one shared GRUCell or one GRUCell per agent over PPO chunks."""
    total_steps, n_agent, _ = inputs.shape
    sequence_length = int(sequence_length)
    if sequence_length <= 0 or total_steps % sequence_length != 0:
        raise ValueError("invalid recurrent IPPO batch")
    batch_size = total_steps // sequence_length
    hidden_size = grus[0].hidden_size
    if hidden_states is None:
        hidden_states = inputs.new_zeros(batch_size, n_agent, hidden_size)
    hidden_states = hidden_states.reshape(batch_size, n_agent, hidden_size)
    if masks is None:
        masks = inputs.new_ones(total_steps, n_agent, 1)
    inputs = inputs.reshape(sequence_length, batch_size, n_agent, -1)
    masks = masks.reshape(sequence_length, batch_size, n_agent, 1)
    hidden = hidden_states
    outputs = []
    for step in range(sequence_length):
        masked_hidden = hidden * masks[step]
        if shared:
            hidden = grus[0](
                inputs[step].reshape(batch_size * n_agent, -1),
                masked_hidden.reshape(batch_size * n_agent, -1),
            ).reshape(batch_size, n_agent, -1)
        else:
            hidden = torch.stack(
                [gru(inputs[step, :, agent], masked_hidden[:, agent])
                 for agent, gru in enumerate(grus)],
                dim=1,
            )
        outputs.append(hidden)
    return torch.stack(outputs, dim=0).reshape(total_steps, n_agent, -1), hidden

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Encoder(nn.Module):

    def __init__(self, args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.share_policy = args.share_policy
        self.use_centralized_critic = bool(
            getattr(args, "use_centralized_critic", False)
        )
        self.use_critic_gru = bool(getattr(args, "use_critic_gru", False))

        critic_obs_dim = state_dim if self.use_centralized_critic else obs_dim
        critic_input_dim = n_embd if self.use_critic_gru else critic_obs_dim
        def make_critic():
            return nn.Sequential(nn.LayerNorm(critic_input_dim),
                            init_(nn.Linear(critic_input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                            init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                            init_(nn.Linear(n_embd, num_quants)))

        self.gru_ = nn.ModuleList()
        if self.use_critic_gru:
            self.gru_.extend(
                nn.GRUCell(critic_obs_dim, n_embd)
                for _ in range(1 if self.share_policy else n_agent)
            )

        if self.share_policy:
            self.head = make_critic()
        else:
            self.head_ = nn.ModuleList([make_critic() for _ in range(n_agent)])

    def forward(self, state, obs, rnn_states=None, masks=None, sequence_length=1,
                return_states=False):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)                
        critic_obs = state if self.use_centralized_critic else obs
        features, new_states = (
            _run_grus(self.gru_, critic_obs, rnn_states, masks, sequence_length,
                      self.share_policy)
            if self.use_critic_gru else (critic_obs, rnn_states)
        )
        v_loc = []
        rep = []
        for n in range(self.n_agent):
            x = features[:, n, :]
            x = x.unsqueeze(1)
            rep_n = x
            critic = self.head if self.share_policy else self.head_[n]
            v_loc_n = critic(rep_n[:,0,:])
            v_loc.append(v_loc_n)
            rep.append(rep_n)
        v_loc = torch.stack(v_loc, dim=1)
        rep = torch.stack(rep, dim=1)

        v_loc, _ = torch.sort(v_loc, dim=-1)
            
        if return_states:
            return v_loc, rep, new_states
        return v_loc, rep

    def agent_forward(self, state, obs, agent_id):
        # obs = torch.cat((obs,action_hat), axis=-1)
        x = state if self.use_centralized_critic else obs
        x = x.unsqueeze(1)
        critic = self.head if self.share_policy else self.head_[agent_id]
        v_loc = critic(x[:,0,:])

        return v_loc

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
        self.share_policy = args.share_policy
        self.use_actor_gru = bool(getattr(args, "use_actor_gru", False))

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim) if self.share_policy else torch.ones(n_agent, action_dim)
            self.log_std = torch.nn.Parameter(log_std)
                        
        print('n_agent = ', n_agent)
        print('action_dim = ', action_dim)
        print('obs_dim = ', obs_dim)
        
        actor_input_dim = n_embd if self.use_actor_gru else obs_dim
        def make_actor():
            return nn.Sequential(nn.LayerNorm(actor_input_dim),
                            init_(nn.Linear(actor_input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                            init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                            init_(nn.Linear(n_embd, action_dim)))

        self.gru_ = nn.ModuleList()
        if self.use_actor_gru:
            self.gru_.extend(
                nn.GRUCell(obs_dim, n_embd)
                for _ in range(1 if self.share_policy else n_agent)
            )

        if self.share_policy:
            self.mlp = make_actor()
        else:
            self.mlp_ = nn.ModuleList([make_actor() for _ in range(n_agent)])

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            if self.share_policy:
                log_std = torch.zeros(self.action_dim).to(device)
            else:
                log_std = torch.zeros(self.n_agent, self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, action, obs_rep, obs, rnn_states=None, masks=None,
                sequence_length=1, return_states=False):
        if torch.isnan(obs).any():
            print("Warning: NaNs in obs input to decoder")
            obs = obs.masked_fill(torch.isnan(obs), 0.0)

        features, new_states = (
            _run_grus(self.gru_, obs, rnn_states, masks, sequence_length,
                      self.share_policy)
            if self.use_actor_gru else (obs, rnn_states)
        )
        logit = []
        for n in range(self.n_agent):
            x = features[:, n, :]
            x = x.unsqueeze(1)
            actor = self.mlp if self.share_policy else self.mlp_[n]
            logit_n = actor(x[:, 0, :])
            logit.append(logit_n)

        logit = torch.stack(logit, dim=1)
        if return_states:
            return logit, new_states
        return logit


class IPPO(nn.Module):

    def __init__(self, args, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, num_quants=50):
        super(IPPO, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device
        self.n_embd = n_embd
        self.use_actor_gru = bool(getattr(args, "use_actor_gru", False))
        self.use_critic_gru = bool(getattr(args, "use_critic_gru", False))
        
        # Actor-Critic Networks
        self.encoder = Encoder(args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, 
                                encode_state, num_quants, device)
        self.decoder = Decoder(args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                                self.action_type, dec_actor=dec_actor, share_actor=share_actor)
            
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None, obs_rep=None,
                rnn_states_actor=None, rnn_states_critic=None, masks=None,
                sequence_length=1):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        if self.action_type == 'Discrete':
            action = action.long()
            logits = self.decoder(None, obs_rep, obs, rnn_states_actor, masks,
                                  sequence_length)
            if available_actions is not None:
                logits = logits.masked_fill(available_actions == 0, -1e10)
            distribution = Categorical(logits=logits)
            action_log = distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = distribution.entropy().unsqueeze(-1)
        else:
            means = self.decoder(None, obs_rep, obs, rnn_states_actor, masks,
                                 sequence_length)
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distribution = torch.distributions.Normal(means, action_std)
            action_log = distribution.log_prob(action)
            entropy = distribution.entropy()

        v_loc, obs_rep = self.encoder(state, obs, rnn_states_critic, masks,
                                      sequence_length)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, obs_rep=None,
                    rnn_states_actor=None, rnn_states_critic=None, masks=None,
                    return_rnn_states=False):
        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        batch_size = np.shape(obs)[0]        
        # print('v_loc shape = ', v_loc.shape)
        
        if self.action_type == "Discrete":
            logits, new_actor_states = self.decoder(
                None, obs_rep, obs, rnn_states_actor, masks, 1,
                return_states=True,
            )
            if available_actions is not None:
                logits = logits.masked_fill(available_actions == 0, -1e10)
            distribution = Categorical(logits=logits)
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
            output_action = action.unsqueeze(-1)
            output_action_log = distribution.log_prob(action).unsqueeze(-1)
        else:
            means, new_actor_states = self.decoder(
                None, obs_rep, obs, rnn_states_actor, masks, 1,
                return_states=True,
            )
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distribution = torch.distributions.Normal(means, action_std)
            output_action = means if deterministic else distribution.sample()
            output_action_log = distribution.log_prob(output_action)
        
        v_loc, obs_rep, new_critic_states = self.encoder(
            state, obs, rnn_states_critic, masks, 1, return_states=True
        )

        if return_rnn_states:
            return output_action, output_action_log, v_loc, new_actor_states, new_critic_states
        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None, rnn_states_critic=None,
                   masks=None):
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)

        v_tot, obs_rep = self.encoder(state, obs, rnn_states_critic, masks)
        return v_tot


class MAPPO(IPPO):
    """MAPPO actor-critic with a centralized state-value encoder.

    The actor remains decentralized and parameter-shared for homogeneous
    agents.  The encoder switches to ``share_obs`` when
    ``use_centralized_critic`` is enabled by the MAPPO configuration.
    """

    pass
