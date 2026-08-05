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
from mat.algorithms.utils.variationalPolicyEncoder import PolicyVAE
# from mat.algorithms.mat.algorithm.aero_gnn import AERO_GNN_Model as gnn
from mat.algorithms.mat.algorithm.aero_gnn import GNN_Model as gnn
from mat.algorithms.mat.algorithm.dg_mat import resolve_agent_devices
# from mat.algorithms.mat.algorithm.aero_gnn import MeanGNN_Model as gnn
# from mat.algorithms.mat.algorithm.aero_gnn import GATv2MultiHop as gat

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def _run_agent_grus(grus, inputs, hidden_states, masks, sequence_length, output_device):
    """Run agent-owned GRUCells over time-major contiguous PPO chunks."""
    total_steps, n_agent, _ = inputs.shape
    sequence_length = int(sequence_length)
    if sequence_length <= 0 or total_steps % sequence_length != 0:
        raise ValueError(
            f"invalid recurrent batch: {total_steps} rows for sequence length "
            f"{sequence_length}"
        )
    batch_size = total_steps // sequence_length
    if hidden_states is None:
        hidden_states = torch.zeros(
            batch_size,
            n_agent,
            grus[0].hidden_size,
            dtype=inputs.dtype,
            device=output_device,
        )
    hidden_states = hidden_states.reshape(batch_size, n_agent, -1)
    if masks is None:
        masks = torch.ones(
            total_steps, n_agent, 1, dtype=inputs.dtype, device=output_device
        )
    masks = masks.reshape(sequence_length, batch_size, n_agent, 1)
    inputs = inputs.reshape(sequence_length, batch_size, n_agent, -1)

    agent_hidden = [hidden_states[:, i, :] for i in range(n_agent)]
    outputs = []
    for step in range(sequence_length):
        step_outputs = []
        for agent_id, gru in enumerate(grus):
            owner = next(gru.parameters()).device
            hidden = agent_hidden[agent_id].to(owner, non_blocking=True)
            hidden = hidden * masks[step, :, agent_id, :].to(
                owner, non_blocking=True
            )
            hidden = gru(
                inputs[step, :, agent_id, :].to(owner, non_blocking=True),
                hidden,
            )
            agent_hidden[agent_id] = hidden
            step_outputs.append(
                hidden.to(output_device, non_blocking=True).unsqueeze(1)
            )
        outputs.append(torch.cat(step_outputs, dim=1))

    outputs = torch.stack(outputs, dim=0).reshape(total_steps, n_agent, -1)
    final_hidden = torch.stack(
        [hidden.to(output_device, non_blocking=True) for hidden in agent_hidden],
        dim=1,
    )
    return outputs, final_hidden

class Encoder(nn.Module):

    def __init__(self, args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.use_centralized_critic = args.use_centralized_critic
        if args.algorithm_name == "mappo_dgnn_dsgd" and self.use_centralized_critic:
            raise ValueError("mappo_dgnn_dsgd critic must remain decentralized")
        self.use_critic_gru = bool(getattr(args, "use_critic_gru", False))
        self.output_device = torch.device(device)

        message_dim = 0 if args.iterations == 0 else n_embd
        input_dim = obs_dim + message_dim

        self.head_ = nn.ModuleList()
        self.gru_ = nn.ModuleList()
        for n in range(n_agent):
            if self.use_critic_gru:
                self.gru_.append(nn.GRUCell(input_dim, n_embd))
                critic = nn.Sequential(
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, num_quants)),
                )
            else:
                critic = nn.Sequential(nn.LayerNorm(input_dim),
                                init_(nn.Linear(input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(
        self,
        state,
        obs,
        rnn_states=None,
        masks=None,
        sequence_length=1,
    ):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)                
        del state
        if self.use_critic_gru:
            features, new_rnn_states = _run_agent_grus(
                self.gru_,
                obs,
                rnn_states,
                masks,
                sequence_length,
                self.output_device,
            )
        else:
            features = obs
            new_rnn_states = rnn_states

        v_loc = []
        for n in range(self.n_agent):
            x = features[:, n, :]
            owner = next(self.head_[n].parameters()).device
            v_loc_n = self.head_[n](x.to(owner, non_blocking=True)).to(
                self.output_device, non_blocking=True
            )
            v_loc.append(v_loc_n)
        v_loc = torch.stack(v_loc, dim=1)

        v_loc, _ = torch.sort(v_loc, dim=-1)
            
        return v_loc, features, new_rnn_states

    def agent_forward(self, state, obs, agent_id):
        # obs = torch.cat((obs,action_hat), axis=-1)
        x = obs
        x = x.unsqueeze(1)
        v_loc = self.head_[agent_id](x[:,0,:])

        return v_loc

    def average_critic_parameters(self):
        """
        Averages the parameters of all critic networks in head_ and 
        distributes the averaged parameters back to each critic.
        """
        with torch.no_grad():
            # Initialize dictionaries to store summed parameters
            avg_params = {}
            param_count = 0
            
            # First, sum up all parameters across all critics
            for critic in self.head_:
                for name, param in critic.named_parameters():
                    if param.requires_grad:
                        if name not in avg_params:
                            avg_params[name] = param.data.clone()
                        else:
                            avg_params[name] += param.data
                param_count += 1
            
            # Compute average
            for name in avg_params:
                avg_params[name] = avg_params[name] / param_count
            
            # Distribute averaged parameters back to all critics
            for critic in self.head_:
                for name, param in critic.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(avg_params[name])


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
        self.output_device = torch.device(device)
        self.use_actor_gru = bool(getattr(args, "use_actor_gru", False))

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
                        
        print('n_agent = ', n_agent)
        print('action_dim = ', action_dim)
        print('obs_dim = ', obs_dim)
        
        self.mlp_ = nn.ModuleList()
        self.gru_ = nn.ModuleList()
        message_dim = 0 if args.iterations == 0 else n_embd
        input_dim = obs_dim + message_dim

        for n in range(n_agent):
            if self.use_actor_gru:
                self.gru_.append(nn.GRUCell(input_dim, n_embd))
                actor = nn.Sequential(
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, action_dim)),
                )
            else:
                actor = nn.Sequential(nn.LayerNorm(input_dim),
                                    init_(nn.Linear(input_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                    init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                    init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(
        self,
        action,
        obs_rep,
        obs,
        rnn_states=None,
        masks=None,
        sequence_length=1,
        return_states=False,
    ):
        del action, obs_rep
        if torch.isnan(obs).any():
            print("Warning: NaNs in obs input to decoder")
            obs = obs.masked_fill(torch.isnan(obs), 0.0)

        if self.use_actor_gru:
            features, new_rnn_states = _run_agent_grus(
                self.gru_,
                obs,
                rnn_states,
                masks,
                sequence_length,
                self.output_device,
            )
        else:
            features = obs
            new_rnn_states = rnn_states

        logit = []
        for n in range(self.n_agent):
            x = features[:, n, :]
            owner = next(self.mlp_[n].parameters()).device
            logit_n = self.mlp_[n](x.to(owner, non_blocking=True)).to(
                self.output_device, non_blocking=True
            )
            logit.append(logit_n)

        logit = torch.stack(logit, dim=1)
        if return_states:
            return logit, new_rnn_states
        return logit

    def agent_forward(self, obs_rep, obs, agent_id):
        if torch.isnan(obs).any():
            print("Warning: NaNs in obs input to decoder")
            obs = obs.masked_fill(torch.isnan(obs), 0.0)

        x = obs
        x = x.unsqueeze(1)
        logit = self.mlp_[agent_id](x[:, 0, :])

        return logit


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
        self.n_embd = n_embd
        self.raw_obs_dim = obs_dim
        self.iterations = args.iterations
        self.use_actor_gru = bool(getattr(args, "use_actor_gru", False))
        self.use_critic_gru = bool(getattr(args, "use_critic_gru", False))
        self.agent_parallel_enabled = bool(
            getattr(args, "agent_parallel", False)
            or getattr(args, "dg_mat_agent_parallel", False)
        )
        if self.agent_parallel_enabled and args.algorithm_name != "mappo_dgnn_dsgd":
            raise ValueError(
                "This GNN implementation supports agent parallelism only for "
                "mappo_dgnn_dsgd."
            )
        device_spec = getattr(args, "agent_parallel_devices", None)
        if device_spec is None:
            device_spec = getattr(args, "dg_mat_devices", None)
        self.agent_devices = resolve_agent_devices(
            primary_device=device,
            n_agent=n_agent,
            enabled=self.agent_parallel_enabled,
            device_spec=device_spec,
        )
        if self.agent_parallel_enabled and action_type != "Discrete":
            raise ValueError(
                "MAPPO-DGNN-DSGD agent parallelism currently supports "
                "discrete action spaces only."
            )
        
        # GNN
        self.obs_encoder = gnn(args, obs_dim, n_embd, n_embd, n_agent)
        # self.policy_encoder = gnn(args, action_dim+n_agent, args.hid_dim, n_embd, n_agent)
        
        # Actor-Critic Networks
        self.encoder = Encoder(args, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, num_quants, device)
        self.decoder = Decoder(args, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device,
                                   self.action_type, dec_actor=dec_actor, share_actor=share_actor)

        self.eye = torch.eye(self.n_agent, device=device).unsqueeze(0)
        self.eye = self.eye / torch.norm(self.eye, p='fro')  # Normalize entire matrix
            
        self.to(device)
        self._place_agent_modules()

    def _place_agent_modules(self):
        for agent_id, owner in enumerate(self.agent_devices):
            self.encoder.head_[agent_id].to(owner)
            self.decoder.mlp_[agent_id].to(owner)
            if self.use_actor_gru:
                self.decoder.gru_[agent_id].to(owner)
            if self.use_critic_gru:
                self.encoder.gru_[agent_id].to(owner)
        self.obs_encoder.configure_agent_parallel(
            self.agent_devices, output_device=self.device
        )

    def agent_device(self, agent_id):
        return self.agent_devices[agent_id]

    def _prepare_model_input(self, value):
        value = check(value)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if self.agent_parallel_enabled:
            return value.to(dtype=torch.float32)
        return value.to(**self.tpdv)

    def agent_parameter_lists(self):
        parameters = []
        for agent_id in range(self.n_agent):
            owned = []
            owned.extend(self.decoder.mlp_[agent_id].parameters())
            owned.extend(self.encoder.head_[agent_id].parameters())
            if self.use_actor_gru:
                owned.extend(self.decoder.gru_[agent_id].parameters())
            if self.use_critic_gru:
                owned.extend(self.encoder.gru_[agent_id].parameters())
            owned.extend(self.obs_encoder.agent_encoders[agent_id].parameters())
            owned.extend(
                self.obs_encoder.node_classifier_heads[agent_id].parameters()
            )
            owned.extend(
                self.obs_encoder.atts[hop][agent_id]
                for hop in range(self.obs_encoder.K)
            )
            parameters.append(list(owned))
        return parameters

    @torch.no_grad()
    def mix_agent_parameters(self, adjacency):
        """Apply graph-neighbor D-SGD across persistent agent owner GPUs."""
        adjacency = check(adjacency)
        if not isinstance(adjacency, torch.Tensor):
            adjacency = torch.as_tensor(adjacency)
        adjacency = adjacency.detach().to(device="cpu", dtype=torch.float32)
        if adjacency.dim() == 3:
            adjacency = adjacency.mean(dim=0)
        adjacency = adjacency.reshape(self.n_agent, self.n_agent).clone()
        adjacency += torch.eye(self.n_agent, dtype=adjacency.dtype)
        weights = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)

        parameter_lists = self.agent_parameter_lists()
        for parameter_id in range(len(parameter_lists[0])):
            current = [owned[parameter_id] for owned in parameter_lists]
            snapshots = [parameter.detach().clone() for parameter in current]
            for receiver_id, parameter in enumerate(current):
                mixed = torch.zeros_like(parameter)
                for sender_id, snapshot in enumerate(snapshots):
                    weight = float(weights[receiver_id, sender_id])
                    if weight:
                        mixed.add_(
                            snapshot.to(parameter.device, non_blocking=True),
                            alpha=weight,
                        )
                parameter.copy_(mixed)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def _batch_edge_index(self, edge_index, batch_size):
        edge_index = check(edge_index)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(edge_index)
        edge_index = edge_index.to(device=self.device, dtype=torch.long)
        if edge_index.dim() == 2:
            return edge_index
        if edge_index.dim() != 3 or edge_index.shape[1] != 2:
            raise ValueError(
                "DGNN edge_index must have shape [2, E] or [B, 2, E]"
            )
        if edge_index.shape[0] != batch_size:
            raise ValueError(
                f"DGNN edge batch {edge_index.shape[0]} does not match "
                f"observation batch {batch_size}"
            )
        edges = []
        for batch_id in range(batch_size):
            current = edge_index[batch_id]
            valid = (current[0] >= 0) & (current[1] >= 0)
            current = current[:, valid].clone()
            current += batch_id * self.n_agent
            edges.append(current)
        if not edges:
            return torch.empty(2, 0, dtype=torch.long, device=self.device)
        return torch.cat(edges, dim=1)

    def _augment_observations(self, obs, edge_index=None):
        raw_obs = obs[..., : self.raw_obs_dim]
        if self.iterations <= 0:
            self.last_gnn_messages = None
            return raw_obs
        if edge_index is None:
            expected = self.raw_obs_dim + self.n_embd
            if obs.shape[-1] == expected:
                self.last_gnn_messages = obs[..., self.raw_obs_dim :]
                return obs
            raise ValueError("DGNN observations require an aligned edge_index")
        batched_edges = self._batch_edge_index(edge_index, raw_obs.shape[0])
        messages = self.obs_encoder(raw_obs, batched_edges)
        self.last_gnn_messages = messages
        return torch.cat((raw_obs, messages), dim=-1)

    def forward(
        self,
        state,
        obs,
        action,
        available_actions=None,
        obs_rep=None,
        rnn_states_actor=None,
        rnn_states_critic=None,
        masks=None,
        edge_index=None,
        sequence_length=1,
    ):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = self._prepare_model_input(state)
        obs = self._prepare_model_input(obs)
        obs = self._augment_observations(obs, edge_index)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        if self.action_type == 'Discrete':
            action = action.long()
            logits = self.decoder(
                None,
                obs_rep,
                obs,
                rnn_states_actor,
                masks,
                sequence_length,
            )
            if available_actions is not None:
                logits = logits.masked_fill(available_actions == 0, -1e10)
            distribution = Categorical(logits=logits)
            action_log = distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = distribution.entropy().unsqueeze(-1)
        else:
            # The recurrent decoder has the same Gaussian policy parameterization
            # as the feed-forward path; only the mean network is recurrent.
            means = self.decoder(
                None,
                obs_rep,
                obs,
                rnn_states_actor,
                masks,
                sequence_length,
            )
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distribution = torch.distributions.Normal(means, action_std)
            action_log = distribution.log_prob(action)
            entropy = distribution.entropy()

        v_loc, obs_rep, _ = self.encoder(
            state,
            obs,
            rnn_states_critic,
            masks,
            sequence_length,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self,
        state,
        obs,
        available_actions=None,
        deterministic=False,
        batched_edge_index=None,
        obs_rep=None,
        rnn_states_actor=None,
        rnn_states_critic=None,
        masks=None,
        return_rnn_states=False,
    ):
        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = self._prepare_model_input(state)
        obs = self._prepare_model_input(obs)
        obs = self._augment_observations(obs, batched_edge_index)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        batch_size = np.shape(obs)[0]        
        # print('v_loc shape = ', v_loc.shape)
        
        if self.action_type == "Discrete":
            logits, new_actor_states = self.decoder(
                None,
                obs_rep,
                obs,
                rnn_states_actor,
                masks,
                1,
                return_states=True,
            )
            if available_actions is not None:
                logits = logits.masked_fill(available_actions == 0, -1e10)
            distribution = Categorical(logits=logits)
            action = (
                distribution.probs.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
            output_action = action.unsqueeze(-1)
            output_action_log = distribution.log_prob(action).unsqueeze(-1)
        else:
            means, new_actor_states = self.decoder(
                None,
                obs_rep,
                obs,
                rnn_states_actor,
                masks,
                1,
                return_states=True,
            )
            action_std = torch.sigmoid(self.decoder.log_std) * 0.5
            distribution = torch.distributions.Normal(means, action_std)
            output_action = means if deterministic else distribution.sample()
            output_action_log = distribution.log_prob(output_action)

        # action_logits = self.decoder(None,None,obs)
        # action_logits = torch.cat((F.gelu(action_logits), self.eye.repeat(action_logits.shape[0],1,1)), axis=-1)

        # output_action_hat = self.policy_encoder(action_logits, batched_edge_index)
        
        v_loc, obs_rep, new_critic_states = self.encoder(
            state, obs, rnn_states_critic, masks, 1
        )

        if return_rnn_states:
            return (
                output_action,
                output_action_log,
                v_loc,
                new_actor_states,
                new_critic_states,
            )
        return output_action, output_action_log, v_loc

    def get_values(
        self,
        state,
        obs,
        available_actions=None,
        rnn_states_critic=None,
        masks=None,
        edge_index=None,
    ):
        # state unused
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = self._prepare_model_input(state)
        obs = self._prepare_model_input(obs)
        obs = self._augment_observations(obs, edge_index)

        v_tot, obs_rep, _ = self.encoder(
            state, obs, rnn_states_critic, masks, 1
        )
        return v_tot
