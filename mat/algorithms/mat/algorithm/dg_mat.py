import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from mat.algorithms.utils.util import check, init


def resolve_agent_devices(
    primary_device,
    n_agent,
    enabled=False,
    device_spec=None,
    cuda_device_count=None,
):
    """Resolve the persistent owner device for every DG-MAT agent."""
    primary_device = torch.device(primary_device)
    if not enabled:
        return [primary_device] * n_agent

    if primary_device.type != "cuda":
        raise ValueError("DG-MAT agent parallelism requires CUDA.")

    count = torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
    if device_spec:
        requested = []
        for item in str(device_spec).split(","):
            item = item.strip()
            if item:
                try:
                    requested.append(
                        torch.device(
                            item if item.startswith("cuda:") else f"cuda:{item}"
                        )
                    )
                except (RuntimeError, ValueError) as error:
                    raise ValueError(
                        "--agent_parallel_devices must contain CUDA indices such as "
                        "0,1 or cuda:0,cuda:1"
                    ) from error
    else:
        requested = [torch.device(f"cuda:{index}") for index in range(count)]

    if len(requested) < 2:
        raise ValueError(
            "Agent parallelism needs at least two CUDA devices. "
            "Request multiple GPUs and optionally set --agent_parallel_devices 0,1."
        )
    invalid = [
        device
        for device in requested
        if device.index is None or device.index < 0 or device.index >= count
    ]
    if invalid:
        raise ValueError(
            f"Agent parallelism requested unavailable CUDA devices {invalid}; "
            f"only {count} logical CUDA devices are visible."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("--agent_parallel_devices must not contain duplicate devices.")

    return [requested[agent_id % len(requested)] for agent_id in range(n_agent)]


def init_(module, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(
        module,
        nn.init.orthogonal_,
        lambda bias: nn.init.constant_(bias, 0),
        gain=gain,
    )


class LocalSelfAttentionLayer(nn.Module):
    """One Transformer-style self-attention layer over local observation tokens."""

    def __init__(self, n_embd, n_head, dropout=0.0, ff_mult=2):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"DG-MAT requires n_embd ({n_embd}) to be divisible by n_head ({n_head})."
            )

        self.attention_input_norm = nn.LayerNorm(n_embd)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_input_norm = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            init_(nn.Linear(n_embd, ff_mult * n_embd), activate=True),
            nn.GELU(),
            nn.Dropout(dropout),
            init_(nn.Linear(ff_mult * n_embd, n_embd)),
        )

    def forward(self, tokens):
        normalized = self.attention_input_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feed_forward(
            self.feed_forward_input_norm(tokens)
        )


class LocalObservationSelfAttentionEncoder(nn.Module):
    """Encode one agent's local observation without accessing any peer data."""

    def __init__(
        self,
        obs_dim,
        n_embd,
        n_head,
        n_block=1,
        obs_tokens=8,
        dropout=0.0,
        ff_mult=2,
    ):
        super().__init__()
        if n_block < 1:
            raise ValueError("DG-MAT requires at least one local self-attention block.")
        if obs_tokens < 1:
            raise ValueError("DG-MAT requires at least one local observation token.")

        self.obs_dim = obs_dim
        self.num_tokens = min(int(obs_tokens), obs_dim)
        self.chunk_dim = (obs_dim + self.num_tokens - 1) // self.num_tokens
        self.padded_obs_dim = self.num_tokens * self.chunk_dim

        self.chunk_projection = init_(
            nn.Linear(self.chunk_dim, n_embd), activate=True
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens + 1, n_embd)
        )
        self.layers = nn.ModuleList(
            [
                LocalSelfAttentionLayer(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(n_block)
            ]
        )
        self.output_norm = nn.LayerNorm(n_embd)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, local_observation):
        if local_observation.size(-1) != self.obs_dim:
            raise ValueError(
                "DG-MAT local observation dimension mismatch: "
                f"{local_observation.size(-1)} != {self.obs_dim}."
            )

        if self.padded_obs_dim != self.obs_dim:
            local_observation = F.pad(
                local_observation,
                (0, self.padded_obs_dim - self.obs_dim),
            )

        chunks = local_observation.reshape(
            local_observation.size(0), self.num_tokens, self.chunk_dim
        )
        feature_tokens = F.gelu(self.chunk_projection(chunks))
        cls_token = self.cls_token.expand(local_observation.size(0), -1, -1)
        tokens = torch.cat((cls_token, feature_tokens), dim=1)
        tokens = tokens + self.position_embedding

        for layer in self.layers:
            tokens = layer(tokens)

        return self.output_norm(tokens[:, 0, :])


class DistributedLocalObservationEncoder(nn.Module):
    """One independently parameterized local self-attention encoder per agent."""

    def __init__(
        self,
        obs_dim,
        n_embd,
        n_head,
        n_agent,
        n_block=1,
        obs_tokens=8,
        dropout=0.0,
        ff_mult=2,
    ):
        super().__init__()
        self.agent_encoders = nn.ModuleList(
            [
                LocalObservationSelfAttentionEncoder(
                    obs_dim=obs_dim,
                    n_embd=n_embd,
                    n_head=n_head,
                    n_block=n_block,
                    obs_tokens=obs_tokens,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(n_agent)
            ]
        )

    def forward(self, observations):
        parts = self.forward_parts(observations)
        return torch.stack([part.to(observations.device) for part in parts], dim=1)

    def forward_parts(self, observations):
        """Encode each local observation on the device owning that agent."""
        parts = []
        for agent_id, encoder in enumerate(self.agent_encoders):
            device = next(encoder.parameters()).device
            local_observation = observations[:, agent_id, :].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            parts.append(encoder(local_observation))
        return parts


class LatentGraphCommunicationBlock(nn.Module):
    """Aggregate sender-owned latent messages for one receiver over the graph."""

    def __init__(self, n_embd, n_head, dropout=0.0, ff_mult=2):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"DG-MAT requires n_embd ({n_embd}) to be divisible by n_head ({n_head})."
            )

        self.message_norm = nn.LayerNorm(n_embd)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            init_(nn.Linear(n_embd, ff_mult * n_embd), activate=True),
            nn.GELU(),
            nn.Dropout(dropout),
            init_(nn.Linear(ff_mult * n_embd, n_embd)),
        )
        self.output_norm = nn.LayerNorm(n_embd)

    def forward(self, messages, receiver_id, adjacency):
        normalized_messages = self.message_norm(messages)
        query = normalized_messages[:, receiver_id : receiver_id + 1, :]

        # MultiheadAttention masks entries whose value is True. Self-loops are
        # added before this block, so every receiver always has a valid key.
        neighbor_mask = adjacency[:, receiver_id, :] <= 0
        attended, _ = self.attention(
            query,
            normalized_messages,
            normalized_messages,
            key_padding_mask=neighbor_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + attended)
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        return hidden[:, 0, :]


class DistributedLatentGraphCommunication(nn.Module):
    """One independently parameterized latent-message aggregator per agent."""

    def __init__(self, n_embd, n_head, n_agent, dropout=0.0, ff_mult=2):
        super().__init__()
        self.agent_blocks = nn.ModuleList(
            [
                LatentGraphCommunicationBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(n_agent)
            ]
        )

    def forward(self, messages, adjacency):
        output_device = messages.device
        parts = self.forward_parts(messages, adjacency, detach_remote=False)
        return torch.stack([part.to(output_device) for part in parts], dim=1)

    def forward_parts(self, messages, adjacency, detach_remote=False):
        """Run each receiver block on its persistent owner device."""
        message_parts = (
            [messages[:, agent_id, :] for agent_id in range(len(self.agent_blocks))]
            if isinstance(messages, torch.Tensor)
            else list(messages)
        )
        outputs = []
        for receiver_id, block in enumerate(self.agent_blocks):
            device = next(block.parameters()).device
            receiver_messages = []
            for sender_id, message in enumerate(message_parts):
                if detach_remote and sender_id != receiver_id:
                    message = message.detach()
                receiver_messages.append(message.to(device, non_blocking=True))
            message_bank = torch.stack(receiver_messages, dim=1)
            receiver_adjacency = adjacency.to(device, non_blocking=True)
            outputs.append(block(message_bank, receiver_id, receiver_adjacency))
        return outputs


class AgentLogStd(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.value = nn.Parameter(torch.ones(action_dim))


class DGMAT(nn.Module):
    """Distributed Graph Multi-Agent Transformer.

    Every agent first uses its own self-attention encoder to turn only its
    local observation into a latent message. Those sender-owned messages are
    then communicated through a graph-masked attention block. Actor and critic
    paths are independent, and the trainer owns the D-SGD parameter mixing
    step across all per-agent modules.
    """

    def __init__(
        self,
        args,
        state_dim,
        obs_dim,
        action_dim,
        n_agent,
        n_block,
        n_embd,
        n_head,
        encode_state=False,
        device=torch.device("cpu"),
        action_type="Discrete",
        dec_actor=False,
        share_actor=False,
        num_quants=1,
    ):
        super().__init__()
        del state_dim, dec_actor, share_actor

        self.n_agent = n_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.num_quants = num_quants
        self.action_type = action_type
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.agent_parallel_enabled = bool(
            getattr(args, "agent_parallel", False)
            or getattr(args, "dg_mat_agent_parallel", False)
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

        # DG-MAT is deliberately based on graph-local observations. Silently
        # switching to the legacy encode_state path would recreate the zero-
        # state critic failure in ma_transformer.py.
        if encode_state:
            raise ValueError(
                "DG-MAT encodes graph-local observations and does not support "
                "--encode_state. Remove that flag."
            )

        dropout = float(getattr(args, "dg_mat_dropout", 0.0))
        ff_mult = int(getattr(args, "dg_mat_ff_mult", 2))
        obs_tokens = int(getattr(args, "dg_mat_obs_tokens", 8))

        self.actor_local_encoder = DistributedLocalObservationEncoder(
            obs_dim=obs_dim,
            n_embd=n_embd,
            n_head=n_head,
            n_agent=n_agent,
            n_block=n_block,
            obs_tokens=obs_tokens,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        self.critic_local_encoder = DistributedLocalObservationEncoder(
            obs_dim=obs_dim,
            n_embd=n_embd,
            n_head=n_head,
            n_agent=n_agent,
            n_block=n_block,
            obs_tokens=obs_tokens,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        self.actor_communication = DistributedLatentGraphCommunication(
            n_embd=n_embd,
            n_head=n_head,
            n_agent=n_agent,
            dropout=dropout,
            ff_mult=ff_mult,
        )
        self.critic_communication = DistributedLatentGraphCommunication(
            n_embd=n_embd,
            n_head=n_head,
            n_agent=n_agent,
            dropout=dropout,
            ff_mult=ff_mult,
        )

        self.actor_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, action_dim)),
                )
                for _ in range(n_agent)
            ]
        )
        self.critic_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, n_embd), activate=True),
                    nn.GELU(),
                    nn.LayerNorm(n_embd),
                    init_(nn.Linear(n_embd, num_quants)),
                )
                for _ in range(n_agent)
            ]
        )

        if action_type != "Discrete":
            self.log_std = nn.ModuleList(
                [AgentLogStd(action_dim) for _ in range(n_agent)]
            )

        self.last_actor_messages = None
        self.last_critic_messages = None
        self.last_actor_context = None
        self.last_critic_context = None
        self._place_agent_modules()

    def _place_agent_modules(self):
        """Move complete agent-owned module bundles to their owner GPUs."""
        module_groups = [
            self.actor_local_encoder.agent_encoders,
            self.critic_local_encoder.agent_encoders,
            self.actor_communication.agent_blocks,
            self.critic_communication.agent_blocks,
            self.actor_heads,
            self.critic_heads,
        ]
        if self.action_type != "Discrete":
            module_groups.append(self.log_std)

        for agent_id, owner in enumerate(self.agent_devices):
            for modules in module_groups:
                modules[agent_id].to(owner)

    def agent_device(self, agent_id):
        return self.agent_devices[agent_id]

    def _stack_on_primary(self, parts):
        return torch.stack(
            [part.to(self.device, non_blocking=True) for part in parts], dim=1
        )

    def _prepare_observations(self, observations):
        observations = check(observations)
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations)
        if self.agent_parallel_enabled:
            # Leave a CPU-backed minibatch on CPU. Each owner GPU receives only
            # its own observation slice in forward_parts().
            observations = observations.to(dtype=torch.float32)
        else:
            observations = observations.to(**self.tpdv)
        return observations.reshape(-1, self.n_agent, self.obs_dim)

    def _prepare_adjacency(self, adjacency_matrix, batch_size):
        if adjacency_matrix is None:
            adjacency = torch.eye(
                self.n_agent, dtype=torch.float32, device=self.device
            ).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            adjacency = check(adjacency_matrix).to(**self.tpdv)
            adjacency = adjacency.reshape(-1, self.n_agent, self.n_agent)
            if adjacency.size(0) == 1 and batch_size > 1:
                adjacency = adjacency.expand(batch_size, -1, -1)
            if adjacency.size(0) != batch_size:
                raise ValueError(
                    "DG-MAT adjacency batch does not match the observation batch: "
                    f"{adjacency.size(0)} != {batch_size}."
                )

            identity = torch.eye(
                self.n_agent, dtype=adjacency.dtype, device=adjacency.device
            ).unsqueeze(0)
            adjacency = torch.maximum(adjacency, identity)

        return adjacency

    def _encode(self, observations, adjacency_matrix, include_actor=True):
        observations = self._prepare_observations(observations)
        adjacency = self._prepare_adjacency(
            adjacency_matrix, batch_size=observations.size(0)
        )

        actor_context_parts = None
        if include_actor:
            actor_message_parts = self.actor_local_encoder.forward_parts(observations)
            diagnostic_actor_messages = (
                [part.detach() for part in actor_message_parts]
                if self.agent_parallel_enabled
                else actor_message_parts
            )
            self.last_actor_messages = self._stack_on_primary(
                diagnostic_actor_messages
            )
            actor_context_parts = self.actor_communication.forward_parts(
                actor_message_parts,
                adjacency,
                detach_remote=self.agent_parallel_enabled,
            )
            self.last_actor_context = self._stack_on_primary(actor_context_parts)

        critic_message_parts = self.critic_local_encoder.forward_parts(observations)
        diagnostic_critic_messages = (
            [part.detach() for part in critic_message_parts]
            if self.agent_parallel_enabled
            else critic_message_parts
        )
        self.last_critic_messages = self._stack_on_primary(
            diagnostic_critic_messages
        )
        critic_context_parts = self.critic_communication.forward_parts(
            critic_message_parts,
            adjacency,
            detach_remote=self.agent_parallel_enabled,
        )
        self.last_critic_context = self._stack_on_primary(critic_context_parts)
        return observations, adjacency, actor_context_parts, critic_context_parts

    def _actor_logits(self, actor_context):
        context_parts = (
            [actor_context[:, agent_id, :] for agent_id in range(self.n_agent)]
            if isinstance(actor_context, torch.Tensor)
            else actor_context
        )
        outputs = []
        for agent_id, head in enumerate(self.actor_heads):
            owner = self.agent_device(agent_id)
            outputs.append(
                head(context_parts[agent_id].to(owner, non_blocking=True)).to(
                    self.device, non_blocking=True
                )
            )
        return torch.stack(outputs, dim=1)

    def _values(self, critic_context):
        context_parts = (
            [critic_context[:, agent_id, :] for agent_id in range(self.n_agent)]
            if isinstance(critic_context, torch.Tensor)
            else critic_context
        )
        outputs = []
        for agent_id, head in enumerate(self.critic_heads):
            owner = self.agent_device(agent_id)
            outputs.append(
                head(context_parts[agent_id].to(owner, non_blocking=True)).to(
                    self.device, non_blocking=True
                )
            )
        values = torch.stack(outputs, dim=1)
        return torch.sort(values, dim=-1).values

    def _continuous_std(self, means):
        std = torch.stack(
            [
                (torch.sigmoid(log_std.value) * 0.5).to(
                    self.device, non_blocking=True
                )
                for log_std in self.log_std
            ],
            dim=0,
        )
        return std.unsqueeze(0).expand_as(means)

    def _agent_module_groups(self):
        groups = [
            self.actor_local_encoder.agent_encoders,
            self.critic_local_encoder.agent_encoders,
            self.actor_communication.agent_blocks,
            self.critic_communication.agent_blocks,
            self.actor_heads,
            self.critic_heads,
        ]
        if self.action_type != "Discrete":
            groups.append(self.log_std)
        return groups

    @torch.no_grad()
    def mix_agent_parameters(self, adjacency):
        """Apply one graph-neighbor D-SGD mixing step across owner devices."""
        adjacency = check(adjacency)
        if not isinstance(adjacency, torch.Tensor):
            adjacency = torch.as_tensor(adjacency)
        adjacency = adjacency.detach().to(device="cpu", dtype=torch.float32)
        if adjacency.dim() == 3:
            adjacency = adjacency.mean(dim=0)
        adjacency = adjacency.reshape(self.n_agent, self.n_agent).clone()
        adjacency += torch.eye(self.n_agent, dtype=adjacency.dtype)
        row_sums = adjacency.sum(dim=1, keepdim=True)
        isolated = row_sums.squeeze(1) == 0
        weights = adjacency / row_sums.clamp_min(1.0)
        if isolated.any():
            weights[isolated] = 0
            weights[isolated, isolated] = 1

        for modules in self._agent_module_groups():
            parameter_maps = [dict(module.named_parameters()) for module in modules]
            for name in parameter_maps[0]:
                parameters = [mapping[name] for mapping in parameter_maps]
                snapshots = [parameter.detach().clone() for parameter in parameters]
                for receiver_id, parameter in enumerate(parameters):
                    mixed = torch.zeros_like(parameter)
                    for sender_id, snapshot in enumerate(snapshots):
                        weight = float(weights[receiver_id, sender_id])
                        if weight:
                            mixed.add_(
                                snapshot.to(parameter.device, non_blocking=True),
                                alpha=weight,
                            )
                    parameter.copy_(mixed)

    @staticmethod
    def _mask_logits(logits, available_actions):
        if available_actions is not None:
            logits = logits.masked_fill(available_actions == 0, -1e10)
        return logits

    def forward(
        self,
        state,
        obs,
        action,
        available_actions=None,
        adjacency_matrix=None,
    ):
        del state
        action = check(action).to(**self.tpdv)
        available_actions = (
            check(available_actions).to(**self.tpdv)
            if available_actions is not None
            else None
        )
        _, _, actor_context, critic_context = self._encode(
            obs, adjacency_matrix, include_actor=True
        )
        actor_output = self._actor_logits(actor_context)

        if self.action_type == "Discrete":
            logits = self._mask_logits(actor_output, available_actions)
            distribution = Categorical(logits=logits)
            action_log = distribution.log_prob(action.long().squeeze(-1)).unsqueeze(-1)
            entropy = distribution.entropy().unsqueeze(-1)
        else:
            distribution = Normal(actor_output, self._continuous_std(actor_output))
            action_log = distribution.log_prob(action)
            entropy = distribution.entropy()

        return action_log, self._values(critic_context), entropy

    def get_actions(
        self,
        state,
        obs,
        available_actions=None,
        deterministic=False,
        adjacency_matrix=None,
    ):
        del state
        available_actions = (
            check(available_actions).to(**self.tpdv)
            if available_actions is not None
            else None
        )
        _, _, actor_context, critic_context = self._encode(
            obs, adjacency_matrix, include_actor=True
        )
        actor_output = self._actor_logits(actor_context)

        if self.action_type == "Discrete":
            logits = self._mask_logits(actor_output, available_actions)
            distribution = Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else distribution.sample()
            action_log = distribution.log_prob(action)
            action = action.unsqueeze(-1)
            action_log = action_log.unsqueeze(-1)
        else:
            distribution = Normal(actor_output, self._continuous_std(actor_output))
            action = actor_output if deterministic else distribution.sample()
            action_log = distribution.log_prob(action)

        return action, action_log, self._values(critic_context)

    def get_values(
        self,
        state,
        obs,
        available_actions=None,
        adjacency_matrix=None,
    ):
        del state, available_actions
        _, _, _, critic_context = self._encode(
            obs, adjacency_matrix, include_actor=False
        )
        return self._values(critic_context)

    def zero_std(self):
        if self.action_type != "Discrete":
            with torch.no_grad():
                for log_std in self.log_std:
                    log_std.value.zero_()
