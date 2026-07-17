import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from mat.algorithms.utils.util import check, init


def init_(module, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain("relu")
    return init(
        module,
        nn.init.orthogonal_,
        lambda bias: nn.init.constant_(bias, 0),
        gain=gain,
    )


class LocalGraphAttentionBlock(nn.Module):
    """Encode one receiver agent from its graph-local observation neighborhood."""

    def __init__(self, obs_dim, n_embd, n_head, dropout=0.0, ff_mult=2):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(
                f"DG-MAT requires n_embd ({n_embd}) to be divisible by n_head ({n_head})."
            )

        self.obs_norm = nn.LayerNorm(obs_dim)
        self.obs_projection = init_(nn.Linear(obs_dim, n_embd), activate=True)
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

    def forward(self, observations, receiver_id, adjacency):
        tokens = torch.nn.functional.gelu(
            self.obs_projection(self.obs_norm(observations))
        )
        query = tokens[:, receiver_id : receiver_id + 1, :]

        # MultiheadAttention masks entries whose value is True. Self-loops are
        # added before this block, so every receiver always has a valid key.
        neighbor_mask = adjacency[:, receiver_id, :] <= 0
        attended, _ = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=neighbor_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(query + attended)
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        return hidden[:, 0, :]


class DistributedGraphAttention(nn.Module):
    """One independently parameterized graph-attention encoder per agent."""

    def __init__(self, obs_dim, n_embd, n_head, n_agent, dropout=0.0, ff_mult=2):
        super().__init__()
        self.agent_blocks = nn.ModuleList(
            [
                LocalGraphAttentionBlock(
                    obs_dim=obs_dim,
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
                for _ in range(n_agent)
            ]
        )

    def forward(self, observations, adjacency):
        return torch.stack(
            [
                block(observations, agent_id, adjacency)
                for agent_id, block in enumerate(self.agent_blocks)
            ],
            dim=1,
        )


class AgentLogStd(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.value = nn.Parameter(torch.ones(action_dim))


class DGMAT(nn.Module):
    """Distributed Graph Multi-Agent Transformer.

    DG-MAT keeps one actor, critic, actor-attention encoder, and
    critic-attention encoder per agent. Each attention encoder can attend only
    to the receiver's graph neighbors (plus itself). The trainer owns the
    D-SGD parameter mixing step across these per-agent modules.
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
        del state_dim, n_block, dec_actor, share_actor

        self.n_agent = n_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.num_quants = num_quants
        self.action_type = action_type
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

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

        self.actor_attention = DistributedGraphAttention(
            obs_dim, n_embd, n_head, n_agent, dropout=dropout, ff_mult=ff_mult
        )
        self.critic_attention = DistributedGraphAttention(
            obs_dim, n_embd, n_head, n_agent, dropout=dropout, ff_mult=ff_mult
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

        self.last_actor_context = None
        self.last_critic_context = None
        self.to(device)

    def _prepare_observations(self, observations):
        observations = check(observations).to(**self.tpdv)
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

        actor_context = None
        if include_actor:
            actor_context = self.actor_attention(observations, adjacency)
            self.last_actor_context = actor_context

        critic_context = self.critic_attention(observations, adjacency)
        self.last_critic_context = critic_context
        return observations, adjacency, actor_context, critic_context

    def _actor_logits(self, actor_context):
        return torch.stack(
            [
                head(actor_context[:, agent_id, :])
                for agent_id, head in enumerate(self.actor_heads)
            ],
            dim=1,
        )

    def _values(self, critic_context):
        values = torch.stack(
            [
                head(critic_context[:, agent_id, :])
                for agent_id, head in enumerate(self.critic_heads)
            ],
            dim=1,
        )
        return torch.sort(values, dim=-1).values

    def _continuous_std(self, means):
        std = torch.stack(
            [torch.sigmoid(log_std.value) * 0.5 for log_std in self.log_std], dim=0
        )
        return std.unsqueeze(0).expand_as(means)

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
