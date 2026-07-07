import copy

import numpy as np
import torch
import torch.nn.functional as F

from mat.algorithms.dgn.dgn_buffer import DGNReplayBuffer
from mat.algorithms.dgn.dgn_model import DGNActor, DGNCritic, DGNQNetwork
from mat.utils.util import get_shape_from_obs_space


def _to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


class DGNTrainer:
    def __init__(self, args, obs_space, action_space, num_agents, device):
        self.args = args
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.gamma = args.gamma
        self.relation_reg_coef = args.dgn_relation_reg_coef
        self.target_tau = args.dgn_target_tau
        self.batch_size = args.dgn_batch_size
        self.action_type = "Continuous" if action_space.__class__.__name__ == "Box" else "Discrete"

        if self.action_type == "Discrete":
            self.action_dim = action_space.n
            self.action_shape = 1
            self.q_net = DGNQNetwork(
                self.obs_dim, self.action_dim, args.dgn_hidden_dim,
                args.dgn_num_layers, args.dgn_num_heads,
            ).to(device)
            self.target_q_net = copy.deepcopy(self.q_net).to(device)
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.lr, eps=args.opti_eps)
        else:
            self.action_dim = action_space.shape[0]
            self.action_shape = self.action_dim
            self.action_low = _to_tensor(action_space.low, device)
            self.action_high = _to_tensor(action_space.high, device)
            self.actor = DGNActor(
                self.obs_dim, self.action_dim, args.dgn_hidden_dim,
                args.dgn_num_layers, args.dgn_num_heads,
            ).to(device)
            self.critic = DGNCritic(
                self.obs_dim, self.action_dim, args.dgn_hidden_dim,
                args.dgn_num_layers, args.dgn_num_heads,
            ).to(device)
            self.target_actor = copy.deepcopy(self.actor).to(device)
            self.target_critic = copy.deepcopy(self.critic).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.dgn_actor_lr, eps=args.opti_eps)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.dgn_critic_lr, eps=args.opti_eps)

        self.replay_buffer = DGNReplayBuffer(
            args.dgn_buffer_size, num_agents, self.obs_dim,
            self.action_shape, self.action_type, device,
        )
        self.total_env_steps = 0

    def prep_rollout(self):
        if self.action_type == "Discrete":
            self.q_net.eval()
        else:
            self.actor.eval()

    def prep_training(self):
        if self.action_type == "Discrete":
            self.q_net.train()
        else:
            self.actor.train()
            self.critic.train()

    def exploration_value(self):
        frac = min(1.0, self.total_env_steps / max(1, self.args.dgn_epsilon_decay_steps))
        if self.action_type == "Discrete":
            return self.args.dgn_epsilon_start + frac * (self.args.dgn_epsilon_end - self.args.dgn_epsilon_start)
        return self.args.dgn_action_noise + frac * (self.args.dgn_action_noise_end - self.args.dgn_action_noise)

    def select_actions(self, obs, adj, available_actions=None, deterministic=False):
        obs_t = _to_tensor(obs, self.device)
        adj_t = _to_tensor(adj, self.device)

        if self.action_type == "Discrete":
            with torch.no_grad():
                q_values, _ = self.q_net(obs_t, adj_t)
                if available_actions is not None:
                    avail_t = _to_tensor(available_actions, self.device)
                    q_values = q_values.masked_fill(avail_t <= 0, -1e9)
                greedy = q_values.argmax(dim=-1, keepdim=True)

            if deterministic:
                return greedy.cpu().numpy()

            eps = self.exploration_value()
            random_actions = np.random.randint(0, self.action_dim, size=greedy.shape)
            if available_actions is not None:
                for env_i in range(random_actions.shape[0]):
                    for agent_i in range(random_actions.shape[1]):
                        valid = np.flatnonzero(available_actions[env_i, agent_i] > 0)
                        if valid.size > 0:
                            random_actions[env_i, agent_i, 0] = np.random.choice(valid)
            explore = np.random.rand(*greedy.shape) < eps
            return np.where(explore, random_actions, greedy.cpu().numpy())

        with torch.no_grad():
            actions, _ = self.actor(obs_t, adj_t)
        if not deterministic:
            noise = torch.randn_like(actions) * self.exploration_value()
            actions = actions + noise
        actions = actions.clamp(-1.0, 1.0) * self.args.dgn_action_scale
        return actions.cpu().numpy()

    def store(self, obs, actions, rewards, dones, next_obs, adj, next_adj,
              available_actions=None, next_available_actions=None):
        if rewards.ndim == 2:
            rewards = rewards[..., None]
        if dones.ndim == 2:
            dones = dones[..., None]
        self.replay_buffer.insert_batch(
            obs, actions, rewards, dones, next_obs, adj, next_adj,
            available_actions, next_available_actions,
        )
        self.total_env_steps += obs.shape[0]

    def relation_regularization(self, attn, next_attn):
        if not attn or not next_attn:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        reg = torch.zeros((), dtype=torch.float32, device=self.device)
        for a, b in zip(attn, next_attn):
            reg = reg + F.kl_div((a + 1e-8).log(), b.detach().clamp_min(1e-8), reduction="batchmean")
        return reg / len(attn)

    def soft_update(self, target, source):
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.target_tau).add_(source_param.data, alpha=self.target_tau)

    def train(self):
        if not self.replay_buffer.can_sample(self.batch_size):
            return self.empty_info()

        if self.action_type == "Discrete":
            return self.train_discrete()
        return self.train_continuous()

    def train_discrete(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs = _to_tensor(batch["obs"], self.device)
        next_obs = _to_tensor(batch["next_obs"], self.device)
        actions = _to_tensor(batch["actions"], self.device, dtype=torch.long)
        rewards = _to_tensor(batch["rewards"], self.device)
        dones = _to_tensor(batch["dones"], self.device)
        adj = _to_tensor(batch["adj"], self.device)

        q_values, attn = self.q_net(obs, adj)
        chosen_q = q_values.gather(-1, actions).squeeze(-1)

        with torch.no_grad():
            next_q, next_attn = self.target_q_net(next_obs, adj)
            if batch["next_available_actions"] is not None:
                next_avail = _to_tensor(batch["next_available_actions"], self.device)
                next_q = next_q.masked_fill(next_avail <= 0, -1e9)
            next_q = next_q.max(dim=-1, keepdim=True)[0]
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        td_loss = F.mse_loss(chosen_q.unsqueeze(-1), target_q)
        _, next_attn_online = self.q_net(next_obs, adj)
        relation_loss = self.relation_regularization(attn, next_attn_online)
        loss = td_loss + self.relation_reg_coef * relation_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.soft_update(self.target_q_net, self.q_net)

        return {
            "dgn_loss": loss.detach(),
            "dgn_td_loss": td_loss.detach(),
            "dgn_actor_loss": torch.zeros((), device=self.device),
            "dgn_critic_loss": td_loss.detach(),
            "dgn_relation_loss": relation_loss.detach(),
            "dgn_exploration": self.exploration_value(),
            "dgn_buffer_size": self.replay_buffer.size,
        }

    def train_continuous(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs = _to_tensor(batch["obs"], self.device)
        next_obs = _to_tensor(batch["next_obs"], self.device)
        actions = _to_tensor(batch["actions"], self.device)
        rewards = _to_tensor(batch["rewards"], self.device)
        dones = _to_tensor(batch["dones"], self.device)
        adj = _to_tensor(batch["adj"], self.device)

        q_values, attn = self.critic(obs, actions, adj)
        with torch.no_grad():
            next_actions, _ = self.target_actor(next_obs, adj)
            target_q, _ = self.target_critic(next_obs, next_actions, adj)
            target = rewards + self.gamma * (1.0 - dones) * target_q

        critic_loss = F.mse_loss(q_values, target)
        next_actions_online, _ = self.actor(next_obs, adj)
        _, next_attn_online = self.critic(next_obs, next_actions_online.detach(), adj)
        relation_loss = self.relation_regularization(attn, next_attn_online)
        critic_total = critic_loss + self.relation_reg_coef * relation_loss

        self.critic_optimizer.zero_grad()
        critic_total.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()

        pred_actions, _ = self.actor(obs, adj)
        actor_q, _ = self.critic(obs, pred_actions, adj)
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

        return {
            "dgn_loss": (critic_total + actor_loss).detach(),
            "dgn_td_loss": critic_loss.detach(),
            "dgn_actor_loss": actor_loss.detach(),
            "dgn_critic_loss": critic_loss.detach(),
            "dgn_relation_loss": relation_loss.detach(),
            "dgn_exploration": self.exploration_value(),
            "dgn_buffer_size": self.replay_buffer.size,
        }

    def empty_info(self):
        z = torch.zeros((), dtype=torch.float32, device=self.device)
        return {
            "dgn_loss": z,
            "dgn_td_loss": z,
            "dgn_actor_loss": z,
            "dgn_critic_loss": z,
            "dgn_relation_loss": z,
            "dgn_exploration": self.exploration_value(),
            "dgn_buffer_size": self.replay_buffer.size,
        }

    def save(self, save_dir, episode):
        if self.action_type == "Discrete":
            payload = {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
            }
        else:
            payload = {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
            }
        torch.save(payload, f"{save_dir}/dgn_{episode}.pt")

    def restore(self, model_path):
        payload = torch.load(model_path, map_location=self.device)
        if self.action_type == "Discrete":
            self.q_net.load_state_dict(payload["q_net"])
            self.target_q_net.load_state_dict(payload.get("target_q_net", payload["q_net"]))
        else:
            self.actor.load_state_dict(payload["actor"])
            self.critic.load_state_dict(payload["critic"])
            self.target_actor.load_state_dict(payload.get("target_actor", payload["actor"]))
            self.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
