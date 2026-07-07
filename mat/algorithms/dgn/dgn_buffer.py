import numpy as np


class DGNReplayBuffer:
    def __init__(self, capacity, num_agents, obs_dim, action_shape, action_type, device):
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_shape = action_shape
        self.action_type = action_type
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.adj = np.zeros((self.capacity, num_agents, num_agents), dtype=np.float32)
        self.next_adj = np.zeros_like(self.adj)
        self.rewards = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, num_agents, 1), dtype=np.float32)

        action_dtype = np.int64 if action_type == "Discrete" else np.float32
        self.actions = np.zeros((self.capacity, num_agents, action_shape), dtype=action_dtype)

        self.available_actions = None
        self.next_available_actions = None

    def add_available_action_storage(self, action_dim):
        if self.available_actions is None:
            self.available_actions = np.ones((self.capacity, self.num_agents, action_dim), dtype=np.float32)
            self.next_available_actions = np.ones_like(self.available_actions)

    def insert_batch(self, obs, actions, rewards, dones, next_obs, adj, next_adj,
                     available_actions=None, next_available_actions=None):
        batch_size = obs.shape[0]
        for env_i in range(batch_size):
            self.obs[self.ptr] = obs[env_i]
            self.actions[self.ptr] = actions[env_i]
            self.rewards[self.ptr] = rewards[env_i]
            self.dones[self.ptr] = dones[env_i]
            self.next_obs[self.ptr] = next_obs[env_i]
            self.adj[self.ptr] = adj[env_i]
            self.next_adj[self.ptr] = next_adj[env_i]

            if available_actions is not None:
                self.add_available_action_storage(available_actions.shape[-1])
                self.available_actions[self.ptr] = available_actions[env_i]
                self.next_available_actions[self.ptr] = next_available_actions[env_i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
            "next_obs": self.next_obs[idx],
            "adj": self.adj[idx],
            "next_adj": self.next_adj[idx],
        }
        if self.available_actions is not None:
            batch["available_actions"] = self.available_actions[idx]
            batch["next_available_actions"] = self.next_available_actions[idx]
        else:
            batch["available_actions"] = None
            batch["next_available_actions"] = None
        return batch
