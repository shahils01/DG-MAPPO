from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
import numpy as np

from .multiagentenv import MultiAgentEnv
from .manyagent_swimmer import ManyAgentSwimmerEnv
from .obsk import get_joints_at_kdist, get_parts_and_edges, build_obs


def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

env_REGISTRY = {}
env_REGISTRY["manyagent_swimmer"] = partial(env_fn, env=ManyAgentSwimmerEnv)


# using code from https://github.com/ikostrikov/pytorch-ddpg-naf
class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def action(self, action_):
        return self._action(action_)

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class MujocoMulti(MultiAgentEnv):

    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]  # e.g. Ant-v5
        self.agent_conf = kwargs["env_args"]["agent_conf"]  # e.g. '2x3'
        
        temp_ = get_parts_and_edges(self.scenario, self.agent_conf)
        # print('## self.scenario = ', self.scenario)
        # print('## self.agent_conf = ', self.agent_conf)

        self.agent_partitions, self.mujoco_edges, self.mujoco_globals, self.visibility_matrix = get_parts_and_edges(self.scenario,
                                                                                            self.agent_conf)

        self.n_agents = len(self.agent_partitions)
        #print('## self.n_agents = ', self.n_agents)
        
        self.n_actions = max([len(l) for l in self.agent_partitions])
        self.obs_add_global_pos = kwargs["env_args"].get("obs_add_global_pos", False)

        self.agent_obsk = kwargs["env_args"].get("agent_obsk",
                                                 None)  # if None, fully observable else k>=0 implies observe nearest k agents or joints
        self.agent_obsk_agents = kwargs["env_args"].get("agent_obsk_agents",
                                                        False)  # observe full k nearest agents (True) or just single joints (False)

        if self.agent_obsk is not None:
            self.k_categories_label = kwargs["env_args"].get("k_categories")
            if self.k_categories_label is None:
                if self.scenario in ["Ant-v5", "manyagent_ant"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext|qpos"
                elif self.scenario in ["Humanoid-v5", "HumanoidStandup-v5"]:
                    self.k_categories_label = "qpos,qvel,cfrc_ext,cvel,cinert,qfrc_actuator|qpos"
                elif self.scenario in ["Reacher-v5"]:
                    self.k_categories_label = "qpos,qvel,fingertip_dist|qpos"
                elif self.scenario in ["coupled_half_cheetah"]:
                    self.k_categories_label = "qpos,qvel,ten_J,ten_length,ten_velocity|"
                else:
                    self.k_categories_label = "qpos,qvel|qpos"
                    
            k_split = self.k_categories_label.split("|")
            self.k_categories = [k_split[k if k < len(k_split) else -1].split(",") for k in range(self.agent_obsk + 1)]

            self.global_categories_label = kwargs["env_args"].get("global_categories")
            # self.global_categories = self.global_categories_label.split(
            #     ",") if self.global_categories_label is not None else []
            self.global_categories = self.global_categories_label.split(
                ",") if self.global_categories_label is not None else self.k_categories[0]

        if self.agent_obsk is not None:
            self.k_dicts = [get_joints_at_kdist(agent_id,
                                                self.agent_partitions,
                                                self.mujoco_edges,
                                                k=self.agent_obsk,
                                                kagents=False, ) for agent_id in range(self.n_agents)]

        # load scenario from script
        self.episode_limit = self.args.episode_limit

        self.env_version = kwargs["env_args"].get("env_version", 2)
        if self.env_version == 2:
            try:
                self.wrapped_env = NormalizedActions(gym.make(self.scenario))
            except gym.error.Error:
                self.wrapped_env = NormalizedActions(
                    TimeLimit(partial(env_REGISTRY[self.scenario], **kwargs["env_args"])(),
                              max_episode_steps=self.episode_limit))
        else:
            assert False, "not implemented!"
        self.timelimit_env = self.wrapped_env.env
        self.timelimit_env._max_episode_steps = self.episode_limit
        self.env = self.timelimit_env.env
        self.timelimit_env.reset()
        self.obs_size = self.get_obs_size()
        # print("obs_size: ", self.obs_size)
        self.share_obs_size = self.get_state_size()

        # COMPATIBILITY
        self.n = self.n_agents
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.obs_size,)) for _ in range(self.n_agents)]
        # self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.obs_size + self.n_agents,)) for _ in range(self.n_agents)]
        self.share_observation_space = [Box(low=-10, high=10, shape=(self.share_obs_size,)) for _ in
                                        range(self.n_agents)]

        acdims = [len(ap) for ap in self.agent_partitions]
        self.action_space = tuple([Box(self.env.action_space.low[sum(acdims[:a]):sum(acdims[:a + 1])],
                                       self.env.action_space.high[sum(acdims[:a]):sum(acdims[:a + 1])]) for a in
                                   range(self.n_agents)])

        pass

    def step(self, actions):
        # need to remove dummy actions that arise due to unequal action vector sizes across agents
        flat_actions = np.concatenate([actions[i][:self.action_space[i].low.shape[0]] for i in range(self.n_agents)])
        
        obs_n, reward_n, truncated_n, terminated_n, info_n = self.wrapped_env.step(flat_actions)
        done_n = truncated_n or terminated_n
        self.steps += 1
                
        #print('## observation shape = ', len(obs_n))

        info = {}
        info.update(info_n)

        # if done_n:
        #     if self.steps < self.episode_limit:
        #         info["episode_limit"] = False   # the next state will be masked out
        #     else:
        #         info["episode_limit"] = True    # the next state will not be masked out
        if done_n:
            if self.steps < self.episode_limit:
                info["bad_transition"] = False  # the next state will be masked out
            else:
                info["bad_transition"] = True  # the next state will not be masked out

        # return reward_n, done_n, info
        rewards = [[reward_n]] * self.n_agents
        dones = [done_n] * self.n_agents
        infos = [info for _ in range(self.n_agents)]
        
        # obs = np.array(self.get_obs())
        # print("obs: ", obs.shape)
        # print("state: ", self.get_state())
        
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    # def get_obs(self):
    #     """ Returns all agent observat3ions in a list """
    #     state = self.env._get_obs()
    #     obs_n = []
    #     for a in range(self.n_agents):
    #         agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
    #         agent_id_feats[a] = 1.0
    #         # obs_n.append(self.get_obs_agent(a))
    #         # obs_n.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
    #         # obs_n.append(np.concatenate([self.get_obs_agent(a), agent_id_feats]))
    #         obs_i = np.concatenate([state, agent_id_feats])
    #         obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
    #         obs_n.append(obs_i)
    #     return obs_n
    
    def _get_obs_custom(self):
        position = self.env.unwrapped.data.qpos.flat.copy()  # added .unwrapped on 03/24/25
        velocity = self.env.unwrapped.data.qvel.flat.copy()  # added .unwrapped on 03/24/25

        #if self._exclude_current_positions_from_observation:
        position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    import numpy as np

    def get_visibility_matrix(self):
        """
        Builds the adjacency matrix implied by your edge_index logic.
        A[i, j] = 1 if there is a directed edge i → j.
        Includes self-loops.
        """
        num_agents = self.n_agents
        A = np.zeros((num_agents, num_agents), dtype=int)

        for i in range(num_agents):
            # left neighbor
            if i == 0:
                A[i, i+1] = 1
            elif i == num_agents - 1:
                A[i, i-1] = 1
            else:
                A[i, i-1] = 1
                A[i, i+1] = 1

        return A

    # def get_edge_index_matrix(self, faulty_node=None):
    #     num_agents = self.n_agents
    #     edge_indices = np.full((2, num_agents * num_agents), -1, dtype=int)

    #     idx = 0  # pointer into flat edge slots

    #     for i in range(num_agents):
    #         start_idx = i * num_agents
    #         end_idx = (i+1) * num_agents

    #         edge_indices[0, start_idx:end_idx] = i
    #         edge_indices[1, start_idx] = i

    #         if i == 0:
    #             edge_indices[1, start_idx+1] = 1
    #         elif i == num_agents - 1:
    #             edge_indices[1, start_idx+1] = i - 1
    #         else:
    #             edge_indices[1, start_idx+1] = i + 1
    #             edge_indices[1, start_idx+2] = i - 1

    #     return edge_indices

    def get_edge_index_matrix(self, faulty_node=None):
        """
        Converts a visibility matrix into edge_index matrices for each agent.
        
        Args:
            visibility_matrix: A boolean numpy array of shape (n_agents, n_agents + n_enemies)
                            indicating which units are visible to each agent.
        
        Returns:
            A list of edge_index matrices (each of shape [2, num_edges]) for each agent.
            Each edge_index represents connections from the agent to visible units.
        """
        visibility_matrix = self.visibility_matrix

        n_agents = visibility_matrix.shape[0]
        edge_indices = np.zeros((2,n_agents*(n_agents)))
        edge_indices[1,:] = -1

        for agent_id in range(n_agents):
            # Get visible units for the current agent (allies and enemies)
            visible_units = np.where(visibility_matrix[agent_id])[0]
            num_edges = len(visible_units)

            start_indx = agent_id * (n_agents)
            end_indx = (agent_id+1) * (n_agents)

            edge_indices[0, start_indx:end_indx] = agent_id  # Source node (current agent)
            # edge_indices[1, start_indx] = agent_id  # Self Loop
            edge_indices[1, start_indx+1:start_indx+num_edges+1] = visible_units  # Target nodes (visible units)

        return edge_indices

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        agent_obs_dim = self.get_obs_agent(0).size
        
        for a in range(self.n_agents):
            # 1. Create Agent ID (One-Hot)
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            
            # 2. Get Local Observation
            local_obs = self.get_obs_agent(a)

            ###### Uncomment below if you want to directly share obs between immediate neighbors
            # # 3. Build Neighbor Observations
            # # Initialize a zero vector large enough to hold obs for ALL agents
            # # Structure: [Agent_0_Obs, Agent_1_Obs, ..., Agent_N_Obs]
            # neighbor_obs_grid = np.zeros(self.n_agents * agent_obs_dim, dtype=np.float32)
            
            # # Get list of visible agents
            # visible_units = np.where(self.visibility_matrix[a])[0]

            # for neigh_id in visible_units:
            #     # Optional: Skip self if you only want *other* agents in this part
            #     # if neigh_id == a:
            #     #     continue 

            #     # Get the neighbor's observation
            #     curr_neigh_obs = self.get_obs_agent(neigh_id)
                
            #     # Calculate where to place this neighbor's data in the long vector
            #     start_index = neigh_id * agent_obs_dim
            #     end_index = start_index + agent_obs_dim
                
            #     # Fill the slot
            #     neighbor_obs_grid[start_index:end_index] = curr_neigh_obs

            # 4. Concatenate: [Local Obs, Neighbor Grid, Agent ID]
            # obs_i = np.concatenate([local_obs, neighbor_obs_grid, agent_id_feats])
            #####################################################################################

            obs_i = np.concatenate([local_obs, agent_id_feats])
            
            obs_n.append(obs_i)

        return obs_n

    def get_obs_agent(self, agent_id):
        if self.agent_obsk is None:
            return self.env._get_obs()
        else:
            # return build_obs(self.env,
            #                       self.k_dicts[agent_id],
            #                       self.k_categories,
            #                       self.mujoco_globals,
            #                       self.global_categories,
            #                       vec_len=getattr(self, "obs_size", None))
            return build_obs(self.env,
                             self.k_dicts[agent_id],
                             self.k_categories,
                             self.mujoco_globals,
                             self.global_categories)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        if self.agent_obsk is None:
            return self.get_obs_agent(0).size
        else:
            return len(self.get_obs()[0])
            # return max([len(self.get_obs_agent(agent_id)) for agent_id in range(self.n_agents)])

    def get_state(self, team=None):
        # TODO: May want global states for different teams (so cannot see what the other team is communicating e.g.)
        state = self._get_obs_custom()
        share_obs = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            # share_obs.append(np.concatenate([state, self.get_obs_agent(a), agent_id_feats]))
            state_i = np.concatenate([state, agent_id_feats])
            state_i = (state_i - np.mean(state_i)) / np.std(state_i)
            share_obs.append(state_i)
        return share_obs

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state()[0])

    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(shape=(self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions  # CAREFUL! - for continuous dims, this is action space dim rather
        # return self.env.action_space.shape[0]

    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        self.steps = 0
        self.timelimit_env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info
    
    
class IsaacMulti(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.agent_conf = kwargs["env_args"]["agent_conf"]
        
        # Initialize Isaac Sim environment
        self._simulation_app = SimulationApp({"headless": True})
        self.isaac_env = ManyAgentSwimmerIsaac(
            num_agents=len(kwargs["env_args"]["agent_conf"].split('x')[0]),
            **kwargs["env_args"]
        )
        self.wrapped_env = GymWrapper(self.isaac_env)

        # Multi-agent partitioning (unchanged from original)
        self.agent_partitions, _, _ = get_parts_and_edges(self.scenario, self.agent_conf)
        self.n_agents = len(self.agent_partitions)
        self.n_actions = max([len(l) for l in self.agent_partitions])

        # Spaces use Isaac Sim's observations (modify shapes as needed)
        self.observation_space = [
            Box(low=-np.inf, high=np.inf, shape=(self.get_obs_size() + self.n_agents,)) 
            for _ in range(self.n_agents)
        ]
        self.action_space = [
            Box(low=-1.0, high=1.0, shape=(len(ap),)) 
            for ap in self.agent_partitions
        ]

    def step(self, actions):
        # Convert list of agent actions to batched tensor
        batched_actions = np.stack([actions[i][:len(ap)] 
                                 for i, ap in enumerate(self.agent_partitions)])
        
        # Step Isaac Sim environment
        obs_n, reward_n, done_n, info_n = self.wrapped_env.step(batched_actions)
        
        # Format outputs for multi-agent
        rewards = [[reward_n]] * self.n_agents
        dones = [done_n] * self.n_agents
        infos = [info_n for _ in range(self.n_agents)]
        
        return self.get_obs(), self.get_state(), rewards, dones, infos, self.get_avail_actions()

    def get_obs(self):
        """Add agent ID features to Isaac Sim's observations"""
        base_obs = self.wrapped_env.get_obs()  # Shape: [num_agents, obs_dim]
        obs_n = []
        for a in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents)
            agent_id_feats[a] = 1.0
            obs_n.append(np.concatenate([base_obs[a], agent_id_feats]))
        return obs_n

    def get_state(self):
        """Global state is concatenation of all agents' observations"""
        return [np.concatenate(self.get_obs()) for _ in range(self.n_agents)]

    def reset(self):
        self.wrapped_env.reset()
        return self.get_obs(), self.get_state(), self.get_avail_actions()

    # Keep these unchanged from original
    get_avail_actions = lambda self: np.ones((self.n_agents, self.n_actions))
    get_avail_agent_actions = lambda self, agent_id: np.ones(self.n_actions)
    get_stats = lambda self: {}
    close = lambda self: self._simulation_app.close()
