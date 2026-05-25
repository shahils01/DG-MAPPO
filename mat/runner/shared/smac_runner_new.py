import time
import wandb
import numpy as np
from functools import reduce
import torch
import torch.nn.functional as F
from torch import Tensor
from mat.runner.shared.base_runner import Runner
import torch.nn.functional as F
from collections import defaultdict
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

        self.eye = torch.eye(self.num_agents, device="cuda:0").unsqueeze(0)
        self.eye = self.eye / torch.norm(self.eye, p='fro')  # Normalize entire matrix
        self.disconnected_net = 0
        self.truelyDistributedGNN = self.all_args.truelyDistributedGNN
        self._use_max_grad_norm = self.all_args.use_max_grad_norm
        self.max_grad_norm = self.all_args.max_grad_norm

    def run2(self):
        for episode in range(1):
            self.eval(episode)

    def get_batch_edge_index(self, edge_index):
        """
        Converts a padded multi-environment edge index into a single batched edge index.
        
        Args:
            edge_index: Padded tensor of shape [batch_size, 2, max_edges]
                    (invalid edges marked with -1 in the 2nd row).
        
        Returns:
            batched_edge_index: Merged edge index of shape [2, total_valid_edges]
        """

        batch_size = edge_index.size(0)
        batched_edges = []

        for i in range(batch_size):
            # Step 1: Remove invalid edges (where edge_index[i, 1, :] == -1)
            valid_mask = edge_index[i, 1, :] != -1
            valid_edges = edge_index[i, :, valid_mask]  # Shape [2, num_valid_edges]

            # Step 2: Apply offset to node indices (avoid collisions across batches)
            valid_edges[0, :] += i * self.num_agents  # Offset source nodes
            valid_edges[1, :] += i * self.num_agents  # Offset target nodes

            # Step 3: Collect all valid edges
            batched_edges.append(valid_edges)

        # Step 4: Concatenate all valid edges into [2, total_edges]
        return torch.cat(batched_edges, dim=1).long()

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            edge_index = self.envs.get_edge_index_matrix()
            edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
            batch_edge_index = self.get_batch_edge_index(edge_index)

            for step in range(self.episode_length):
                # Sample actions
                if self.algorithm_name == 'mappo_gnn' or self.algorithm_name == 'mappo_dgnn' or self.algorithm_name == 'mappo_dgnn_dsgd':
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step, batch_edge_index)
                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions.cpu().detach().numpy())
                
                obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
                share_obs = torch.tensor(share_obs, dtype=torch.float32, device="cuda:0")
                rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda:0")
                dones = torch.tensor(dones, dtype=torch.float32, device="cuda:0")
                available_actions = torch.tensor(available_actions, dtype=torch.float32, device="cuda:0")

                #if step % 100 == 0:
                edge_index = self.envs.get_edge_index_matrix()
                edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
                batch_edge_index = self.get_batch_edge_index(edge_index)

                adjcency_matrix = self.envs.get_visibility_matrix()[:,:,:self.num_agents]
                adjcency_matrix = torch.tensor(adjcency_matrix, dtype=torch.float32, device="cuda:0")
                                                
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                
                # insert data into buffer
                self.insert(data, batch_edge_index, edge_index, adjcency_matrix)

            # compute return and update network
            self.compute()
            train_infos = self.train(episode)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0) or (episode + 1 == int(self.num_env_steps)):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                print("incre win rate is {}.".format(incre_win_rate))
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    wandb.log({"num_disconnected_nets": self.disconnected_net}, step=total_num_steps)
                else:
                    self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                last_battles_game = battles_game
                last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        share_obs = torch.tensor(share_obs, dtype=torch.float32, device="cuda:0")
        available_actions = torch.tensor(available_actions, dtype=torch.float32, device="cuda:0")
        obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
    
        if self.all_args.iterations > 0:
            if self.algorithm_name == 'mappo_gnn' or self.algorithm_name == 'mappo_dgnn' or self.algorithm_name == 'mappo_dgnn_dsgd':
                self.buffer.obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.obs_dim+self.n_embd), dtype=torch.float32, device='cuda:0')

                print('obs shape = ', obs.shape)

                adjcency_matrix = self.envs.get_visibility_matrix()[:,:,:self.num_agents]
                adjcency_matrix = torch.tensor(adjcency_matrix, dtype=torch.float32, device="cuda:0")
                
                edge_index = self.envs.get_edge_index_matrix()
                edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
                batch_edge_index = self.get_batch_edge_index(edge_index)
                
                x = self.trainer.policy.transformer.obs_encoder(obs, batch_edge_index)

                obs = torch.cat([obs,x],dim=-1).detach()

                self.buffer.adjcency_matrix[0] = adjcency_matrix.clone()
            
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.clone()
        self.buffer.obs[0] = obs.clone()
        self.buffer.available_actions[0] = available_actions.clone()

    @torch.no_grad()
    def collect(self, step, batched_edge_index=None):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
                        self.buffer.share_obs[step],
                        self.buffer.obs[step],
                        self.buffer.rnn_states[step],
                        self.buffer.rnn_states_critic[step],
                        self.buffer.masks[step],
                        self.buffer.available_actions[step],
                        batched_edge_index
                    )

        # [self.envs, agents, dim]
        values = value.reshape(self.n_rollout_threads, self.num_agents, -1)
        actions = action.reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs = action_log_prob.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states = rnn_state.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states_critic = rnn_state_critic.reshape(self.n_rollout_threads, self.num_agents, -1)
        # action_hats = action_hat.reshape(self.n_rollout_threads, self.num_agents, -1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, batched_edge_index=None, edge_index=None, adjcency_matrix=None):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        if self.all_args.iterations > 0:
            if self.algorithm_name == 'mappo_gnn' or self.algorithm_name == 'mappo_dgnn' or self.algorithm_name == 'mappo_dgnn_dsgd':
                x = self.trainer.policy.transformer.obs_encoder(obs, batched_edge_index)
                obs = torch.cat([obs,x],dim=-1).detach()

        dones_env = torch.all(dones, dim=1)

        rnn_states[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device="cuda:0")
        rnn_states_critic[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device="cuda:0")

        masks = torch.ones((self.n_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")
        masks[dones_env == True] = torch.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device="cuda:0")

        active_masks = torch.ones((self.n_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")
        active_masks[dones == True] = torch.zeros(((dones == True).sum(), 1), dtype=torch.float32, device="cuda:0")
        active_masks[dones_env == True] = torch.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device="cuda:0")

        bad_masks = torch.tensor(
            [
                [
                    [0.0] if info[agent_id]['bad_transition'] else [1.0] 
                    for agent_id in range(self.num_agents)
                ] 
                for info in infos
            ],
            dtype=torch.float32,
            device="cuda:0"  # Optional: specify device if needed (e.g., 'cuda' or 'cpu')
        )        

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states.unsqueeze(-2), rnn_states_critic.unsqueeze(-2), actions, action_log_probs, values, rewards, masks, bad_masks,
                            active_masks, available_actions, edge_index, adjcency_matrix)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = torch.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0
        batch_edge_index = None
        avg_node_degree = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
        eval_share_obs = torch.tensor(eval_share_obs, dtype=torch.float32, device="cuda:0")
        eval_available_actions = torch.tensor(eval_available_actions, dtype=torch.float32, device="cuda:0")

        eval_rnn_states = torch.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.n_embd), dtype=torch.float32, device="cuda:0")
        eval_masks = torch.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")

        while True:
            if self.all_args.iterations > 0:
                if self.algorithm_name == 'mappo_gnn' or self.algorithm_name == 'mappo_dgnn' or self.algorithm_name == 'mappo_dgnn_dsgd':
                    edge_index = self.eval_envs.get_edge_index_matrix()
                    edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
                    batch_edge_index = self.get_batch_edge_index(edge_index)

                    avg_node_degree = batch_edge_index.shape[1]/self.num_agents
            
                    x = self.trainer.policy.transformer.obs_encoder(eval_obs, batch_edge_index)

                    eval_obs = torch.cat([eval_obs,x],dim=-1).detach()
                    # eval_obs = eval_obs.cpu().detach().numpy()
            
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(eval_share_obs,
                                        eval_obs,
                                        eval_rnn_states,
                                        eval_masks,
                                        eval_available_actions,
                                        deterministic=True,
                                        batched_edge_index=batch_edge_index)
            eval_actions = eval_actions.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
            eval_rnn_states = eval_rnn_states.reshape(self.n_eval_rollout_threads, self.num_agents, -1)
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
            eval_share_obs = torch.tensor(eval_share_obs, dtype=torch.float32, device="cuda:0")
            eval_dones = torch.tensor(eval_dones, dtype=torch.float32, device="cuda:0")
            eval_available_actions = torch.tensor(eval_available_actions, dtype=torch.float32, device="cuda:0")

            one_episode_rewards.append(eval_rewards)
            eval_dones_env = torch.all(eval_dones, dim=1)
            eval_rnn_states[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_agents, self.n_embd), dtype=torch.float32, device="cuda:0")

            eval_masks = torch.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=torch.float32, device="cuda:0")
            eval_masks[eval_dones_env == True] = torch.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=torch.float32, device="cuda:0")

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

                    # self.eval_envs.save_replay()

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                    wandb.log({"avg_node_degree": avg_node_degree}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
