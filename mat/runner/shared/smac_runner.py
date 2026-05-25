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
#from .consensus import Consensus

#from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
#from deepseek_vl.utils.io import load_pil_images

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

        self.eye = torch.eye(self.num_agents, device="cuda:0").unsqueeze(0)
        self.eye = self.eye / torch.norm(self.eye, p='fro')  # Normalize entire matrix
        self.disconnected_net = 0

        if self.all_args.algorithm_name == 'vlm_mat_gnn':
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            from deepseek_vl.utils.io import load_pil_images

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

        edge_index = torch.tensor(edge_index, dtype=torch.float32, device="cuda:0")
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

    def get_unique_batch_edge_index(self, edge_index):
        """
        Converts a padded multi-environment edge index into a single batched edge index
        where each agent has exactly one outgoing connection.
        
        Args:
            edge_index: Padded array of shape [batch_size, 2, max_edges]
                    (invalid edges marked with -1 in the 2nd row).
        
        Returns:
            batched_edge_index: Merged edge index of shape [2, total_edges]
        """
        batch_size = edge_index.shape[0]
        batched_edges = []

        for i in range(batch_size):
            # Step 1: Get all valid edges for this batch [2, num_valid_edges]
            valid_mask = edge_index[i, 1, :] != -1
            valid_edges = edge_index[i, :, valid_mask].T  # Keep as [2, N] shape
            
            # Step 2: For each agent, select exactly one outgoing edge
            if valid_edges.size == 0:
                continue  # Skip if no valid edges
                
            # Get unique source nodes and select one edge per source
            source_nodes = valid_edges[0, :]
            unique_sources = np.unique(source_nodes)
            selected_indices = []
            
            for src in unique_sources:
                # Find all edges from this source
                src_edge_indices = np.where(source_nodes == src)[0]
                # Select first edge (or random if preferred)
                selected_idx = src_edge_indices[0]
                selected_indices.append(selected_idx)
            
            # Get the selected edges [2, num_selected_edges]
            selected_edges = valid_edges[:, selected_indices]
            
            # Step 3: Apply offset to node indices
            selected_edges[0, :] += i * self.num_agents
            selected_edges[1, :] += i * self.num_agents
            
            # Step 4: Collect the selected edges
            batched_edges.append(selected_edges)

        # Step 5: Concatenate along the second dimension
        if len(batched_edges) == 0:
            return np.zeros((2, 0), dtype=edge_index.dtype)
        return np.concatenate(batched_edges, axis=1)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                edge_index = self.envs.get_edge_index_matrix()
                batch_edge_index = self.get_batch_edge_index(edge_index)

                if batch_edge_index.shape[1] < (self.num_agents+ 2*(self.num_agents-1))*self.n_rollout_threads:
                    self.disconnected_net += 1
                                                
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data, batch_edge_index)
                #self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
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
    

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def format_obs(self, agent_obs):
        return ", ".join([f"{x:.4f}" for x in agent_obs.cpu().numpy()])
                
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
                
        '''edge_index = self.get_edge_index_deg_2(obs)
        edge_index = torch.from_numpy(edge_index).long().to(device="cuda:0")

        sorted_values, sorted_indices = torch.sort(edge_index[0])
        edge_index = edge_index[:, sorted_indices]'''
        
        if self.algorithm_name == 'mat_gnn':
            self.buffer.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.obs_dim+2*self.n_embd), dtype=np.float32)
        
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            #obs_embd = self.trainer.policy.transformer.obs_embedding(obs)

            edge_index = self.envs.get_edge_index_matrix()
            batch_edge_index = self.get_batch_edge_index(edge_index)
            batch_edge_index = torch.from_numpy(batch_edge_index).long().to(device="cuda:0")
            
            if batch_edge_index.shape[1] < (self.num_agents+ 2*(self.num_agents-1))*self.n_rollout_threads:
                self.disconnected_net += 1

            #available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device="cuda:0")
            #obs = torch.cat([obs, available_actions_tensor], dim=-1)
            
            x = self.trainer.policy.transformer.obs_encoder(obs, batch_edge_index)
            #x = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index)
                        
            if self.all_args.use_CNN_for_pi:
                # Apply to each agent, then concat:
                params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]) #.permute(0, 2, 1)
                params = self.trainer.policy.transformer.policy_processor(params).unsqueeze(0)
                params = torch.cat((F.gelu(params), self.eye), axis=-1)
            elif self.all_args.use_VAE_for_pi:
                #flat_params = self.trainer.policy.transformer.vae.flatten_weights(self.trainer.policy.transformer.decoder.mlp_)
                flat_params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1)
                print('flat_params shape = ', flat_params.shape)
                reconst_params, params_mu, params_logvar = self.trainer.policy.transformer.vae(flat_params)
                params = self.trainer.policy.transformer.vae.reparameterize(params_mu, params_logvar)
                params = torch.cat((params.unsqueeze(0), self.eye), axis=-1)
                
            else:
                params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1).unsqueeze(0)  # [emb_size * action_dim]
                params = torch.cat((F.gelu(params), self.eye), axis=-1)
                                    
            y = self.trainer.policy.transformer.policy_encoder(params.repeat(obs.shape[0],1,1), batch_edge_index)
            #y = self.trainer.policy.transformer.policy_encoder(params, self.trainer.policy.transformer.edge_index)
            y = y.expand_as(x)
            
            obs = torch.cat([obs,x,y],dim=-1).cpu().detach().numpy()

        elif self.algorithm_name == 'generative_mat_gnn':
            self.buffer.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 2*self.n_embd), dtype=np.float32)
        
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            #obs_embd = self.trainer.policy.transformer.obs_embedding(obs)

            edge_index = self.envs.get_edge_index_matrix()
            batch_edge_index = self.get_batch_edge_index(edge_index)
            batch_edge_index = torch.from_numpy(batch_edge_index).long().to(device="cuda:0")
            
            obs_hat = self.trainer.policy.transformer.obs_encoder(obs, batch_edge_index)
            #x = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index)

            action_preds = self.trainer.policy.transformer.generator(obs_hat)
            
            obs = torch.cat([obs_hat,action_preds],dim=-1).cpu().detach().numpy()

            action_preds = action_preds.cpu().detach().numpy()

            self.buffer.action_preds[0] = action_preds.copy()

        elif self.algorithm_name == 'vlm_mat_gnn':
            self.buffer.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 256), dtype=np.float32)

            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            vlm_action_embds = []

            for _ in range(self.n_rollout_threads):
                for i in range(self.num_agents):
                    input_texts = [f'query: Agent ({i})s enemy features are  {self.format_obs(obs[0, i, 4*17:4*17+6*5])},'
                        f'query: Agent ({i})s move features are  {self.format_obs(obs[0, i, 4*17+6*5:4*17+6*5+4])},'
                        f'query: Agent ({i})s own features are  {self.format_obs(obs[0, i, 4*17+6*5+4:4*17+6*5+21])},'
                        f'query: and Agent ({i})s agent_id features are  {self.format_obs(obs[0, i, 4*17+6*5+21:])}.'
                        f'query: The availabel actions for Agent ({i}) are  {self.format_obs(available_actions[0, i, :])}.'
                        f'query: What should be Agent ({i})s actions be?']

                    # Tokenize the input texts
                    batch_dict = self.trainer.policy.transformer.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device="cuda:0")

                    outputs = self.trainer.policy.transformer.model(**batch_dict)

                    embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                    # normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                vlm_action_embds.append(embeddings)            

            obs = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index).cpu().detach().numpy()

        elif self.algorithm_name == 'deepseek_vlm_mat_gnn':
            self.buffer.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 256), dtype=np.float32)

            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            vlm_obs = []

            for _ in range(self.n_rollout_threads):
                for agent_idx in range(self.num_agents):
                    obs_text = ", ".join([f"{x:.4f}" for x in obs[0, agent_idx, :].cpu().numpy()])

                    conversation = [
                        {
                            "role": "User",
                            "content": f"Agent {agent_idx} observation in the StarCraft games 5m_vs_6m map, which is a concatenation of ally_features, enemy_features, move_feats, own_feats and agent_id_feats is as follows: {obs_text}. what do you this is a valid action for this agent?",
                            "images": []
                        },
                        {
                            "role": "Assistant",
                            "content": ""
                        }
                    ]

                    # load images and prepare for inputs
                    pil_images = self.trainer.policy.transformer.load_pil_images(conversation)
                    prepare_inputs = self.trainer.policy.transformer.vl_chat_processor(
                            conversations=conversation,
                            images=pil_images,
                            force_batchify=True
                        ).to(self.trainer.policy.transformer.vl_gpt.device)

                    # run image encoder to get the image embeddings
                    inputs_embeds = self.trainer.policy.transformer.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                    # run the model to get the response
                    # run the model to get the response
                    '''
                    outputs = self.trainer.policy.transformer.vl_gpt.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=self.trainer.policy.transformer.tokenizer.eos_token_id,
                        bos_token_id=self.trainer.policy.transformer.tokenizer.bos_token_id,
                        eos_token_id=self.trainer.policy.transformer.tokenizer.eos_token_id,
                        max_new_tokens=512,
                        do_sample=False,
                        use_cache=True
                    )'''

                    with torch.no_grad():
                        outputs = self.trainer.policy.transformer.vl_gpt.language_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepare_inputs.attention_mask,
                            output_hidden_states=True
                        ).hidden_states[-1].mean(dim=1)

                    vlm_obs.append(outputs)

            #answer = self.trainer.policy.transformer.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            #print(f"{prepare_inputs['sft_format'][0]}", answer)

            vlm_obs = torch.stack(vlm_obs, dim=-2).float()
            vlm_obs = self.trainer.policy.transformer.obs_encoder(vlm_obs, self.trainer.policy.transformer.edge_index)

            obs = vlm_obs.cpu().detach().numpy()

        '''elif self.algorithm_name == 'mappo_consensus':

            for agent_id in range(self.num_agents):
                avg_state_consensus, obs_H2, obs_Hinfty, conv_steps = self.consensus.forward(obs[0,:], agent_id)
                observer_obs = np.array([[obs_H2, obs_Hinfty, 0, 0]])
                self.buffer[agent_id].share_obs[0] = observer_obs.copy()                
                self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()'''
            
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
        
        '''self.buffer.flat_params[0] = flat_params.unsqueeze(0)
        self.buffer.reconst_params[0] = reconst_params.unsqueeze(0)
        self.buffer.params_mu[0] = params_mu.unsqueeze(0)
        self.buffer.params_logvar[0] = params_logvar.unsqueeze(0)'''

    @torch.no_grad()
    def collect(self, step, batched_edge_index=None):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data, edge_index=None, action_preds=None, real_action_preds=None):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        
        if self.algorithm_name == 'mat_gnn':
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")  #[:,:,17*4:]
            #obs_embd = self.trainer.policy.transformer.obs_embedding(obs)
            
            #available_actions_tensor = torch.tensor(available_actions, dtype=torch.float32, device="cuda:0")
            #obs = torch.cat([obs, available_actions_tensor], dim=-1)
            
            x = self.trainer.policy.transformer.obs_encoder(obs, edge_index)
            #x = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index)
            
            if self.all_args.use_CNN_for_pi:
                # Apply to each agent, then concat:
                params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]) #.permute(0, 2, 1)
                params = self.trainer.policy.transformer.policy_processor(params).unsqueeze(0)
                params = torch.cat((F.gelu(params), self.eye), axis=-1)
            elif self.all_args.use_VAE_for_pi:
                #flat_params = self.trainer.policy.transformer.vae.flatten_weights(self.trainer.policy.transformer.decoder.mlp_)
                flat_params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1)
                reconst_params, params_mu, params_logvar = self.trainer.policy.transformer.vae(flat_params)
                params = self.trainer.policy.transformer.vae.reparameterize(params_mu, params_logvar)
                params = torch.cat((params.unsqueeze(0), self.eye), axis=-1)
            else:
                params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1).unsqueeze(0)
                params = torch.cat((F.gelu(params), self.eye), axis=-1)
        
            y = self.trainer.policy.transformer.policy_encoder(params.repeat(obs.shape[0],1,1), edge_index)
            #y = self.trainer.policy.transformer.policy_encoder(params, self.trainer.policy.transformer.edge_index)
            y = y.expand_as(x)
                    
            obs = torch.cat([obs,x,y],dim=-1).cpu().detach().numpy()

        elif self.algorithm_name == 'generative_mat_gnn':
            self.buffer.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 2*self.n_embd), dtype=np.float32)
        
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            actions_tensor = torch.tensor(actions, dtype=torch.float32, device="cuda:0")
            action_log_probs_tensor = torch.tensor(action_log_probs, dtype=torch.float32, device="cuda:0")
            actions_tensor = torch.cat((actions_tensor,action_log_probs_tensor),dim=-1)
            #obs_embd = self.trainer.policy.transformer.obs_embedding(obs)

            edge_index = self.envs.get_edge_index_matrix()
            batch_edge_index = self.get_batch_edge_index(edge_index)
            batch_edge_index = torch.from_numpy(batch_edge_index).long().to(device="cuda:0")
            
            obs_hat = self.trainer.policy.transformer.obs_encoder(obs, batch_edge_index)
            #x = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index)

            action_preds = self.trainer.policy.transformer.generator(obs_hat)
            
            obs = torch.cat([obs_hat,action_preds],dim=-1).cpu().detach().numpy()

            action_preds = action_preds.cpu().detach().numpy()

            real_action_preds = self.trainer.policy.transformer.policy_encoder(actions_tensor, batch_edge_index).cpu().detach().numpy()

        elif self.algorithm_name == 'vlm_mat_gnn':
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            vlm_obs = []

            for _ in range(self.n_rollout_threads):
                input_texts = [
                    f'query: Agent {i} observation in the StarCraft games 5m_vs_6m map, which is a concatenation of ally_features, enemy_features, move_feats, own_feats and agent_id_feats is as follows: {self.format_obs(obs[0, i, :])}. what do you think is a valid action for this agent?'
                    for i in range(self.num_agents)
                ]

                # Tokenize the input texts
                batch_dict = self.trainer.policy.transformer.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.trainer.policy.transformer.model.device)

                outputs = self.trainer.policy.transformer.model(**batch_dict)

                embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)

                obs = embeddings.unsqueeze(0)
                obs = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index).cpu().detach().numpy()


        elif self.algorithm_name == 'deepseek_vlm_mat_gnn':
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0")
            vlm_obs = []

            for _ in range(self.n_rollout_threads):
                for agent_idx in range(self.num_agents):
                    obs_text = ", ".join([f"{x:.4f}" for x in obs[0, agent_idx, :].cpu().numpy()])

                    conversation = [
                        {
                            "role": "User",
                            "content": f"Agent {agent_idx} observation in the StarCraft games 5m_vs_6m map, which is a concatenation of ally_features, enemy_features, move_feats, own_feats and agent_id_feats is as follows: {obs_text}. what do you this is a valid action for this agent?",
                            "images": []
                        },
                        {
                            "role": "Assistant",
                            "content": ""
                        }
                    ]
                    # load images and prepare for inputs
                    pil_images = self.trainer.policy.transformer.load_pil_images(conversation)
                    prepare_inputs = self.trainer.policy.transformer.vl_chat_processor(
                            conversations=conversation,
                            images=pil_images,
                            force_batchify=True
                        ).to(self.trainer.policy.transformer.vl_gpt.device)

                    # run image encoder to get the image embeddings
                    inputs_embeds = self.trainer.policy.transformer.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                    with torch.no_grad():
                        outputs = self.trainer.policy.transformer.vl_gpt.language_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepare_inputs.attention_mask,
                            output_hidden_states=True
                        ).hidden_states[-1].mean(dim=1)

                    vlm_obs.append(outputs)

            vlm_obs = torch.stack(vlm_obs, dim=-2).float()
            vlm_obs = self.trainer.policy.transformer.obs_encoder(vlm_obs, self.trainer.policy.transformer.edge_index)

            obs = vlm_obs.cpu().detach().numpy()

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        flat_params = flat_params.unsqueeze(0)
        reconst_params = reconst_params.unsqueeze(0)
        params_mu = params_mu.unsqueeze(0)
        params_logvar = params_logvar.unsqueeze(0)

        vae_loss = self.trainer.vae_loss(reconst_params, flat_params, params_mu, params_logvar)

        self.trainer.policy.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.trainer.policy.vae_optimizer.step()

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks,
                            active_masks, available_actions, action_preds, real_action_preds)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            '''edge_index = self.get_edge_index_deg_2(eval_obs)
            edge_index = torch.from_numpy(edge_index).long().to(device="cuda:0")'''
            
            if self.algorithm_name == 'mat_gnn':
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")   #[:,:,17*4:]
                #eval_obs_embd = self.trainer.policy.transformer.obs_embedding(eval_obs)

                edge_index = self.eval_envs.get_edge_index_matrix()
                batch_edge_index = self.get_batch_edge_index(edge_index)
                #batch_edge_index = torch.from_numpy(batch_edge_index).long().to(device="cuda:0")
                
                #available_actions_tensor = torch.tensor(eval_available_actions, dtype=torch.float32, device="cuda:0")
                #eval_obs = torch.cat([eval_obs, available_actions_tensor], dim=-1)   
        
                x = self.trainer.policy.transformer.obs_encoder(eval_obs, batch_edge_index)
                #x = self.trainer.policy.transformer.obs_encoder(eval_obs, self.trainer.policy.transformer.edge_index)

                if self.all_args.use_CNN_for_pi:
                    # Apply to each agent, then concat:
                    params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]) #.permute(0, 2, 1)
                    params = self.trainer.policy.transformer.policy_processor(params).unsqueeze(0)
                    params = torch.cat((F.gelu(params), self.eye), axis=-1)
                elif self.all_args.use_VAE_for_pi:
                    #flat_params = self.trainer.policy.transformer.vae.flatten_weights(self.trainer.policy.transformer.decoder.mlp_)
                    flat_params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1)
                    reconst_params, params_mu, params_logvar = self.trainer.policy.transformer.vae(flat_params)
                    params = self.trainer.policy.transformer.vae.reparameterize(params_mu, params_logvar)
                    params = torch.cat((params.unsqueeze(0), self.eye), axis=-1)
                else:
                    params = torch.stack([actor[-1].weight for actor in self.trainer.policy.transformer.decoder.mlp_]).view(self.num_agents,-1).unsqueeze(0)
                    params = torch.cat((F.gelu(params), self.eye), axis=-1)

                y = self.trainer.policy.transformer.policy_encoder(params.repeat(eval_obs.shape[0],1,1), batch_edge_index)
                #y = self.trainer.policy.transformer.policy_encoder(params, self.trainer.policy.transformer.edge_index)
                y = y.expand_as(x)
        
                eval_obs = torch.cat([eval_obs,x,y],dim=-1).cpu().detach().numpy()

            elif self.algorithm_name == 'generative_mat_gnn':
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
                #obs_embd = self.trainer.policy.transformer.obs_embedding(obs)

                edge_index = self.eval_envs.get_edge_index_matrix()
                batch_edge_index = self.get_batch_edge_index(edge_index)
                batch_edge_index = torch.from_numpy(batch_edge_index).long().to(device="cuda:0")
                
                #eval_obs_hat = self.trainer.policy.transformer.obs_encoder(eval_obs, batch_edge_index)
                eval_obs_hat = self.trainer.policy.transformer.obs_encoder(obs, self.trainer.policy.transformer.edge_index)

                action_preds = self.trainer.policy.transformer.generator(eval_obs_hat)
                
                eval_obs = torch.cat([eval_obs_hat,action_preds],dim=-1).cpu().detach().numpy()

            elif self.algorithm_name == 'vlm_mat_gnn':
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
                vlm_obs = []

                for _ in range(self.n_rollout_threads):
                    input_texts = [
                        f'query: Agent {i} observation in the StarCraft games 5m_vs_6m map, which is a concatenation of ally_features, enemy_features, move_feats, own_feats and agent_id_feats is as follows: {self.format_obs(eval_obs[0, i, :])}. what do you think is a valid action for this agent?'
                        for i in range(self.num_agents)
                    ]

                    # Tokenize the input texts
                    batch_dict = self.trainer.policy.transformer.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.trainer.policy.transformer.model.device)

                    outputs = self.trainer.policy.transformer.model(**batch_dict)

                    embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                    # normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                    eval_obs = embeddings.unsqueeze(0)
                    eval_obs = self.trainer.policy.transformer.obs_encoder(eval_obs, self.trainer.policy.transformer.edge_index).cpu().detach().numpy()

            elif self.algorithm_name == 'deepseek_vlm_mat_gnn':
                eval_obs = torch.tensor(eval_obs, dtype=torch.float32, device="cuda:0")
                vlm_obs = []

                for _ in range(self.n_rollout_threads):
                    for agent_idx in range(self.num_agents):
                        obs_text = ", ".join([f"{x:.4f}" for x in eval_obs[0, agent_idx, :].cpu().numpy()])

                        conversation = [
                            {
                                "role": "User",
                                "content": f"Agent {agent_idx} observation in the StarCraft games 5m_vs_6m map, which is a concatenation of ally_features, enemy_features, move_feats, own_feats and agent_id_feats is as follows: {obs_text}. what do you this is a valid action for this agent?",
                                "images": []
                            },
                            {
                                "role": "Assistant",
                                "content": ""
                            }
                        ]

                        # load images and prepare for inputs
                        pil_images = self.trainer.policy.transformer.load_pil_images(conversation)
                        prepare_inputs = self.trainer.policy.transformer.vl_chat_processor(
                                conversations=conversation,
                                images=pil_images,
                                force_batchify=True
                            ).to(self.trainer.policy.transformer.vl_gpt.device)

                        # run image encoder to get the image embeddings
                        inputs_embeds = self.trainer.policy.transformer.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                        with torch.no_grad():
                            outputs = self.trainer.policy.transformer.vl_gpt.language_model(
                                inputs_embeds=inputs_embeds,
                                attention_mask=prepare_inputs.attention_mask,
                                output_hidden_states=True
                            ).hidden_states[-1].mean(dim=1)

                        vlm_obs.append(outputs)

                vlm_obs = torch.stack(vlm_obs, dim=-2).float()
                vlm_obs = self.trainer.policy.transformer.obs_encoder(vlm_obs, self.trainer.policy.transformer.edge_index)

                eval_obs = vlm_obs.cpu().detach().numpy()
            
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                # self.eval_envs.save_replay()
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
