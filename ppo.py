from network import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

# hyperparameters = {
#     'n_steps': 128,
#     'n_timesteps_per_batch': 4800,
#     'max_timesteps_per_episode': 1600,
#     'n_updates_per_iteration': 5,
#     'gamma': 0.99,
#     'lr': 2.5e-4,
#     'clip_range': 0.2,
#     'anneal_lr': False,
#     'cuda': True,

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        pass


class PPO:
    def __init__(self, env):
        hyperparameters = {
            'n_steps': 128,
            'n_timesteps_per_batch': 4800,
            'max_timesteps_per_episode': 1600,
            'n_updates_per_iteration': 5,
            'gamma': 0.99,
            'lr': 2.5e-4,
            'clip_range': 0.2,
            'anneal_lr': False,
            'cuda': True
        }
        self._init_hyperparameters(hyperparameters)
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Agent(env).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), self.lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), self.lr, eps=1e-5)

    def learn(self, total_timesteps):
        t_so_far = 0 # timesteps simulated so far
        
        while t_so_far < total_timesteps: # Algorithm Step 2
            # Algorithm Step 3: Collect a rollout batch
            batch_obs, batch_actions, batch_logprobs, batch_rtgs, batch_lens = self.rollout()

            # Increment timesteps simulated this batch so far
            t_so_far += np.sum(batch_lens)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_actions)

            # Algorithm Step 5: Calculate advantage estimates
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # Algorithm Step 6: PPO update
            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_actions)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_logprobs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * A_k
                
                # Calculate actor and critic losses
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                #print(actor_loss, critic_loss)
                # Calculate gradients and perform backward propagation for actor network
                #self.actor_optimizer.zero_grad()
                #actor_loss.backward(retain_graph=True)
                #self.actor_optimizer.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
    def rollout(self):
        batch_obs = []       # batch observations 
        batch_actions = []   # batch actions
        batch_logprobs = []  # log probs of each action
        batch_rewards = []   # batch rewards
        batch_rtgs = []      # batch rewards-to-go
        batch_lens = []      # episodic lengths in batch

        t = 0
        while t < self.n_timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            obs, _ = self.env.reset()
            #obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 # increment timestep counter

                # Collect observation
                batch_obs.append(obs)

                # Calculate policy
                action, logprob, _ = self.agent.get_action_and_value(obs)
                obs, reward, done, _, _ = self.env.step(action)

                # Collect reward, action, and logprob
                ep_rews.append(reward)
                batch_actions.append(action)
                batch_logprobs.append(logprob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float).to(self.device)
        batch_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float).to(self.device)

        # Compute rewards-to-go, to be targets for the value function
        batch_rtgs = self.compute_rtgs(batch_rewards)

        # Return the batch data
        return batch_obs, batch_actions, batch_logprobs, batch_rtgs, batch_lens

    def evaluate(self, batch_obs, batch_actions):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.agent.get_value(batch_obs).squeeze()

        # Calulate the log probabilities of batch actions using self.agent
        _, logprobs, _ = self.agent.get_action_and_value(batch_obs, batch_actions)
        return V, logprobs

    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rewards in reversed(batch_rewards):
                discounted_reward = 0 # The discounted reward so far
                
                for reward in reversed(ep_rewards):
                    discounted_reward = reward + discounted_reward * self.gamma
                    batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor   
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs

    def _init_hyperparameters(self, hyperparameters):
        for param, val in hyperparameters.items():
            setattr(self, param, val)
    
    def get_action(self, x):
        return self.agent.get_action(x)