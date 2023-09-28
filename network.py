import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# control camera by velocity

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape[0]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape[0]).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.shape[0]), std=0.01),
        )

    def get_value(self, x):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        # Convert observation to tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        
        #logits = self.actor(x)
        #probs = torch.distributions.Categorical(logits=logits)
        #if action is None:
        #    action = probs.sample()
        
        cov_var = torch.full(size=(self.env.action_space.shape[0],), fill_value=0.5)
        cov_mat = torch.diag(cov_var)

        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(x)

        # Create a Multivariate Normal Distribution
        dist = torch.distributions.MultivariateNormal(mean, cov_mat)

        # Sample an action from the distribution
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.detach().numpy(), log_prob.detach(), self.critic(x)
    
    def get_action(self, x):
        x = torch.tensor(x, dtype=torch.float)
        return self.actor(x).detach().numpy()