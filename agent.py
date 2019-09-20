import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import PolicyNetwork, Actor, Critic
from noise import OUNoise
from replay import ReplayBuffer

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

def soft_update(local_network, target_network, tau):
    """Soft update of target model parameters.
    Update weights as linear combination of local and target models,
    $w = \tau w^- + (1 - \tau) w.
    Args:
        local_model (QNetwork): Local model, source of changes
        target_model (QNetwork): Target model, receiver of changes
    """
    for target_param, local_param in zip(target_network.parameters(),
                                         local_network.parameters()):
        target_param.data.copy_(tau * local_param.data +
                                (1.0 - tau) * target_param.data)

class Agent(object):
    """DDPG Agent that interacts and learns from the environment."""

    def __init__(self, state_size, action_size, device,
                 actor_args={}, critic_args={}):
        """Initializes the DQN agent.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            device (torch.device): Device to use for calculations
            replay_buffer_size (int): Size of replay buffer
            batch_size (int): Size of experience batches during training
            discount_factor (float): Discount factor (gamma)
            soft_update (float): Soft update coefficient (tau)
            learning_rate (float): Learning rate (alpha)
            update_every (int): Steps between updating the network
            **kwargs: Arguments describing the networks
        """
        self.state_size = state_size
        """Dimension of each state"""

        self.action_size = action_size
        """Dimension of each action"""

        self.device = device
        """Device to use for calculations"""

        # Parameters

        # Actor network
        self.actor_local = Actor(state_size, action_size, **actor_args).to(device)
        self.actor_target = Actor(state_size, action_size, **actor_args).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic network

        self.critic_local = Critic(state_size, action_size, **critic_args).to(device)
        self.critic_target = Critic(state_size, action_size, **critic_args).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process for exploration
        self.noise = OUNoise(action_size)


        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device)

    def save_weights(self, path):
        """Save local network weights.

        Args:
            path (string): File to save to"""
        self.action_network.save_weights(path)

    def load_weights(self, path):
        """Load local network weights.

        Args:
            path (string): File to load weights from"""
        self.action_network.load_weights(path)

    def act(self, state, add_noise=True):
        """Returns action for given state according to the current policy
            
        Args:
            state (np.ndarray): Current state

        Returns:
            action (np.ndarray): Action tuple
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Temporarily set evaluation mode (no dropout &c) & turn off autograd
        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().detach().numpy()

        # Resume training mode
        self.actor_local.train()

        # Add noise if exploring
        if add_noise:
            action += self.noise.sample()
            # The noise might take us out of range
            action = np.clip(action, -1, 1)

        return action

    def randomly_displaced(self, noise_scale):
        """Create copy with random displacement of weights. """
        # TODO: Make more efficient, not having to create weights in first place
        displaced = Agent(self.state_size, self.action_size, self.device,
                          **self.network_parameters)
        dist = torch.distributions.uniform.Uniform(-noise_scale, noise_scale)
        for displaced_param, source_param in zip(
                self.action_network.parameters(),
                displaced.action_network.parameters()):
            displaced_param.data.copy_(source_param.data +
                                       dist.sample(source_param.size()))
        return displaced

