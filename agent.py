import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from noise import OUNoise
from replay import ReplayBuffer

# TODO: Make parameters of Agent
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
WEIGHT_DECAY = 0.0001  # L2 weight decay


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
        self.actor_local = Actor(state_size, action_size, **actor_args).to(
            device)
        self.actor_target = Actor(state_size, action_size, **actor_args).to(
            device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic network

        self.critic_local = Critic(state_size, action_size, **critic_args).to(
            device)
        self.critic_target = Critic(state_size, action_size, **critic_args).to(
            device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        # Noise process for exploration
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device)

    def save_weights(self, path):
        """Save local network weights.

        Args:
            path (string): File to save to"""
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, path)

    def load_weights(self, path):
        """Load local network weights.

        Args:
            path (string): File to load weights from"""
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

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

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn if due.
        Args:
            state (Tensor): Current state
            action (int): Chosen action
            reward (float): Resulting reward
            next_state (Tensor): State after action
            done (bool): True if terminal state
        """
        self.memory.add(state, action, reward, next_state, done)

        # Learn as soon as we have enough stored experiences
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # region Update Critic
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)

        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_targets)

        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()
        # endregion

        # region Update Actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # endregion

        # Update target networks
        soft_update(self.critic_local, self.critic_target, TAU)
        soft_update(self.actor_local, self.actor_target, TAU)