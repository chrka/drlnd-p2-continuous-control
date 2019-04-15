import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from replay import ReplayBuffer


class Agent(object):
    """DQN Agent that interacts and learns from the environment."""

    def __init__(self, state_size, action_size, device,
                 replay_buffer_size=int(1e5), batch_size=64,
                 discount_factor=0.99, soft_update=1e-3,
                 learning_rate=5e-4, update_every=4,
                 **kwargs):
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
            **kwargs: Arguments describing the QNetwork
        """
        self.state_size = state_size
        """Dimension of each state"""

        self.action_size = action_size
        """Dimension of each action"""

        self.device = device
        """Device to use for calculations"""

        # Parameters
        self.batch_size = batch_size
        """Size of experience batches during training"""

        self.discount_factor = discount_factor
        """Discount factor (gamma)"""

        self.soft_update = soft_update
        """Soft update coefficient (tau)"""

        self.update_every = update_every
        """Steps between updating the network"""

        # Q Networks
        self.target_network = QNetwork(state_size, action_size, **kwargs) \
            .to(device)
        """Target Q-Network"""

        self.local_network = QNetwork(state_size, action_size, **kwargs) \
            .to(device)
        """Local Q-Network"""

        self.optimizer = optim.Adam(self.local_network.parameters(),
                                    lr=learning_rate)
        """Optimizer used when training the Q-network."""

        # Memory
        self.memory = ReplayBuffer(replay_buffer_size, batch_size, device)

        # Time step
        self.t_step = 0
        """Current time step"""

    def save_weights(self, path):
        """Save local network weights.

        Args:
            path (string): File to save to"""
        self.local_network.save_weights(path)

    def load_weights(self, path):
        """Load local network weights.

        Args:
            path (string): File to load weights from"""
        self.local_network.load_weights(path)

    def act(self, state, eps=0.):
        """Returns action for given state according to the current policy
            
        Args:
            state (np.ndarray): Current state
            eps (float): Probability of selecting random action (epsilon)
            
        Returns:
            int: Epsilon-greedily selected action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Temporarily set evaluation mode (no dropout &c) & turn off autograd
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        self.local_network.train()

        # Select action epsilon-greedily
        if random.random() > eps:
            return np.argmax(action_values.cpu().detach().numpy())
        else:
            return random.choice(np.arange(self.action_size))

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

        # Learn if at update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Check that we have enough stored experiences
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """Update Q-network using given experiences

        Args:
            experiences (Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                SARS'+done tuple
        """
        states, actions, rewards, next_states, dones = experiences

        # Predicted Q values from target model for next states
        # (NB. torch.max returns tuple (max, argmax)
        q_target_next = self.target_network(next_states).max(dim=1,
                                                             keepdim=True)[0]

        # Computed target Q values for current state
        q_target = rewards + self.discount_factor * q_target_next * (1 - dones)

        # Predicted Q values from local model for current state
        q_local = self.local_network(states).gather(dim=1, index=actions)

        loss = F.mse_loss(q_local, q_target)

        # Update local network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        soft_update(self.local_network, self.target_network,
                    self.soft_update)


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
