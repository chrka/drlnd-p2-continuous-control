import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import PolicyNetwork


class Agent(object):
    """Stochastic Policy Search Agent that interacts and learns from the
    environment."""

    def __init__(self, state_size, action_size, device, **kwargs):
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

        # Networks
        self.action_network = PolicyNetwork(state_size, action_size, **kwargs) \
            .to(device)
        """Action Network"""

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

    def act(self, state):
        """Returns action for given state according to the current policy
            
        Args:
            state (np.ndarray): Current state

        Returns:
            action (np.ndarray): Action tuple
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Temporarily set evaluation mode (no dropout &c) & turn off autograd
        self.action_network.eval()

        with torch.no_grad():
            action = self.action_network(state)

        # Resume training mode
        self.action_network.train()

        return action.cpu().detach().numpy()
