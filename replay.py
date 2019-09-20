import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "next_state",
                                     "done"])
"""Experience tuple."""


class ReplayBuffer(object):
    """Fixed-size buffer for storing experience tuples."""

    def __init__(self, buffer_size, batch_size, device):
        """Initialize the replay buffer.

        Args:
            buffer_size (int): Max number of stored experiences
            batch_size (int): Size of training batches
            device (torch.device): Device for tensors
        """
        self.batch_size = batch_size
        """Size of training batches"""

        self.memory = deque(maxlen=buffer_size)
        """Stored experiences."""

        self.device = device
        """Device to be used for tensors."""

    def add(self, state, action, reward, next_state, done):
        """Add an experience to memory.

        Args:
            state (Tensor): Current state
            action (int): Chosen action
            reward (float): Resulting reward
            next_state (Tensor): State after action
            done (bool): True if terminal state
        """
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        """Returns a sample batch of experiences from memory.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: SARS'+done tuple"""
        experiences = random.sample(self.memory, k=self.batch_size)

        device = self.device

        state_list = [e.state for e in experiences if e is not None]
        action_list = [e.action for e in experiences if e is not None]
        reward_list = [e.reward for e in experiences if e is not None]
        next_state_list = [e.next_state for e in experiences if e is not None]
        done_list = [e.done for e in experiences if e is not None]

        states = torch.from_numpy(np.vstack(state_list)).float().to(device)
        actions = torch.from_numpy(np.vstack(action_list)).float().to(device)
        rewards = torch.from_numpy(np.vstack(reward_list)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_state_list)).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack(done_list).astype(np.uint8)).float() \
            .to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Returns the current number of stored experiences.

        Returns:
            int: Number of stored experiences"""
        return len(self.memory)
