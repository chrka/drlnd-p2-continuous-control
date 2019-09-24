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


# TODO: Add tests...
# TODO: In case we want to adjust `a` (for $p_i^a$ probs) during training,
#       it makes sense to keep an original and one raised tree and recalc as
#       needed.
class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity

        # Round capacity to 2^n (could be done using lb instead)
        self.tree_depth = 1
        self.actual_capacity = 1
        while self.actual_capacity < capacity:
            self.actual_capacity *= 2
            self.tree_depth += 1

        self.tree_nodes = [np.zeros(2 ** i) for i in range(self.tree_depth)]
        self.start_index = -1

    def append(self, p):
        self.start_index = (self.start_index + 1) % self.capacity
        self.set(self.start_index, p)

    def get(self, i):
        return self.tree_nodes[-1][i]

    def set(self, i, p):
        self.tree_nodes[-1][i] = p

        # Update sums
        for j in range(self.tree_depth - 2, -1, -1):
            i //= 2
            self.tree_nodes[j][i] = (self.tree_nodes[j + 1][2 * i] +
                                     self.tree_nodes[j + 1][2 * i + 1])

    def set_multiple(self, indices, ps):
        # TODO: Smarter update which sets all and recalculates range as needed
        for i, p in zip(indices, ps):
            self.set(i, p)

    def total_sum(self):
        return self.tree_nodes[0][0]

    def index(self, p):
        i = 0
        for j in range(self.tree_depth - 1):
            left = self.tree_nodes[j + 1][2 * i]
            if p < left:
                i = 2 * i
            else:
                p = p - left
                i = 2 * i + 1
        return i

    def sample(self, size):
        indices = []
        bins = np.linspace(0, self.total_sum(), size + 1)
        for a, b in zip(bins, bins[1:]):
            # There's a chance we'll sample the same index more than once
            indices.append(self.index(np.random.uniform(a, b)))

        return indices


class PrioritizedExperienceReplayBuffer(object):
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

        self.priorities = SumTree(capacity=buffer_size)
        """Stored priorities"""

        self.device = device
        """Device to be used for tensors."""

    def add(self, state, action, reward, next_state, done, priority):
        """Add an experience to memory.

        Args:
            state (Tensor): Current state
            action (int): Chosen action
            reward (float): Resulting reward
            next_state (Tensor): State after action
            done (bool): True if terminal state
            priority (float): Priority of experience (abs TD-error)
        """
        self.memory.append(Experience(state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self):
        """Returns a sample batch of experiences from memory.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: SARS'+done tuple"""
        indices = self.priorities.sample(self.batch_size)

        # TODO: Make this whole indexing and unpacking thing more efficient
        experiences = [self.memory[i] for i in indices]

        device = self.device

        state_list = [e.state for e in experiences if e is not None]
        action_list = [e.action for e in experiences if e is not None]
        reward_list = [e.reward for e in experiences if e is not None]
        next_state_list = [e.next_state for e in experiences if e is not None]
        done_list = [e.done for e in experiences if e is not None]
        # TODO: Add `__getitem__` to SumTree
        priorities_list = [self.priorities.get(i) for i in indices]

        states = torch.from_numpy(np.vstack(state_list)).float().to(device)
        actions = torch.from_numpy(np.vstack(action_list)).float().to(device)
        rewards = torch.from_numpy(np.vstack(reward_list)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_state_list)).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack(done_list).astype(np.uint8)).float() \
            .to(device)
        priorities = torch.from_numpy(np.vstack(priorities_list)).float() \
            .to(device)


        return (indices, states, actions, rewards, next_states, dones,
                priorities)

    def priority_sum(self):
        return self.priorities.total_sum()

    def set_priorities(self, i, p):
        # NB. Works with multiple indices and priorities
        self.priorities.set_multiple(i, p)

    def __len__(self):
        """Returns the current number of stored experiences.

        Returns:
            int: Number of stored experiences"""
        return len(self.memory)
