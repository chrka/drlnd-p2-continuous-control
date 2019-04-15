import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q-value estimator"""

    def __init__(self, state_size, action_size, layer1=16, layer2=16):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            layer1: Size of input layer
            layer2: Size of hidden layer
        """
        super().__init__()

        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, action_size)

    def save_weights(self, path):
        """Save network weights.

        Args:
            path (string): File to save to"""
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """Load network weights.

        Args:
            path (string): File to load weights from"""
        self.load_state_dict(torch.load(path))

    def forward(self, state):
        """Maps state to action values

        Args:
            state (torch.Tensor): State (or rows of states)

        Returns:
            torch.Tensor: Tensor of action values for state(s)"""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
