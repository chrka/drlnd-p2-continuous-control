import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network"""

    def __init__(self, state_size, action_size, layer1=16):
        super().__init__()

        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, action_size)

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
        x = F.tanh(x)

        return x


class Actor(nn.Module):
    """Actor network (implements policy)"""

    def __init__(self, state_size, action_size, layer1=16, layer2=8):
        super().__init__()

        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, action_size)

    def initialize_parameters(self):
        """Initializes parameters with random values"""
        # TODO: Implement this
        pass

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
        x = F.tanh(x)

        return x


class Critic(nn.Module):
    """Critic network (estimates Q-values)"""

    def __init__(self, state_size, action_size, layer1=32, layer2=16, layer3=8):
        super().__init__()

        self.fc1 = nn.Linear(state_size + action_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, layer3)
        self.fc4 = nn.Linear(layer3, 1)

    def initialize_parameters(self):
        """Initializes parameters with random values"""
        # TODO: Implement this
        pass

    def forward(self, state, action):
        """Maps (state, action) to Q value

        Args:
            state (torch.Tensor): State (or rows of states)

        Returns:
            torch.Tensor: Tensor of action values for state(s)"""
        x = torch.cat(state, action)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x
