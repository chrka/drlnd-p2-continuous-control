import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor network (implements policy)"""

    def __init__(self, state_size, action_size, layer1=128, layer2=64):
        super().__init__()

        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, action_size)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights with random values"""
        torch.nn.init.xavier_normal_(self.fc1.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.fc2.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.fc3.weight,
                                     gain=torch.nn.init.calculate_gain('tanh'))

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

    def __init__(self, state_size, action_size, layer0=64,
                 layer1=128, layer2=64, layer3=32):
        super().__init__()

        self.fc0 = nn.Linear(state_size, layer0)
        self.fc1 = nn.Linear(layer0 + action_size, layer1)
        # self.fc1 = nn.Linear(state_size + action_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, layer3)
        self.fc4 = nn.Linear(layer3, 1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights with random values"""
        torch.nn.init.xavier_normal_(self.fc0.weight,
                                     gain=torch.nn.init.calculate_gain(
                                         'leaky_relu'))
        torch.nn.init.xavier_normal_(self.fc1.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.fc2.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.fc3.weight,
                                     gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(self.fc4.weight,
                                     gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        """Maps (state, action) to Q value

        Args:
            state (torch.Tensor): State (or rows of states)

        Returns:
            torch.Tensor: Tensor of action values for state(s)"""
        x = self.fc0(state)
        x = F.leaky_relu(x)
        # x = state
        x = torch.cat((x, action), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x
