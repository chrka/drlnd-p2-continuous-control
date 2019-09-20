import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor network (implements policy)"""

    def __init__(self, state_size, action_size, layer1=256, layer2=128):
        super().__init__()

        self.fc1 = nn.Linear(state_size, layer1)
        self.fc2 = nn.Linear(layer1, layer2)
        self.fc3 = nn.Linear(layer2, action_size)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights with random values"""
        # torch.nn.init.xavier_normal_(self.fc1.weight,
        #                              gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal_(self.fc2.weight,
        #                              gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal_(self.fc3.weight,
        #                              gain=torch.nn.init.calculate_gain('tanh'))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

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

    def __init__(self, state_size, action_size, layer0=256, layer1=128):
        super().__init__()

        self.fc0 = nn.Linear(state_size, layer0)
        self.fc1 = nn.Linear(layer0 + action_size, layer1)
        # self.fc1 = nn.Linear(state_size + action_size, layer1)
        self.fc2 = nn.Linear(layer1, 1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights with random values"""
        # torch.nn.init.xavier_normal_(self.fc0.weight,
        #                              gain=torch.nn.init.calculate_gain(
        #                                  'leaky_relu'))
        # torch.nn.init.xavier_normal_(self.fc1.weight,
        #                              gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal_(self.fc2.weight,
        #                              gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal_(self.fc3.weight,
        #                              gain=torch.nn.init.calculate_gain('relu'))
        # torch.nn.init.xavier_normal_(self.fc4.weight,
        #                              gain=torch.nn.init.calculate_gain('linear'))
        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Maps (state, action) to Q value

        Args:
            state (torch.Tensor): State (or rows of states)

        Returns:
            torch.Tensor: Tensor of action values for state(s)"""
        x = self.fc0(state)
        x = F.relu(x)
        # x = state
        x = torch.cat((x, action), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
