import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import random
from collections import deque


"""
DQN implementation is based on:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl.ipynb
w. necessary adaptations to handle non-image based observations.
"""


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque([], maxlen=buffer_size)

    def push(self, experience):
        """Save experience in buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, input_dims, n_actions, batch_size,
                 gamma, epsilon, device):
        super(QNet, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

        # define layers (MLP)
        '''self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        72 %
        '''
        '''
        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_actions)
        '''

        self.fc1 = nn.Linear(input_dims, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        # print('fc1:', out.shape)
        out = F.relu(self.fc2(out))
        # print('fc2:', out.shape)
        q = F.relu(self.fc3(out))
        # print('fc3:', q.shape)
        return q

    def sample_action(self, state, current_episode):
        # epsilon decay
        self.epsilon = max(0.1, 0.9 - current_episode / 10000)

        if random.random() > self.epsilon:
            # choose action index w. highest Q-value
            with torch.no_grad():
                q = self.forward(state)
                return torch.argmax(q, dim=1).item()
        else:
            # choose random action based on decaying epsilon
            # which results in more exploration early on and more exploitation later.
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long).item()