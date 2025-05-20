import torch.nn as nn
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(64, 64)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))  # Output layer for Q-values
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
