import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from environments.car_hill import CarHillEnvironment
from environments.inverted_pendulus import PendulumEnvironment
from utils import QNetwork, ReplayBuffer
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

"""THIS IS AN ONLINE ALGORITHM

"""

class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=0.001, hidden_layers=(64, 64), batch_size=64, buffer_capacity=1000000, target_update_freq=5):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = len(env.actions)
        self.actions = env.actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_layers)
        self.target_network = QNetwork(self.state_dim, self.action_dim, hidden_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        # Logging
        self.rewards_log = []
        self.loss_log = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        action_index = torch.argmax(q_values).item()
        return self.actions[action_index]  # Map index back to action


    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a minibatch
        minibatch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = [self.actions.index(a) for a in actions]  # Map actions to indices
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        actions = torch.clamp(actions, 0, self.action_dim - 1) # like np.clip kinda 
        q_values = self.q_network(states).gather(1, actions) # the network predicts the q-values from the given action

        # Compute target Q-values
        with torch.no_grad(): # validation-ish
            q_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, q_target) # this is the TD error.

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_log.append(loss.item())

    def train(self, episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, max_steps=5000):
        epsilon = epsilon_start
        t = trange(episodes, desc="Training", leave=True)
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, epsilon)
                next_state, reward, done = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.update()
                if isinstance(self.env, PendulumEnvironment):
                    done = self.gamma**self.env.steps > 1e-4
                if done:
                    break

            if episode % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict()) #update the target network for a validation that isn't late on the learning or so.

            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            self.rewards_log.append(total_reward)
            t.set_description(f"Reward : {total_reward:.3f}, Epsilon : {epsilon:.4f}")
            t.update()

        self.plot_training()

    def plot_training(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_log, label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_log, label="Loss")
        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/dqn_training_{sys.argv[1]}_{sys.argv[2]}.png")
        plt.show()

    def save_model(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(torch.load(path, weights_only=True))
        self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    env_name = sys.argv[1]
    nb_episodes = int(sys.argv[2])

    if env_name == "car":
        env = CarHillEnvironment()
    elif env_name == "pendulum":
        env = PendulumEnvironment()
    else:
        raise ValueError("Unknown environment")

    agent = DQNAgent(env, gamma=0.99, lr=0.001, hidden_layers=(5,5), batch_size=64)
    agent.train(episodes=nb_episodes, max_steps=env.max_steps)
    agent.save_model(f"./models/dqn_{env_name}_{nb_episodes}_ep.pth")
