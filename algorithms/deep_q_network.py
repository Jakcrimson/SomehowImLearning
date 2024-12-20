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
import json
import seaborn as sns

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

        # Metrics dict
        self.metrics = {
            "cumulative_reward": [],
            "success_rate": [],
            "time_to_success": [],
            "loss": []
        }

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
        actions = torch.clamp(actions, 0, self.action_dim - 1)  # like np.clip kinda
        q_values = self.q_network(states).gather(1, actions)  # the network predicts the q-values from the given action

        # Compute target Q-values
        with torch.no_grad():  # validation-ish
            q_next = self.target_network(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, q_target)  # this is the TD error.

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.metrics["loss"].append(loss.item())

    def train(self, episodes=500, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99995, max_steps=5000):
        epsilon = epsilon_start
        successes = 0
        training_reward = 0
        t = trange(episodes, desc="Training", leave=True)
        for episode in t:
            state = self.env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, epsilon)
                next_state, reward, done = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.update()

                if self.env.is_successful(state):
                    successes += 1

                if isinstance(self.env, PendulumEnvironment):
                    done = self.gamma**self.env.steps > 1e-4

                if done:
                    break

            # Target network update
            if episode % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # Update exploration factor
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            # update the tqdm
            t.set_description(f"Reward: {training_reward:.3f}, Epsilon: {epsilon:.4f}")
            t.update()
        
            # Log metrics at the end of each episode
            training_reward += total_reward
            self.metrics['cumulative_reward'].append(training_reward)


    def plot_training(self, env_name, nb_ep):
        sns.set_theme(style="whitegrid")  # Set a professional theme
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))  # Create 2x2 subplots
        fig.suptitle(f"Training Metrics for {env_name} | Mean Cummulative Reward {np.mean(self.metrics["cumulative_reward"])}", fontsize=16, fontweight='bold')

        # Plot Cumulative Reward
        ax1.plot(self.metrics["cumulative_reward"], label="Cumulative Reward", color="#4C72B0", linewidth=2)
        ax1.set_title("Cumulative Reward", fontsize=14)
        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Reward", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.5)

        # Plot Loss
        ax2.plot(self.metrics["loss"], label="Loss", color="#8172B2", linewidth=2)
        ax2.set_title("Loss", fontsize=14)
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        save_path = f"./results/{env_name}_{nb_ep}_ep_metrics_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as high-res image
        plt.show()
        print(f"Metric plot saved to {save_path}")

    def save_metrics(self, metrics, filepath):
        metrics["mean_cumulative_reward"] = np.mean(metrics["cumulative_reward"])
        # Convert any NumPy arrays to lists for JSON serialization
        serializable_metrics = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in metrics.items()
        }
        with open(filepath, "w") as f:
            json.dump(serializable_metrics, f, indent=4)

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

    agent = DQNAgent(env, gamma=0.99, lr=0.001, hidden_layers=(5, 5), batch_size=64)
    agent.train(episodes=nb_episodes, max_steps=env.max_steps)
    agent.save_model(f"./models/dqn_{env_name}_{nb_episodes}_ep.pth")
    # agent.save_metrics(agent.metrics, f"./results/{env_name}_{nb_episodes}_ep_metrics.json")
    agent.plot_training(env_name=env_name, nb_ep=nb_episodes)
