import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from environments.car_hill import CarHillEnvironment
from environments.inverted_pendulus import PendulumEnvironment
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns
import json
import sys


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))  # Output probabilities for actions
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)


class REINFORCEAgent:
    def __init__(self, env, gamma=0.99, lr=0.001, hidden_layers=(64, 64)):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = len(env.actions)
        self.actions = env.actions
        self.gamma = gamma
        self.lr = lr
        self.hidden_layers = hidden_layers

        # Policy Network
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim, hidden_layers)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)

        # Metrics dict
        self.metrics = {
            "cumulative_reward": [],
            "loss": []
        }

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        action = np.random.choice(self.actions, p=action_probs.cpu().detach().numpy().flatten())
        return action

    def compute_returns(self, rewards):
        """Compute discounted rewards (returns) for an episode."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_policy(self, states, actions, returns):
        states = torch.FloatTensor(states).to(self.device)
        actions = [self.actions.index(a) for a in actions]
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Compute log probabilities of the actions
        log_probs = torch.log(self.policy_network(states))
        log_action_probs = log_probs[range(len(actions)), actions]

        # Policy loss
        loss = -(log_action_probs * returns).mean()

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["loss"].append(loss.item())

    def train(self, episodes=500, max_steps=5000):
        t = trange(episodes, desc="Training", leave=True)
        cumulative_reward = 0
        for episode in t:
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []

            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                state = next_state
                if isinstance(self.env, PendulumEnvironment):
                    done = self.gamma**self.env.steps > 1e-4
                if done:
                    break

            # Compute returns and update policy
            returns = self.compute_returns(episode_rewards)
            self.update_policy(episode_states, episode_actions, returns)

            # Log metrics
            cumulative_reward += sum(episode_rewards)
            self.metrics['cumulative_reward'].append(cumulative_reward)
            t.set_description(f"Episode {episode+1}: Reward: {cumulative_reward:.3f}")
            t.update()

    def plot_training(self, env_name, nb_ep):
        sns.set_theme(style="whitegrid")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
        fig.suptitle(f"Training Metrics for {env_name} | Mean Cumulative Reward {np.mean(self.metrics['cumulative_reward']):.2f}", fontsize=16, fontweight='bold')

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
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = f"./results/reinforce_{env_name}_{nb_ep}_ep_metrics_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Metric plot saved to {save_path}")

    def save_model(self, path):
        torch.save({
            'network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'lr': self.lr,
                'hidden_layers': self.hidden_layers
            },
            'version': '1.0'
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded with hyperparameters: {checkpoint['hyperparameters']}")

if __name__ == "__main__":
    env_name = sys.argv[1]
    nb_episodes = int(sys.argv[2])

    if env_name == "car":
        env = CarHillEnvironment()
    elif env_name == "pendulum":
        env = PendulumEnvironment()
    else:
        raise ValueError("Unknown environment")

    agent = REINFORCEAgent(env, gamma=0.99, lr=0.001, hidden_layers=(64, 64))
    agent.train(episodes=nb_episodes, max_steps=env.max_steps)
    agent.save_model(f"./models/reinforce_{env_name}_{nb_episodes}_ep.pth")
    agent.plot_training(env_name=env_name, nb_ep=nb_episodes)
