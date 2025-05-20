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
import os


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


class QValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super(QValueNetwork, self).__init__()
        layers=[]
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1)) # output a single value "q"
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

class ActorCriticAgent:
    def __init__(self, env, gamma=0.99, lr_actor=0.005, lr_critic=0.001, actor_hidden_layers=(64, 64), critic_hidden_layers=(64, 64)):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = len(env.actions)
        self.actions = env.actions
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_layers_actor = actor_hidden_layers
        self.hidden_layers_critic = critic_hidden_layers

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy network
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim, actor_hidden_layers).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr_actor)
        
        # value network
        self.value_network = QValueNetwork(self.state_dim, critic_hidden_layers).to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr_critic)                                                                                         

        # Metrics dict
        self.metrics = {
            "episode_rewards": [], # Changed from episode_rewards for clearer per-episode tracking
            "actor_loss": [],
            "critic_loss": [],
            "actions_sum_in_episode": [] # Renamed from 'actions' for clarity
        }

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        action = np.random.choice(self.actions, p=action_probs.cpu().detach().numpy().flatten())
        return action
    
    def greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state) # Shape [1, action_dim]
        action_idx = torch.argmax(action_probs, dim=1).item() # Get index of max prob
        action = self.actions[action_idx] # Map index to actual action value
        return action
    
    def compute_returns(self, rewards):
        """Compute discounted rewards (returns) for an episode."""
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return returns

    def update_networks(self, states_ep, actions_ep, returns_G_t_ep):
        states = torch.FloatTensor(np.array(states_ep)).to(self.device)
        # converts actions to indices
        action_indices = torch.LongTensor([self.actions.index(a) for a in actions_ep]).to(self.device)
        returns_G_t = torch.FloatTensor(returns_G_t_ep).to(self.device)

        # --- Critic Update ---
        # Predict V(s_t) for all states in the episode
        state_values_V_s_t = self.value_network(states).squeeze() # Output is [N, 1], squeeze to [N]


        # Critic loss: MSE(V(s_t), G_t)
        critic_loss = nn.MSELoss()(state_values_V_s_t, returns_G_t)

        self.value_optimizer.zero_grad()
        critic_loss.backward() # Calculate gradients for value_network parameters
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        # --- Actor Update ---
        # Advantages: A_t = G_t - V(s_t)
        # Detach state_values_V_s_t from graph: actor loss shouldn't propagate to critic.
        advantages = (returns_G_t - state_values_V_s_t).detach()

        # Compute log probabilities of the actions taken
        log_probs_all_actions = torch.log(self.policy_network(states) + 1e-6) # Add epsilon for numerical stability
        # Select log_probs for actions actually taken: log_probs_all_actions has shape [N, num_actions]
        # action_indices has shape [N]
        log_action_probs = log_probs_all_actions.gather(1, action_indices.unsqueeze(1)).squeeze() # Shape [N]
        
        # Policy loss (Actor loss)
        actor_loss = -(log_action_probs * advantages).mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward() # Calculate gradients for policy_network parameters
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        # Store losses
        self.metrics["actor_loss"].append(actor_loss.item())
        self.metrics["critic_loss"].append(critic_loss.item())

    def train(self, episodes=500, max_steps=10):
        max_steps_per_episode = getattr(self.env, 'max_steps', max_steps)

        t = trange(episodes, desc="Training", leave=True)
        total_reward_across_all_episodes = 0 

        for episode in t:
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards_raw  = []

            current_episode_reward_sum = 0
            current_episode_actions_sum = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards_raw.append(reward)

                state = next_state
                current_episode_reward_sum += reward
                if isinstance(action, (int, float)): # Ensure action is summable (e.g. not a list/array)
                    current_episode_actions_sum += action

                if done:
                    break

            # Compute discounted returns (G_t) and update networks
            returns_G_t = self.compute_returns(episode_rewards_raw)
            self.update_networks(episode_states, episode_actions, returns_G_t)

            # Log metrics
            total_reward_across_all_episodes += current_episode_reward_sum
            self.metrics['episode_rewards'].append(current_episode_reward_sum)
            self.metrics['actions_sum_in_episode'].append(current_episode_actions_sum)

            avg_reward_last_10 = np.mean(self.metrics['episode_rewards'][-10:]) if len(self.metrics['episode_rewards']) > 0 else 0.0
            t.set_description(
                f"Ep {episode+1}: Reward: {current_episode_reward_sum:.2f}, AvgR_10ep: {avg_reward_last_10:.2f}, TotalR: {total_reward_across_all_episodes:.2f}"
            )
            t.update()

    def plot_training(self, env_name, nb_ep):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # Changed to 2x2 layout
        fig.suptitle(f"Actor-Critic Training Metrics for {env_name} ({nb_ep} episodes)\nMean Episode Reward: {np.mean(self.metrics['episode_rewards']):.2f}", fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        ax1.plot(self.metrics["episode_rewards"], label="Per-Episode Reward", color="#4C72B0", linewidth=2)
        # Optional: Add a moving average for episode rewards
        if len(self.metrics["episode_rewards"]) >= 10:
            moving_avg = np.convolve(self.metrics["episode_rewards"], np.ones(10)/10, mode='valid')
            ax1.plot(np.arange(9, len(self.metrics["episode_rewards"])), moving_avg, label="10-ep Mov Avg Reward", color="#F57F17", linestyle="--")
        ax1.set_title("Episode Rewards", fontsize=14)
        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Reward", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.5)

        ax2 = axes[0, 1]
        ax2.plot(self.metrics["actor_loss"], label="Actor Loss", color="#8172B2", linewidth=2)
        ax2.set_title("Actor Loss", fontsize=14)
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.5)

        ax3 = axes[1, 0]
        ax3.plot(self.metrics["critic_loss"], label="Critic Loss", color="#55A868", linewidth=2)
        ax3.set_title("Critic Loss", fontsize=14)
        ax3.set_xlabel("Episode", fontsize=12)
        ax3.set_ylabel("Loss", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.5)
        
        ax4 = axes[1, 1]
        ax4.plot(self.metrics["actions_sum_in_episode"], label="Sum of Action Values per Episode", color="#C44E52", linewidth=2)
        ax4.set_title("Sum of Action Values", fontsize=14)
        ax4.set_xlabel("Episode", fontsize=12)
        ax4.set_ylabel("Sum of Actions", fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.5)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
        
        # Ensure results directory exists
        plot_dir = "./results" # Changed from ./results to ./plots
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"actor_critic_{env_name}_{nb_ep}_ep_metrics_plot.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show() # Comment out if running in a non-interactive environment
        print(f"Metric plot saved to {save_path}")


    def save_model(self, path):
        torch.save({
            'network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'hyperparameters': {
                'lr': self.lr_actor,
                'hidden_layers': self.hidden_layers_actor
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

    agent = ActorCriticAgent(env, gamma=0.99, lr_actor=0.0005, lr_critic=0.001, actor_hidden_layers=(64, 64), critic_hidden_layers=(64,64))    
    max_steps_for_env = getattr(env, 'max_steps', 200) 
    
    agent.train(episodes=nb_episodes, max_steps=max_steps_for_env) # Pass max_steps to train
    
    model_save_path = f"./models/actor_critic_{env_name}_{nb_episodes}_ep.pth"
    agent.save_model(model_save_path)
    
    agent.plot_training(env_name=env_name, nb_ep=nb_episodes)
