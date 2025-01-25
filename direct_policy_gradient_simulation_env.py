import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import sys
from torch import nn
from algorithms.environments.inverted_pendulus import PendulumEnvironment
from algorithms.environments.car_hill import CarHillEnvironment

# Define the REINFORCE Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(64,64)):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class REINFORCEAgentWrapper:
    def __init__(self, model_path, state_dim, action_dim, hidden_layers):
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_layers)
        
        # Load the saved model
        checkpoint = torch.load(f"./models/{model_path}")
        self.policy_network.load_state_dict(checkpoint['network_state_dict'])
        self.policy_network.eval()  # Set model to evaluation mode
        
        self.actions = list(range(action_dim))  # Action space
    

    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor).squeeze().detach().numpy()
        if deterministic:
            action_index = np.argmax(action_probs)
        else:
            action_index = np.random.choice(self.actions, p=action_probs)
        return self.actions[action_index]


class Visualization:
    def __init__(self, agent, env, simulation_time=200):
        self.agent = agent
        self.env = env
        self.simulation_time = simulation_time
        self.trajectory = {
            "positions": [],
            "velocities": [],
            "actions": [],
            "rewards": []
        }

    def run_simulation(self):
        state = self.env.reset()
        cumulative_reward = 0
        for _ in range(self.simulation_time):
            action = self.agent.select_action(state, deterministic=True)
            next_state, reward, done = self.env.step(action)

            self.trajectory["positions"].append(state[0])
            self.trajectory["velocities"].append(state[1])
            self.trajectory["actions"].append(action)
            cumulative_reward += reward
            self.trajectory["rewards"].append(cumulative_reward)

            state = next_state
            if done:
                break

    def visualize(self):
        self.run_simulation()

        positions = self.trajectory["positions"]
        velocities = self.trajectory["velocities"]
        actions = self.trajectory["actions"]
        rewards = self.trajectory["rewards"]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        ax_rewards = axs[0, 0]
        ax_positions = axs[0, 1]
        ax_velocity = axs[1, 0]
        ax_actions = axs[1, 1]

        # Rewards plot
        ax_rewards.set_title("Cumulative Rewards")
        ax_rewards.plot(range(len(rewards)), rewards, color='blue', label='Rewards')
        ax_rewards.set_xlabel("Time Step")
        ax_rewards.set_ylabel("Cumulative Reward")
        ax_rewards.legend()

        # Positions plot
        ax_positions.set_title("Positions")
        ax_positions.plot(range(len(positions)), positions, color='orange', label='Positions')
        ax_positions.set_xlabel("Time Step")
        ax_positions.set_ylabel("Position")
        ax_positions.legend()

        # Velocity plot
        ax_velocity.set_title("Velocity")
        ax_velocity.plot(range(len(velocities)), velocities, color='purple', label='Velocity')
        ax_velocity.set_xlabel("Time Step")
        ax_velocity.set_ylabel("Velocity")
        ax_velocity.legend()

        # Actions plot
        ax_actions.set_title("Actions Taken")
        ax_actions.plot(range(len(actions)), actions, color='green', label='Actions')
        ax_actions.set_xlabel("Time Step")
        ax_actions.set_ylabel("Action")
        ax_actions.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env_name = sys.argv[1]
    model_path = sys.argv[2]
    simulation_time = int(sys.argv[3])

    if env_name == "car":
        env = CarHillEnvironment()
        state_dim = env.state_dim
        action_dim = len(env.actions)
        agent = REINFORCEAgentWrapper(model_path, state_dim, action_dim, hidden_layers=(64, 64))
        viz = Visualization(agent, env, simulation_time=simulation_time)
        print("Visualizing Car Environment with REINFORCE...")
        viz.visualize()
    elif env_name == "pendulum":
        env = PendulumEnvironment()
        state_dim = env.state_dim
        action_dim = len(env.actions)
        agent = REINFORCEAgentWrapper(model_path, state_dim, action_dim, hidden_layers=(64, 64))
        viz = Visualization(agent, env, simulation_time=simulation_time)
        print("Visualizing Pendulum with REINFORCE...")
        viz.visualize()
    else:
        raise ValueError("Unknown environment")
