import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn as nn
from joblib import load
from algorithms.environments.inverted_pendulus import PendulumEnvironment
import sys

# Define the Q-Network for PyTorch models
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(5, 5)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class NFQAgentWrapperPendulum:
    def __init__(self, model_path, env):
        """
        Wrapper for NFQ models (.pkl or .pth) to interface with the Pendulum environment.
        """
        self.actions = env.actions
        self.env = env

        if model_path.endswith(".pkl"):
            # Load scikit-learn model
            self.model_type = "pkl"
            self.model = load(model_path)
        elif model_path.endswith(".pth"):
            # Reconstruct PyTorch model and load the state_dict
            self.model_type = "pth"
            self.model = QNetwork(state_dim=2, action_dim=len(self.actions), hidden_layers=(5, 5))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            raise ValueError("Unsupported model format. Use '.pkl' or '.pth'.")


    def greedy_action(self, state):
        """
        Choose the greedy action based on the model's Q-values.
        """
        q_values = []
        for action in self.actions:
            sa = np.hstack((state, [action]))
            if self.model_type == "pkl":
                q_values.append(self.model.predict(sa.reshape(1, -1))[0])
            elif self.model_type == "pth":
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values.append(self.model(state_tensor).squeeze().numpy()[self.actions.index(action)])
        return self.actions[np.argmax(q_values)]



class PendulumVisualizationWithMetrics:
    def __init__(self, agent, env, simulation_time=500):
        """
        Visualization for the inverted pendulum with live plots of rewards, actions, velocity, and positions.
        """
        self.agent = agent
        self.env = env
        self.simulation_time = simulation_time
        self.trajectory = {
            "angles": [],
            "angular_velocities": [],
            "actions": [],
            "rewards": []
        }

    def run_simulation(self):
        """
        Run the simulation and collect data for visualization.
        """
        state = self.env.reset()
        cumulative_reward = 0
        for _ in range(self.simulation_time):
            action = self.agent.greedy_action(state)
            next_state, reward, done = self.env.step(action)

            self.trajectory["angles"].append(state[0])
            self.trajectory["angular_velocities"].append(state[1])
            self.trajectory["actions"].append(action)
            cumulative_reward += reward
            self.trajectory["rewards"].append(cumulative_reward)

            state = next_state
            if done:
                break

    def visualize(self):
        """
        Visualize the pendulum and live metrics.
        """
        self.run_simulation()

        angles = self.trajectory["angles"]
        angular_velocities = self.trajectory["angular_velocities"]
        actions = self.trajectory["actions"]
        rewards = self.trajectory["rewards"]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        ax_circle = axs[0, 0]
        ax_rewards = axs[0, 1]
        ax_actions = axs[0, 2]
        ax_positions = axs[1, 0]
        ax_velocity = axs[1, 1]

        # Prepare the circular plot
        ax_circle.set_xlim(-1.2, 1.2)
        ax_circle.set_ylim(-1.2, 1.2)
        ax_circle.set_aspect('equal')
        circle = plt.Circle((0, 0), 1.0, color='lightgray', fill=False, linestyle='--')
        ax_circle.add_patch(circle)
        pendulum_line, = ax_circle.plot([], [], 'o-', lw=3, color='red', markersize=10)
        ax_circle.set_title("Pendulum Visualization")

        # Rewards plot
        ax_rewards.set_title("Cumulative Rewards")
        ax_rewards.set_xlim(0, len(rewards))
        ax_rewards.set_ylim(min(rewards) - 10, max(rewards) + 10)
        rewards_line, = ax_rewards.plot([], [], color='blue', label='Rewards')
        ax_rewards.legend()

        # Actions plot
        ax_actions.set_title("Actions Taken")
        ax_actions.set_xlim(0, len(actions))
        ax_actions.set_ylim(min(self.agent.actions) - 1, max(self.agent.actions) + 1)
        actions_line, = ax_actions.plot([], [], color='green', label='Actions')
        ax_actions.legend()

        # Positions plot
        ax_positions.set_title("Pendulum Angles (Positions)")
        ax_positions.set_xlim(0, len(angles))
        ax_positions.set_ylim(-np.pi - 0.5, np.pi + 0.5)
        positions_line, = ax_positions.plot([], [], color='orange', label='Angles')
        ax_positions.legend()

        # Velocity plot
        ax_velocity.set_title("Angular Velocity")
        ax_velocity.set_xlim(0, len(angular_velocities))
        ax_velocity.set_ylim(min(angular_velocities) - 1, max(angular_velocities) + 1)
        velocity_line, = ax_velocity.plot([], [], color='purple', label='Angular Velocity')
        ax_velocity.legend()

        # Function to compute pendulum position
        def get_pendulum_position(angle):
            x = np.sin(angle)
            y = np.cos(angle)
            return x, y

        # Animation initialization
        def init():
            pendulum_line.set_data([], [])
            rewards_line.set_data([], [])
            actions_line.set_data([], [])
            positions_line.set_data([], [])
            velocity_line.set_data([], [])
            return pendulum_line, rewards_line, actions_line, positions_line, velocity_line

        # Animation update function
        def update(frame):
            if frame >= len(angles):
                return pendulum_line, rewards_line, actions_line, positions_line, velocity_line

            # Update pendulum
            angle = angles[frame]
            x, y = get_pendulum_position(angle)
            pendulum_line.set_data([0, x], [0, y])

            # Update metrics
            rewards_line.set_data(range(frame + 1), rewards[:frame + 1])
            actions_line.set_data(range(frame + 1), actions[:frame + 1])
            positions_line.set_data(range(frame + 1), angles[:frame + 1])
            velocity_line.set_data(range(frame + 1), angular_velocities[:frame + 1])

            return pendulum_line, rewards_line, actions_line, positions_line, velocity_line

        ani = animation.FuncAnimation(fig, update, frames=len(angles), init_func=init,
                                      blit=True, interval=30)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    model_path = f"./models/{sys.argv[1]}"
    env = PendulumEnvironment()
    agent = NFQAgentWrapperPendulum(model_path, env)
    print("Visualizing Pendulum with Metrics...")
    viz = PendulumVisualizationWithMetrics(agent, env, simulation_time=500)
    viz.visualize()
