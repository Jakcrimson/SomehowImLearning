import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from algorithms.environments.car_hill import CarHillEnvironment
from joblib import load  # For loading .pkl models
import torch
import torch.nn as nn
import sys

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=(5,)):
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


class NFQAgentWrapper:
    def __init__(self, model_path, env):
        """
        Wrapper for the NFQ model to interface with the environment.
        Args:
            model_path: Path to the saved NFQ model (pkl or pth).
            env: CarHillEnvironment.
        """
        self.env = env
        self.actions = self.env.actions  # Discrete action space

        if model_path.endswith(".pkl"):
            # Load scikit-learn model
            self.model_type = "pkl"
            self.model = load(model_path)
        elif model_path.endswith(".pth"):
            # Reconstruct the model and load state_dict
            self.model_type = "pth"
            self.model = QNetwork(state_dim=2, action_dim=3, hidden_layers=(5,5))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set to evaluation mode
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
                    q_values.append(self.model(state_tensor).squeeze().numpy()[action])
        return self.actions[np.argmax(q_values)]


def visualize_car_environment(agent, env, simulation_time=200):
    """
    Visualize the car on the hill environment using the trained NFQAgentWrapper.

    Args:
        agent: Trained NFQAgentWrapper.
        env: CarHillEnvironment.
        simulation_time: Maximum simulation time in steps.
    """
    state = env.reset()
    trajectory = {"positions": [], "velocities": [], "rewards": []}

    for ep in range(simulation_time):
        action = agent.greedy_action(state)
        next_state, reward, done = env.step(action)
        trajectory["positions"].append(state[0])
        trajectory["velocities"].append(state[1])
        trajectory["rewards"].append(reward)
        state = next_state
        if reward > 0:
            print(f"[+] Success at step: {ep}")
            break


    positions = trajectory["positions"]
    velocities = trajectory["velocities"]
    rewards = trajectory["rewards"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_car = axs[0, 0]
    ax_position = axs[0, 1]
    ax_velocity = axs[1, 0]
    ax_reward = axs[1, 1]

    ax_car.set_xlim(env.min_position, env.max_position)
    ax_car.set_ylim(-1.5, 1.5)
    car_line, = ax_car.plot([], [], 'ro', markersize=10)
    hill_line, = ax_car.plot([], [], 'b-', lw=2)
    ax_car.set_title("Car on the Hill")

    ax_position.set_xlim(0, simulation_time)
    ax_position.set_ylim(env.min_position, env.max_position)
    position_line, = ax_position.plot([], [], label="Position")
    ax_position.legend()

    ax_velocity.set_xlim(0, simulation_time)
    ax_velocity.set_ylim(env.min_velocity, env.max_velocity)
    velocity_line, = ax_velocity.plot([], [], label="Velocity")
    ax_velocity.legend()

    ax_reward.set_xlim(0, simulation_time)
    ax_reward.set_ylim(-1, 10000)
    reward_line, = ax_reward.plot([], [], label="Reward")
    ax_reward.legend()

    def hill_function(x):
        return np.sin(3 * x)

    def init():
        car_line.set_data([], [])
        hill_x = np.linspace(env.min_position, env.max_position, 500)
        hill_y = hill_function(hill_x)
        hill_line.set_data(hill_x, hill_y)
        position_line.set_data([], [])
        velocity_line.set_data([], [])
        reward_line.set_data([], [])
        return car_line, position_line, velocity_line, reward_line

    def update(frame):
        if frame >= len(positions):
            return car_line, position_line, velocity_line, reward_line

        x = np.array([positions[frame]])
        y = hill_function(x)
        car_line.set_data(x, y)

        position_line.set_data(range(len(positions[:frame + 1])), positions[:frame + 1])
        velocity_line.set_data(range(len(velocities[:frame + 1])), velocities[:frame + 1])
        reward_line.set_data(range(len(rewards[:frame + 1])), rewards[:frame + 1])

        return car_line, position_line, velocity_line, reward_line

    ani = animation.FuncAnimation(
        fig, update, frames=simulation_time, init_func=init, blit=True, interval=1
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = f"./models/{sys.argv[1]}"
    
    env = CarHillEnvironment()
    agent = NFQAgentWrapper(model_path, env)
    print("Visualizing the car on the hill using the trained NFQ model...")
    visualize_car_environment(agent, env, simulation_time=500)
