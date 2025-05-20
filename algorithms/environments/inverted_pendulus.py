import numpy as np
import math

# It's good practice to have a single, configurable prng
# If you run this script multiple times, this global prng will continue its sequence
# unless re-initialized. For full reproducibility in experiments, pass a seed to the env.
# global_prng = np.random.default_rng(1234567)

class PendulumEnvironment:
    def __init__(self, seed=None): # Allow passing a seed for prng
        self.mass = 1
        self.length = 1
        self.g_force = 9.81
        self.friction = 0.01
        self.delta_t = 0.01
        self.max_steps = 500
        self.actions = [-5, 0, 5]  
        self.state_dim = 2
        
        if seed is not None:
            self.prng = np.random.default_rng(seed)
        else:
            self.prng = np.random.default_rng()

        self.angular_position = 0.0 # Initialize attributes before reset is called
        self.angular_velocity = 0.0
        self.steps = 0
        self.angular_history = []
        self.reset()


    def reset(self):
        self.angular_position = self.prng.uniform(-np.pi, np.pi)
        self.angular_velocity = self.prng.uniform(-9, 9)
        self.steps = 0
        self.angular_history = []  # Reset history on new episode
        return np.array([self.angular_position, self.angular_velocity])

    def compute_angle_deviation(self):
        # This computes the mean absolute angle over the *current* episode's history
        return np.mean(np.abs(self.angular_history)) if self.angular_history else 0.0

    def is_successful(self, state_tuple=None):
        # Checks if the current or a given state is "successful" (upright)
        angle = state_tuple[0] if state_tuple is not None else self.angular_position
        return abs(angle) < 0.05

    def step(self, action):
        # Ensure action is one of the allowed values.
        if action not in self.actions:
            print(f"Warning: Action {action} not in allowed actions {self.actions}. Behavior undefined.")
            # Decide how to handle: error, clamp, or assume agent is correct.
            # For now, we assume the agent provides a valid action from self.actions.

        self.steps += 1

        # Dynamics calculation
        # a = angular acceleration
        torque = action # Action is the direct torque
        angular_acceleration = (1 / (self.mass * self.length**2)) * \
            (-self.friction * self.angular_velocity +
             (self.mass * self.g_force * self.length) * np.sin(self.angular_position) +
             torque)

        self.angular_velocity = self.angular_velocity + angular_acceleration * self.delta_t
        # Optional: Clip angular velocity if it can grow too large, e.g.:
        # max_angular_velocity = 15 # Example limit
        # self.angular_velocity = np.clip(self.angular_velocity, -max_angular_velocity, max_angular_velocity)

        self.angular_position = self.angular_position + self.angular_velocity * self.delta_t

        # Normalize angular_position to be within [-pi, pi]
        # (x + pi) % (2*pi) - pi is a robust way to wrap to [-pi, pi]
        self.angular_position = (self.angular_position + np.pi) % (2 * np.pi) - np.pi

        reward = np.cos(self.angular_position)
        self.angular_history.append(self.angular_position)

        # Termination condition: episode ends if max_steps is reached
        done = False
        if self.steps >= self.max_steps:
            done = True

        return np.array([self.angular_position, self.angular_velocity]), reward, done