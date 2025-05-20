import numpy as np
import math

class PendulumEnvironment:
    def __init__(self, seed=None): 
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
        self.angular_position = 0.0 
        self.angular_velocity = 0.0
        self.steps = 0
        self.angular_history = []
        self.reset()


    def reset(self):
        self.angular_position = self.prng.uniform(-np.pi, np.pi)
        self.angular_velocity = self.prng.uniform(-9, 9)
        self.steps = 0
        self.angular_history = [] 
        return np.array([self.angular_position, self.angular_velocity])

    def compute_angle_deviation(self):
        return np.mean(np.abs(self.angular_history)) if self.angular_history else 0.0

    def is_successful(self, state_tuple=None):
        angle = state_tuple[0] if state_tuple is not None else self.angular_position
        return abs(angle) < 0.05

    def step(self, action):
        if action not in self.actions:
            print(f"Warning: Action {action} not in allowed actions {self.actions}. Behavior undefined.")
        self.steps += 1
        torque = action
        angular_acceleration = (1 / (self.mass * self.length**2)) * \
            (-self.friction * self.angular_velocity +
             (self.mass * self.g_force * self.length) * np.sin(self.angular_position) +
             torque)
        self.angular_velocity = self.angular_velocity + angular_acceleration * self.delta_t
        self.angular_position = self.angular_position + self.angular_velocity * self.delta_t
        self.angular_position = (self.angular_position + np.pi) % (2 * np.pi) - np.pi
        reward = np.cos(self.angular_position)
        self.angular_history.append(self.angular_position)
        done = False
        if self.steps >= self.max_steps:
            done = True
        return np.array([self.angular_position, self.angular_velocity]), reward, done