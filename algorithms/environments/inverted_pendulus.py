import numpy as np
prng = np.random.default_rng(1234567)
import math

class PendulumEnvironment:
    def __init__(self):
        self.mass = 1
        self.length = 1
        self.g_force = 9.81
        self.friction = 0.01
        self.delta_t = 0.01
        self.max_steps = 500000
        self.actions = [-5, 0, 5]
        self.reset()
        self.state_dim = 2
        self.angular_history = []
    
    def reset(self):
        """
        Reset the environment to initial state.

        Initializes the pendulums's position uniformly between -pi and pi,
        and velocity uniformly between -9 and 9.

        Returns:
            numpy.ndarray: Initial state as [angular_position, angular_velocity]
        """
        self.angular_position = prng.uniform(-np.pi, np.pi)
        self.angular_velocity = prng.uniform(-9, 9)
        self.steps = 0
        return np.array([self.angular_position, self.angular_velocity])

    def compute_angle_deviation(self):
        return np.mean(abs(self.angular_position))  # Track angle deviation over time

    def is_successful(self, state):
        angle, _ = state
        return abs(angle) < 0.05  # Consider upright if deviation is small

    def step(self, action):
        """
        Simulate one step in the environment.

        Updates the pendulum's angular_position and angular_velocity based on the chosen action
        and environmental physics including gravity, friction, lenght and mass.

        Args:
            action (int): Chosen action from {-5, 0, 5}
                -5: Rotate left
                0: No rotation
                5: Rotate right

        Returns:
            tuple:
                - numpy.ndarray: Next state as [position, velocity]
                - float: Reward (cos(angular_position), if balanced, then the reward is 1)
                - bool: if the episode is terminated

        """
        done = False
        self.steps+=1
        a = (1/self.mass*self.length**2) * (-self.friction*self.angular_velocity + (self.mass*self.g_force*self.length)*np.sin(self.angular_position) + action)
        self.angular_velocity  = self.angular_velocity + a * self.delta_t
        self.angular_position = self.angular_position + self.angular_velocity * self.delta_t
        if self.angular_position < -math.pi:
            self.angular_position = self.angular_position+2*math.pi
        elif self.angular_position > math.pi:
            self.angular_position = self.angular_position-2*math.pi
        reward = np.cos(self.angular_position)
        self.angular_history.append(self.angular_position)
        return np.array([self.angular_position, self.angular_velocity]), reward, done
