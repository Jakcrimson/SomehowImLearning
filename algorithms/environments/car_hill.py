import numpy as np

class CarHillEnvironment:
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.goal_position = 0.5
        self.max_steps = 50 
        self.prng = np.random.default_rng(123456)
        self.actions = [-0.1, 0, 0.1]
        self.reset()
        self.state_dim = 2

    def reset(self):
        """
        Reset the environment to initial state.

        Initializes the car's position uniformly between -0.6 and -0.4,
        and velocity uniformly between -0.07 and 0.07.

        Returns:
            numpy.ndarray: Initial state as [position, velocity]
        """
        self.position = self.prng.uniform(-0.6, -0.4) 
        self.velocity = self.prng.uniform(-0.07, 0.07)
        self.steps = 0
        return np.array([self.position, self.velocity])
    
    def is_successful(self, state):
        position, _ = state
        return position >= self.goal_position
    
    def compute_energy_efficiency(self, actions):
        return sum(abs(action) for action in actions) / len(actions)  # Penalize large actions

    def step(self, action):
        """
        Simulate one step in the environment.

        Updates the car's position and velocity based on the chosen action
        and environmental physics including gravity and hill shape effects.

        Args:
            action (int): Chosen action from {-1, 0, 1}
                -1: Push car left
                0: No push
                1: Push car right

        Returns:
            tuple:
                - numpy.ndarray: Next state as [position, velocity]
                - float: Reward (1000000 if goal reached, -1 otherwise)
                - bool: Whether episode has terminated

        Notes:
            - Velocity is affected by action force (0.001 * action) and
            hill shape (-0.0025 * cos(3 * position))
            - Position and velocity are clipped to their allowed ranges
            - Episode terminates if max_steps is reached or goal is achieved
        """
        self.steps += 1
        self.velocity += 0.001 * action - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)

        done = self.steps >= self.max_steps
        if self.position >= self.goal_position:
            reward = 1000
        else:
            reward = -1
        return np.array([self.position, self.velocity]), reward, done
