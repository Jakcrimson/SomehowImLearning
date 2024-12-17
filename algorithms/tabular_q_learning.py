import numpy as np
from tqdm import tqdm
from environments.inverted_pendulus import PendulumEnvironment
from environments.car_hill import CarHillEnvironment

class QLearningAgent:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.position_bins = env.position_bins
        self.speed_bins = env.speed_bins
        self.actions = env.actions
        self.q_table = env.q_table
        self.visits = env.visits 

    def discretize_state(self, state):
        position, velocity = state
        pos_idx = np.digitize(position, self.position_bins) - 1
        vel_idx = np.digitize(velocity, self.speed_bins) - 1
        return pos_idx, vel_idx

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions) 
        else:
            pos_idx, vel_idx = self.discretize_state(state)
            return self.actions[np.argmax(self.q_table[pos_idx, vel_idx])]

    def update_alpha(self, state_idx, action_idx):
        visits = self.visits[state_idx + (action_idx,)]
        return 1 / (1 + visits) 

    def train(self, num_episodes=2000):
        rewards = []
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Décision de l'action
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                if isinstance(self.env, PendulumEnvironment):
                    done = self.gamma**self.env.steps > 1e-4 or self.env.steps >= self.env.max_steps
                if isinstance(self.env, CarHillEnvironment):
                    done = self.env.steps >= self.env.max_steps

                # Mise à jour Q-learning
                pos_idx, vel_idx = self.discretize_state(state)
                next_pos_idx, next_vel_idx = self.discretize_state(next_state)
                action_idx = self.actions.index(action)
                
                # Mettre à jour le compteur de visites
                self.visits[pos_idx, vel_idx, action_idx] += 1
                
                # Calcul du taux d'apprentissage dynamique
                alpha = self.update_alpha((pos_idx, vel_idx), action_idx)
                
                # Calcul de la mise à jour de la table Q
                best_next_action = np.max(self.q_table[next_pos_idx, next_vel_idx])
                td_target = reward + self.gamma * best_next_action
                td_error = td_target - self.q_table[pos_idx, vel_idx, action_idx]
                self.q_table[pos_idx, vel_idx, action_idx] += 0.2 * td_error
                
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            
            # Décroissance de epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return rewards, self.q_table

    def set_qtable(self, q_table):
        self.q_table = q_table