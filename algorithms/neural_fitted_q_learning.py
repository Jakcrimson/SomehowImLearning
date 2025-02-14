import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import joblib
from environments.car_hill import CarHillEnvironment
from environments.inverted_pendulus import PendulumEnvironment
import sys
import time 
import seaborn as sns

"""THIS IS AN OFFLINE ALGORITHM

"""

class NFQAgent:
    def __init__(self, env, gamma=0.99, hidden_layers=(5,5), max_iter=1_000, lr=0.001):
        self.env = env
        self.actions = self.env.actions
        self.gamma = gamma
        self.prng = np.random.default_rng(123456)
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, 
                                  solver='adam', 
                                  learning_rate_init=lr,
                                  max_iter=max_iter, 
                                  warm_start=True)
        dummy_X = np.zeros((1, 2+1))
        dummy_y = np.zeros((1,))
        self.model.fit(dummy_X, dummy_y)

        # Logs
        self.metrics = {
            "cumulative_reward": [],
        }
        
    def Q(self, state, action):
        """
        Compute Q-value for a given state-action pair.

        Args:
            state (numpy.ndarray): Current state
            action (int/float): Action to evaluate

        Returns:
            float: Predicted Q-value for the state-action pair
        """
        sa = np.hstack((state, [action]))
        return self.model.predict(sa.reshape(1,-1))[0]

    def Qmax(self, next_state):
        """
        Compute maximum Q-value over all actions for a given state.

        Args:
            next_state (numpy.ndarray): State to evaluate

        Returns:
            float: Maximum Q-value for the state
        """
        q_values = [self.Q(next_state, a) for a in self.actions]
        return np.max(q_values)

    def greedy_action(self, state):
        """
        Select the action with highest Q-value for given state.

        Args:
            state (numpy.ndarray): Current state

        Returns:
            Action that maximizes Q-value in current state
        """
        q_values = [self.Q(state, a) for a in self.actions]
        return self.actions[np.argmax(q_values)]

    def collect_data(self, policy, num_episodes=10, max_steps=200):
        """
        Collect experience data using specified policy.

        Args:
            policy (str): Either 'random' or 'greedy'
            num_episodes (int, optional): Number of episodes to collect. Defaults to 10
            max_steps (int, optional): Maximum steps per episode. Defaults to 200

        Returns:
            list: Collection of (state, action, reward, next_state) tuples

        Raises:
            ValueError: If policy is neither 'random' nor 'greedy'
        """
        data = []
        for _ in range(num_episodes):
            state = self.env.reset()
            for _ in range(max_steps):
                if policy == 'random':
                    action = self.prng.choice(self.actions)
                elif policy == 'greedy':
                    action = self.greedy_action(state)
                else:
                    raise ValueError("Unknown policy")

                next_state, reward, done = self.env.step(action)
                data.append((state, action, reward, next_state))
                state = next_state
                if isinstance(self.env, PendulumEnvironment):
                    done = self.gamma**self.env.steps > 1e-4
                if done:
                    break

        return data

    def update_Q(self, data):
        """
        Update Q-function approximation using collected experience data.

        Args:
            data (list): List of (state, action, reward, next_state) tuples
        """
        X = []
        y = []

        for (s,a,r,s_next) in data:
            Q_next = self.Qmax(s_next)
            target = r + self.gamma * Q_next
            X.append(np.hstack((s, [a])))
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)


    def train(self, iterations=20, episodes_per_iter=10, eval_episodes=5):
        """
        Execute the main NFQ training loop.

        The training process alternates between collecting data using the current policy
        and updating the Q-function approximation.

        Args:
            iterations (int, optional): Number of training iterations. Defaults to 20
            episodes_per_iter (int, optional): Episodes to collect per iteration. Defaults to 10
            eval_episodes (int, optional): Number of episodes for evaluation. Defaults to 5
            visualize_interval (int, optional): Interval for visualization. Defaults to 5
        """
        data = self.collect_data(policy='random', num_episodes=episodes_per_iter)
        
        for i in range(iterations):
            self.update_Q(data)
            returns = self.evaluate_policy(num_episodes=eval_episodes)
            self.metrics["cumulative_reward"].append(np.mean(returns))
            print(f"Iteration {i+1}/{iterations}, Return (mean over {eval_episodes} episodes): {np.mean(returns):.2f}")
            new_data = self.collect_data(policy='greedy', num_episodes=episodes_per_iter)
            data = data + new_data

    def evaluate_policy(self, num_episodes=5, max_steps=200):
        """
        Evaluate the current policy over multiple episodes.

        Args:
            num_episodes (int, optional): Number of evaluation episodes. Defaults to 5
            max_steps (int, optional): Maximum steps per episode. Defaults to 200

        Returns:
            list: Returns (total rewards) for each evaluation episode
        """
        returns = []
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action = self.greedy_action(state)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                state = next_state
                if done:
                    break
            returns.append(total_reward)

        return returns    
    
    def plot_training(self, env_name, nb_ep):
        sns.set_theme(style="whitegrid")  # Set a professional theme
        fig, (ax1) = plt.subplots(1, 1, figsize=(14, 10))  # Create 2x2 subplots
        fig.suptitle(f"Training Metrics for {env_name} | Mean Cumulative Reward {np.mean(self.metrics['cumulative_reward'])}", fontsize=16, fontweight='bold')

        # Plot Cumulative Reward
        ax1.plot(self.metrics["cumulative_reward"], label="Cumulative Reward", color="#4C72B0", linewidth=2)
        ax1.set_title("Cumulative Reward", fontsize=14)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Reward", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        save_path = f"./results/nfq_{env_name}_{nb_ep}_ep_metrics_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as high-res image
        plt.show()
        print(f"Metric plot saved to {save_path}")

    def save_model(self, model_path):
        """
        Save the trained model to disk.

        Args:
            model_path (str): Path where model should be saved
        """
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model file
        """
        self.model = joblib.load(model_path)

if __name__ == "__main__":  
    """Command line usage:
        python script.py [environment] [model_name]

        where environment is either "car" or "pendulum" and model_name is the name of the model you want to train and save.

        The script will:
        1. Create the specified environment
        2. Initialize and train an NFQ agent
        3. Evaluate the final policy
        4. Save the trained model
    """
    env_name = sys.argv[1]
    nb_episodes = int(sys.argv[2])

    if env_name == "car":
        env = CarHillEnvironment()
    elif env_name == "pendulum":
        env = PendulumEnvironment()
    else:
        raise ValueError("Unknown environment")

    agent = NFQAgent(env, gamma=0.99, hidden_layers=(5,5), max_iter=200, lr=0.001) 
    start_time = time.time()
    agent.train(iterations=nb_episodes, episodes_per_iter=env.max_steps, eval_episodes=10)
    end_time = time.time()

    returns = agent.evaluate_policy(num_episodes=10)
    print("Time of inference :", (end_time - start_time))

    agent.save_model(f"./models/nfq_{env_name}_model_{nb_episodes}_ep.pkl")
    agent.plot_training(env_name=env_name, nb_ep=nb_episodes)
