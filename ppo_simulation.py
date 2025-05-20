import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import sys
from torch import nn
from algorithms.environments.inverted_pendulus import PendulumEnvironment
from algorithms.environments.car_hill import CarHillEnvironment
from q_learning_simulation_env import PendulumVisualizationWithMetrics, CarVisualization

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
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

class REINFORCEAgentWrapper:
    def __init__(self, model_path, state_dim, action_dim, hidden_layers, env_actions_map=None):
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_layers)
        
        full_model_path = f"./models/{model_path}"
        print(f"Loading REINFORCE model from: {full_model_path}")
        
        checkpoint = torch.load(full_model_path)
        self.policy_network.load_state_dict(checkpoint['network_state_dict'])
        self.policy_network.eval()
        
        if env_actions_map:
            self.actions = env_actions_map
        else:
            self.actions = list(range(action_dim))


    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor).squeeze().detach().numpy()
        
        action_probs = action_probs / np.sum(action_probs)

        if deterministic:
            action_index = np.argmax(action_probs)
        else:
            action_index = np.random.choice(len(self.actions), p=action_probs)
        return self.actions[action_index] 
    
    def greedy_action(self, state): 
        return self.select_action(state, deterministic=True)

# --- Actor-Critic Components ---
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(ActorNetwork, self).__init__()
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

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super(CriticNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))  
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

class ActorCriticAgentWrapper:
    def __init__(self, model_path, state_dim, action_dim, hidden_layers_actor, hidden_layers_critic, env_actions_map=None):
        self.actor = ActorNetwork(state_dim, action_dim, hidden_layers_actor)
        self.critic = CriticNetwork(state_dim, hidden_layers_critic)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        if env_actions_map:
            self.actions = env_actions_map
        else:
            self.actions = list(range(action_dim))
        
        self.load_model(model_path)


    def load_model(self, model_filename):
        full_model_path = f"./models/{model_filename}"
        print(f"Loading Actor-Critic model from: {full_model_path}")
        
        checkpoint = torch.load(full_model_path, map_location=self.device)
        
        if 'actor_state_dict' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
        else:
            print("Warning: 'actor_state_dict' not found in checkpoint. Trying to load 'network_state_dict' or root for actor.")
            if 'network_state_dict' in checkpoint:
                 self.actor.load_state_dict(checkpoint['network_state_dict'])
            else:
                 self.actor.load_state_dict(checkpoint) 

        self.actor.eval()

        if 'critic_state_dict' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic.eval()
        else:
            print("Note: 'critic_state_dict' not found in checkpoint. Critic will remain initialized randomly (if not used for inference, this is okay).")


    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad(): # Important for inference
            action_probs = self.actor(state_tensor)
        
        action_probs_np = action_probs.cpu().squeeze().numpy()
        action_probs_np = action_probs_np / np.sum(action_probs_np)

        if deterministic:
            action_index = np.argmax(action_probs_np)
        else:
            action_index = np.random.choice(len(self.actions), p=action_probs_np)
        
        return self.actions[action_index] 

    def greedy_action(self, state): # Consistent with select_action(deterministic=True)
        return self.select_action(state, deterministic=True)

# --- Main Script Updated ---
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python your_script_name.py <env_name> <model_file_name> <ppo_method> <simulation_time>")
        print("Example: python your_script_name.py pendulum reinforce_pendulum_model.pth reinforce 200")
        print("Example: python your_script_name.py car ac_car_model.pth ac 300")
        sys.exit(1)

    env_name = sys.argv[1]
    model_filename = sys.argv[2] 
    agent_method = sys.argv[3] 
    simulation_time = int(sys.argv[4])
    
    agent = None
    viz = None

    hidden_layers_config = (64, 64)
    hidden_layers_actor_config = (64, 64)
    hidden_layers_critic_config = (64, 64)

    if env_name == "pendulum":
        env = PendulumEnvironment()
        state_dim = env.state_dim
        action_dim = len(env.actions) 
        env_actions_map = env.actions 

        if agent_method == "reinforce":
            agent = REINFORCEAgentWrapper(model_filename, state_dim, action_dim, 
                                          hidden_layers_config, env_actions_map=env_actions_map)
            print("Visualizing Pendulum Environment with REINFORCE agent...")
        elif agent_method == "ac":
            agent = ActorCriticAgentWrapper(model_filename, state_dim, action_dim, 
                                            hidden_layers_actor_config, hidden_layers_critic_config, 
                                            env_actions_map=env_actions_map)
            print("Visualizing Pendulum Environment with Actor-Critic agent...")
        else:
            raise ValueError(f"Unknown agent_method: {agent_method}. Choose 'reinforce' or 'ac'.")
        
        viz = PendulumVisualizationWithMetrics(agent, env, simulation_time=simulation_time)
        viz.visualize()

    elif env_name == "car":
        env = CarHillEnvironment()
        state_dim = env.state_dim
        action_dim = len(env.actions)
        env_actions_map = env.actions

        if agent_method == "reinforce":
            agent = REINFORCEAgentWrapper(model_filename, state_dim, action_dim, 
                                          hidden_layers_config, env_actions_map=env_actions_map)
            print("Visualizing Car Environment with REINFORCE agent...")
        elif agent_method == "ac":
            agent = ActorCriticAgentWrapper(model_filename, state_dim, action_dim, 
                                            hidden_layers_actor_config, hidden_layers_critic_config,
                                            env_actions_map=env_actions_map)
            print("Visualizing Car Environment with Actor-Critic agent...")
        else:
            raise ValueError(f"Unknown agent_method: {agent_method}. Choose 'reinforce' or 'ac'.")

        viz = CarVisualization(agent, env, simulation_time=simulation_time)
        viz.visualize_car_environment()
    else:
        raise ValueError(f"Unknown environment: {env_name}. Choose 'pendulum' or 'car'.")