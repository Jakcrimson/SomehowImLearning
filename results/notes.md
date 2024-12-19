# Notes about the results

---

## NFQ (offline)

### Car Environment
- _Setup 1_ : 
    - nb_episodes : 10
    - episodes_per_iter : 200
    - evaluation_episodes : 10
    - gamma : 0.99
    - lr : 0.001
    - hidden_layers : (5, 5)
    - **Training Time : 544.59 seconds (~9min)**
    - **Iterations to first success : 17**  
    - model : `nfq_car_model_1O_ep.pkl` 


### Pendulum Environment
- _Setup 1_ : 
    - nb_episodes : 150
    - episodes_per_iter : 5000
    - evaluation_episodes : 10
    - gamma : 0.99
    - lr : 0.001
    - hidden_layers : (5, 5)
    - **Training Time : 140.84 seconds (~2min30)**
    - model : `nfq_pendulum_model_15O_ep.pkl`

## DQN (online)

### Car Environment
- _Setup 1_ : 
    - nb_episodes : 500
    - iterations_per_ep : 200
    - target_network_update : 5
    - gamma : 0.99
    - lr : 0.001
    - hidden_layers : (5, 5)
    - epsilon_decay : 0.9995
    - **Training Time : 402.24 seconds (~7min)**
    - **Iterations to first success : 17**   
    - model : `dqn_car_500_ep.pth`

### Pendulum Environment
- _Setup 1_ : 
    - nb_episodes : 50000
    - iterations_per_ep : 5000
    - target_network_update : 5
    - gamma : 0.99
    - lr : 0.001
    - hidden_layers : (5, 5)
    - epsilon_decay : 0.9995
    - **Training Time : 305.56 seconds (~5min)**
    - model : `dqn_pendulum_50000_ep.pth`
---


# Pendulum Environment