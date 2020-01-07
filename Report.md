#Report

To solve the presented problem a deep Q-network (DQN) has been chosen. The approach combines reinforcement learning with a class of artificial neural network. The network consist of several layers of nodes which makes it possible for artificial neural network to learn progressively more abstract representations of the raw sensory data.
The role of the agent is to learn through interactions wit the environment. The goal is to select actions that maximizes cumulative future reward. Using nonlinear function approximator to represesent the action-value function makes learning process unstable or even diverging. To address both of those issues two key ideas have been introduced. First, mechanism termed as experience replay that randomizes over data. The process removes correlations in the sequence of observations and smooths changes in the data distribution. Second, an itertive update has been used to adjust action-values towards target values. The update is periodicall, therefore the correlactions with the target are reduced.

[source: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf]

###Model architecture

The input to the neural network consist of 37 values. There are 64 nodes for each first and second hidden layer. The final output layer provides 4 values for each valid action. All nodes are followed by a rectified linear unit (ReLU). The network consists only of fully connected layers.

###Hyperparameters

dqn_agent.py file:  
BUFFER_SIZE = 1e5  - size of the replay buffer  
BATCH_SIZE = 64  - number of samples used during learning step  
TAU = 1e-3  - local parameters to target update factor  
LR - 5e-4 - learning rate  
UPDATE_EVERY = 4 - how often to update the network  

Navigation.ipnyb file:  
max_t = 1000  - maksimum number of agent's steps before an episode finishes  
eps_start = 1.0  - initial value of the epsilon paramater  
eps_end = 0.01  - minimal value of the epsilon parameter  
eps_deay = 0.995  - decay rate of the epsilon paramter  

###Plot of rewards 

The environment has been solved (mean score over 100 episodes was higher than 13) in less than 500 episodes.  
![mean score](final_plot_no_PER.png)  

Weights and model are stored in file 'model_weigths.pth'  

###Future improvement

Prioritized Experience Replay (PER):  
Prioritized experience replay relies on the idea that the agent can learn more from some transisions than the others. The more important ones shall be sampled more frequently than the others.  

[source: https://arxiv.org/pdf/1511.05952.pdf]

Double Deep Q-Network (DDQN):  
To avoid overestimation of action values Double DQN has been proposed.  

[source: https://arxiv.org/pdf/1509.06461.pdf]

Dueling DQN:  
With a dueling architecture, the values of each state can be accessed, without learning the effect of each action.

[source: https://arxiv.org/pdf/1511.06581.pdf]
