# DRL-banana-agent

Udacity "Navigation" project solved with DQN
![agent performance visualisation](fin_banana.gif)

### Introduction  

For this project, an agent was trained to navigate (and collect bananas) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Dependencies (OS: Ubuntu 18.04)  

Install Anaconda - https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html  
Run install.sh file to add required components.

Clone the repository, in conda 'drlnd' environment start jupyter notebook by typing in terminal 'jupyter-notebook', switch kernel on 'drlnd'.


Download built environment - https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip 

### Training the agent
To train the agent use Navigation.ipynb file  
Type in terminal:  
	jupyter-notebook Navigation.ipnyb  