[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

# Project 2: Continuous Control

![Trained Agent][image1]

### Overview
This is the 2nd project for the Udacity Deep Reinforcement Learning Nanodegree program. The goal of this project is to use deep neural network to solve a continuous control task. For this project, the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment is utilized.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, I choose to use the Unity environment which contains 20 identical agents, each with its own copy of the environment.  

This version of environment is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

#### Solving the Environment

- The task is episodic, and in order to solve the environment, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  
- Specifically, after each episode,the rewards that each agent received (without discounting) are added up to get a score for each agent.  This yields 20 (potentially different) scores.  Then the average of these 20 scores are calculated. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is solved at **38** episodes with the code attached, when the average (over 100 episodes) of those **scores** is above 30. 

### File Instruction
_**Continuous_Control.ipynb**_: This is the python jupyter notebook file which performs the followings:
1. Start the environment and examine the state and action space
2. Take random actions in the environement to become familar with the environment API
3. Train the agent using deep neural network to solve the environement

_**agent.py**_: The python code to implement the [deep deterministic policy agent (DDPG) algorithm](https://arxiv.org/abs/1509.02971).
- The DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that operate over continuous action spaces. 
- The **critic** network is to approximate the _Q_ fuction that takes in state action pair and returns a single scalar value as output,  whereas the **actor** network directly approximates the policy gradient that takes in the observations and output the continuous action space vectors. 
- The **critic's model weights** are learned by minimizing the error between the estimated **target _Q_** value _=reward+gamma*critic(state',action') where action'=actor(state')_ and **actual _Q_**_=critic(state,action)_ value, while the **actor's model weights** are learned by maximizing the **expected _Q_**_=critic(state, action")_ value with the estimated _action"=actor(state)_ input.

_**model.py**_: The python code to configure the neural network for both actor and critic.

_**checkpoint_actor|critic.pth**_: Both agents' actor and critic model weights are saved in the checkpoint file

### Requirements
1. Install Anaconda distribution of python3

2. Install PyTorch, Jupyter Notebook in the Python3 environment

3. Download the environment from one of the links below and place it in the same directory folder. Only select the environment that matches the operating system:
  
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

4. Open Jupyter Notebook and run the Tennis.ipynb file to train the agent. 

5. To watch the agents to play, load the model weights from those two checkpiont _.pth_ files by executing all the notebook cells except **Training** session.
