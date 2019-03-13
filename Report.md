

# Project 3: Collaboration and Competition

### MADDPG

In the excise of project2, we learn how to use DDPG algorithm to solve the continuous control task. To faciliate the learning process, 
a multi-agent environment is used to collect the experience data (state, action, reward, next_state, done) which are then added to replay buffer. 
Note the data collection process here are done in a parallel manner as each agent is learning to solve the task independently. 
In contrast, the MADDPG is a variant of DDPG algorithm where the agents are no longer working independently but collaborate under certain situations 
(e.g. don't let the ball hit the ground) and compete under other situations (e.g. collect as many points as possible).

To achieve this goal, all agents still train their own actor (_policy gradient_) network using their own observations,
whereas share the critic (_Q-value_ network) training process, i.e., to train the critic network using all agent's observations and actions.
The algorithm details are described in this [paper](https://github.com/ChaoLiRV/Udacity_DRL_Collaboration_and_Competition/blob/master/MADDPG_paper.pdf)
 
### Implementation details
_**actor network:**_ It takes the state observation as input and returns the action as the output. The neural network has two hidden layers with 256 and 128 nodes. Both layers use _ReLU_ activation function that performs better than
_leaky_ReLU_ in terms of learning efficiency. The output layer uses _tanh_ as activation function

_**critic network:**_ Similar architecture as actor network, except that it takes both state observation and action as input and return the scalar (_Q_ value) as the output.
The state input happens at the input layer, and then the output of the first hidden layer is concatenated with the action to feed the second hidden layer.

In the algorithm, there're two MADDPG agent instances interacting with the environment to learn their own actor and critic.
Both agents add their experience to the shared replay buffer with size _1e+6_. The learning process is in a batch manner with _256 batchsize_ at every other time step.
The reward discount _gamma_ is _0.99_ and soft update parameter _TAU_ for the target network is _6e-2_. Need experiments to determine this value in order to strike the balance between the learning speed and stability. 
The learning rate for actor is _1e-4_, whereas for critic it is _3e-4_ for fast learning as the critic is the basis for actor training. The noise added to action is sampled from standard normal distribution.
The noise level start at a high level _6_ and decay to _0_ with _1/256_ episode rate. 

### Score plot
The environment is solved at **1126** episodes. Refer to the plot below to see how the average scores evolve as a function of episodes.

![score plot](https://github.com/ChaoLiRV/Udacity_DRL_Collaboration_and_Competition/blob/master/scores.png)  

### Future work
Prioritized Experience replay can be implemented to faciliate the learning process, as the learning focuses on more important experiences.
Also, it is observed that the learning process is unstable. From the score plot above, we see the score goes up and then drop around 900 episodes. 
Even the environment is solved at 1126 episode, but the score is still not guaranteed to stay above +0.5 if running more episodes.
Will try implement A2C and A3C to the model and compare the performance of different algorithms. 