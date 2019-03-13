

# Project 2: Continuous Control task

### DDPG

In this excise, we learn how to use DDPG algorithm to solve the continuous control task. 
The DDPG is a model-free, off-policy actor-critic algorithm based on the deterministic policy gradient 
[DPG](http://proceedings.mlr.press/v32/silver14.pdf). It uses deep neural network as function approximator
to learn policies in the high-dimensional continuous action spaces. Moreover, it is able to learn the
large, non-linear actor-critic in a stable and robust way, as it combines the insights from the Deep Q Network 
into the algorithm: 1. train the network off-policy with samples from a replay buffer to minimize correlations 
between samples; 2. use double Q network with separated target and local Q network to increase the stability.
 
The algorithm details is seen **Algorithm 1** figure below. First it creates an agent with both critic _Q(s,a|&theta;)_ and 
actor _&pi;(s|&theta;)_ network initialized. Since this work is an episodic task, the network is trained iteratively
with many episodes. Within each episode, the action _a<sub>t</sub>=&pi;(s|&theta;)+Noise<sub>t</sub>_ is generated
based on the current policy network &pi;, and here the noise term is added to allow for exploration. 
By executing the action, we can observe the reward and the new state, and then add these experiences to the replay buffer.
Next it comes to the essential part of the algorithm, to learn the model weight. The local critic network parameters are 
updated by minimizing the error between estimated Q target and the actual Q value at current timestep. Then the local 
actor network parameters can be learned by maximizing the expected Q(s,a') values where a' is from the actor network prediction &pi;(s). 
After the local critic and local actor network weights are updated, a soft update on the target critic and actor network is done.

![DDPG algorithm](https://github.com/ChaoLiRV/Udacity_DRL_ContinuousControl/blob/master/ddpg_algo.png)

To faciliate the learning process, a 20 identical agent environment is used to collect the experience data 
(state, action, reward, next_state, done) which are then added to replay buffer. 
Note the data collection process here are done in a parallel manner. 
 
### Implementation details
_**actor network:**_ It takes the state observation as input and returns the action as the output. The neural network has two hidden layers with 256 and 128 nodes. 
Both layers use _leaky_ReLU_ activation function that empirically performs better than
_ReLU_ in terms of learning efficiency. The output layer uses _tanh_ as activation function

_**critic network:**_ Similar architecture as actor network, except that it takes both state observation and action as input and return the scalar (_Q_ value) as the output.
The state input happens at the input layer, and then the output of the first hidden layer is concatenated with the action to feed the second hidden layer.

In the algorithm, there're twenty identical DDPG agent instances interacting with the environment to learn the actor and critic network.
The replay buffer size is set _1e+5_. The learning process is in a batch manner with _256 batchsize_ at every other time step.
The reward discount _gamma_ is _0.99_ and soft update parameter _TAU_ for the target network is _1e-3_. Need experiments to determine this value in order to strike the balance between the learning speed and stability. 
The learning rate for both actor and critic is _1e-4_. The noise added to action is sampled from standard normal distribution with _mu_ = 0, _theta_ = 0.15 and _sigma_ = 0.2. 

### Score plot
The environment is solved at **38** episodes. Refer to the plot below to see how the average scores evolve as a function of episodes.

![score plot](https://github.com/ChaoLiRV/Udacity_DRL_ContinuousControl/blob/master/score_plot.png)  

### Future work
- Prioritized Experience replay can be implemented to faciliate the learning process, as the learning focuses on more important experiences.
- The noise is introduced when generating actions to allow for exploration. It's reasonable to set a large noise level at early learning stage in order 
to fully explore the environment. Then progressively decay the noise level to certain level to allow for better learning efficiency
- The DDPG paper also mentions to use Batch normalization to preprocess the input to network, which is expected to 
further improve the model stability and learning speed.  