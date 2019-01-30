import random
import numpy as np
import torch
from collections import deque, namedtuple
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.seed = random.seed(seed)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
    def step(self, state, action, reward, next_state,done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.learn(self.memory.sample())

    def act(self):
        """Returns actions for given state as per current policy."""

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
           actor_target(state) -> action
           critic_target(state, action) -> Q-value

        Params
        ======
           experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
           gamma (float): discount factor
        """
    def update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """



class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, seed):
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience',
                                     field_names=['state','action','reward','next_state','done'])

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.unit8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def __len__(self):
        return len(self.memory)
