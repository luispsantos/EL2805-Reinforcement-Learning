# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code authors: [Simon Mello - smello@kth.se]
#               [Luis Santos - lmpss@kth.se]

# Load packages
import numpy as np
import gym
from collections import deque, namedtuple
import torch
import torch.nn as nn


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer:
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def random_fill(self, num_exps):
        env = gym.make('LunarLander-v2')
        state = env.reset()
        seen_exps = 0
        
        while seen_exps < num_exps:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            exp = Experience(state, action, reward, next_state, done)
            self.append(exp)
            seen_exps += 1

            if done:
                state = env.reset()
            else:
                state = next_state
        
    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


class NeuralNet(nn.Module):
    # In the init define each layer individually
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)


    # In the forward function you define how each layer is interconnected
    def forward(self, x):
        # First layer (1st hidden layer)
        x = self.linear1(x)  
        x = self.act1(x)
        # Second layer (2nd hidden layer)
        x = self.linear2(x)  
        x = self.act2(x)
        # Third layer (output layer)
        x = self.linear3(x)
        return x
