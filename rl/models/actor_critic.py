import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256, fc3_units=128, fc4_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.bn3 = nn.BatchNorm1d(fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        # self.bn4 = nn.BatchNorm1d(fc4_units)
        # self.fc5 = nn.Linear(fc4_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if len(state.size()) < 2:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.bn4(self.fc4(x)))
        return F.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=512, fc2_units=256, fc3_units=128, fc4_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        # self.bn1 = nn.BatchNorm1d(fcs1_units)

        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units, fc3_units)
        # self.bn3 = nn.BatchNorm1d(fc3_units)

        self.fc4 = nn.Linear(fc3_units, 1)
        # self.bn4 = nn.BatchNorm1d(fc4_units)

        # self.fc5 = nn.Linear(fc4_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.bn4(self.fc4(x)))
        return self.fc4(x)

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 512)
        #self.fc1_bn = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(512, 256)
        #self.fc2_bn = nn.BatchNorm1d(256)
        #self.fc3 = nn.Linear(200, 100)
        #self.fc3_bn = nn.BatchNorm1d(100)

        self.fc_actor_mean = nn.Linear(256, self.action_size)
        self.fc_actor_std = nn.Linear(256, self.action_size)
        self.fc_critic = nn.Linear(256, 1)

        self.std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, x, action=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor
        # print(x)
        # print(self.fc_actor_mean(x))
        try:
            mean = torch.tanh(self.fc_actor_mean(x))  #Continuous action are modeled with a normal probability distribution
            std = F.softplus(self.fc_actor_std(x))
            dist = torch.distributions.Normal(mean, std)
        except:
            print(x)
            print(x.shape)

            print(mean)
            print(std)
            print(dist)
            print(dist.shape)


        if action is None:
            action = dist.sample() # Sample action given the state
        log_prob = dist.log_prob(action) #Calculate the log prob for a given action in the current state given current policy

        # Critic
        # State value V(s)
        v = self.fc_critic(x)

        return action, log_prob, dist.entropy(), v