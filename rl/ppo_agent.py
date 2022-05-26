import os
import pdb

import numpy as np
from .models.actor_critic import ActorCritic
import nav2D_envs

import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, nsteps=200, epochs=10, nbatchs=32,
                 ratio_clip=0.2, lrate=1e-3, lrate_schedule=lambda it: 1.0, beta=0.01,
                 gae_tau=0.95, gamma=0.99, weight_decay=0.0, gradient_clip=0.5, restore=None, train_mode = True):
        
        self.env = gym.make('nav2D_envs/Nav2DWorld-v0')
        self.num_agents = 1
        self.policy = ActorCritic(state_size = self.env.state_size, action_size = self.env.action_size).to(device)
        self.nsteps = nsteps
        
        self.gamma = gamma
        self.epochs = epochs
        self.nbatchs = nbatchs
        self.ratio_clip = ratio_clip
        self.lrate = lrate
        self.gradient_clip = gradient_clip
        self.beta = beta
        self.gae_tau = gae_tau
        self.restore = restore
        self.lrate_schedule = lrate_schedule
        self.weight_decay = weight_decay
        
        
        self.state, info = self.env.reset(seed=42, return_info=True)
        self.opt = optim.Adam(self.policy.parameters(), lr=lrate, weight_decay=self.weight_decay)

        # lrate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lrate_schedule)

        # restore weights
        if restore is not None:
            checkpoint = torch.load(restore)
            self.policy.load_state_dict(checkpoint)

        self.reward = np.zeros(self.num_agents)
        self.episodes_reward = []
        self.steps = 0

        assert((self.nsteps * self.num_agents) % self.nbatchs == 0)
    
    @property
    def running_lrate(self):
        return self.opt.param_groups[0]['lr']
    
    
    def save(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.policy.state_dict(), path)
        
    def tensor_from_np(self, x):
        return torch.from_numpy(x).float().to(device)
    
    def get_batch(self, states, actions, old_log_probs, returns, advs):
        length = states.shape[0] # nsteps * num_agents
        batch_size = int(length / self.nbatchs)
        idx = np.random.permutation(length)
        for i in range(self.nbatchs):
            rge = idx[i*batch_size:(i+1)*batch_size]
            yield (
                states[rge], actions[rge], old_log_probs[rge], returns[rge], advs[rge].squeeze(1)
                )
            
    def act(self):
        actions = []
        for state in self.state:
            state = torch.from_numpy(state).float().to(device)
            self.policy.eval()
            with torch.no_grad():
                action, log_p, _, value = self.policy(state)
                action = action.cpu().data.numpy()
                #print(action)
                #print(type(action))
            self.policy.train()
            actions.append(action)
        actions = np.clip(np.array(actions), -1, 1)
        print("actions: {}".format(actions))
        next_states, rewards, dones, _ = self.env.step(actions) # Perform action and get new state and reward
        print("next_states: {}".format(next_states))
        print("rewards: {}".format(rewards))

        self.state = next_states
        return rewards, dones
    
    def reset(self):
        self.state = self.env.reset()
    
    def step(self):
        # step lrate scheduler
        self.scheduler.step()

        # Collect n steps on all agents
        trajectory_raw = []
        for _ in range(self.nsteps):
            # Policy step to choose action based on current state
            state = self.tensor_from_np(self.state)
            action, log_p, _, value = self.policy(state)

            log_p = log_p.detach().cpu().numpy()
            value = value.detach().squeeze(1).cpu().numpy()
            action = action.detach().cpu().numpy()
            
            # Step on the environment to perform action based on policy output
            print("actions: {}".format(action))

            next_state, reward, done, _ = self.env.step(action)
            print("next_states: {}".format(next_state))
            print("rewards: {}".format(reward))
            
            reward = np.array(reward)
            done = np.array(done)
            
            # deal with nan returns from unity environment
            if math.isnan(np.array(reward).mean()):
                reward = np.nan_to_num(reward)
            else:
                self.reward += reward
                for i, d in enumerate(done):
                    if d:
                        self.episodes_reward.append(self.reward[i])
                        self.reward[i] = 0
                # Append each step to the trajectory
                trajectory_raw.append((state, action, reward, log_p, value, 1-done))
                self.state = next_state
        
        #Value calculated by the critic 
        next_value = self.policy(self.tensor_from_np(self.state))[-1].detach().squeeze(1)
        trajectory_raw.append((state, None, None, None, next_value.cpu().numpy(), None))
        trajectory = [None] * (len(trajectory_raw)-1)
        
        # process raw trajectories
        # calculate advantages and returns
        advs = torch.zeros(self.num_agents, 1).to(device)
        R = next_value

        if(len(trajectory_raw) > 1 ):
            for i in reversed(range(len(trajectory_raw)-1)):

                states, actions, rewards, log_probs, values, dones = trajectory_raw[i]
                actions, rewards, dones, values, next_values, log_probs = map(
                    lambda x: torch.tensor(x).float().to(device),
                    (actions, rewards, dones, values, trajectory_raw[i+1][-2], log_probs)
                )
                R = rewards + self.gamma * R * dones
                # without gae, advantage is calculated as:
                #advs = R[:,None] - values[:,None]
                td_errors = rewards + self.gamma * dones * next_values - values
                advs = advs * self.gae_tau * self.gamma * dones[:, None] + td_errors[:, None]
                # with gae
                trajectory[i] = (states, actions, log_probs, R, advs)

            states, actions, old_log_probs, returns, advs = map(
                lambda x: torch.cat(x, dim=0), zip(*trajectory)
                )

            # normalize advantages
            advs = (advs - advs.mean())  / (advs.std() + 1.0e-10)
            # train policy with random batchs of accumulated trajectories
            for _ in range(self.epochs):

                for states_b, actions_b, old_log_probs_b, returns_b, advs_b in \
                    self.get_batch(states, actions, old_log_probs, returns, advs):

                    # get updated values from policy
                    _, new_log_probs_b, entropy_b, values_b = self.policy(states_b, actions_b)

                    # ratio for clipping
                    ratio = (new_log_probs_b - old_log_probs_b).exp()

                    # Clipped function
                    clip = torch.clamp(ratio, 1-self.ratio_clip, 1+self.ratio_clip)
                    clipped_surrogate = torch.min(ratio*advs_b.unsqueeze(1), clip*advs_b.unsqueeze(1))

                    actor_loss = -torch.mean(clipped_surrogate) - self.beta * entropy_b.mean()
                    #critic_loss = 0.5 * (returns_b - values_b).pow(2).mean() 
                    critic_loss = F.smooth_l1_loss(values_b, returns_b.unsqueeze(1))

                    self.opt.zero_grad()
                    (actor_loss + critic_loss).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
                    self.opt.step()

            # steps of the environement processed by the agent 
            self.steps += self.nsteps * self.num_agents