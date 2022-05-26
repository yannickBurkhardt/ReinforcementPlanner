from rl.ddpg_agent import Agent
import nav2D_envs
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import glob
import os
import gym

actor_model_path = 'rl/experiments/05_ppo/actor-26000.pth'
critic_model_path = 'rl/experiments/05_ppo/critic-26000.pth'


def evaluate():

    env = gym.make('nav2D_envs/Nav2DWorld-v0')
    print(env.state_size)
    print(env.action_size)
    agent = Agent(num_agents = env.num_agents, state_size=env.state_size, action_size=env.action_size, random_seed=2)
    agent.actor_local.load_state_dict(torch.load(actor_model_path))
    agent.critic_local.load_state_dict(torch.load(critic_model_path))

    for i in range(3):
        states = env.reset()
        score = 0                                          # initialize the score
        while True:
            env.render(mode="human") #env.render(mode="rgb_array")
            actions = agent.act(states, add_noise=False)
            next_states, rewards, dones, _ = env.step(actions) # Perform action and get new state and reward
            score += np.average(rewards)                                # update the score
            states = next_states                             # roll over the state to next time step
            if any(dones):                                       # exit loop if episode finished
                break
        print("Score: {}".format(score))
        env.close()

if __name__ == "__main__":
    evaluate()