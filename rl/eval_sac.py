import gym
import numpy as np
import argparse
from configs import config
import nav2D_envs
from stable_baselines3 import SAC


def main(model_path: str):

    # Define Environment and Model
    env = gym.make('nav2D_envs/Nav2DWorld-v0')
    model = SAC.load(model_path)

    # Make sure zero noise
    model.action_noise = None

    obs = env.reset()
    max_steps = 500
    while True:
        for i in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break
        obs = env.reset()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning - based path planning")
    parser.add_argument("--experiment", type=str,
                        default="20", required=False)
    args = parser.parse_args()
    print("\n Training SAC with configuration parameters: \n", config)
    config.merge_from_file("experiments/" + args.experiment + "/config.yaml")
    main("experiments/" + args.experiment + "/sac_nav")
