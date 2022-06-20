import gym
from gym import spaces
import pygame
import numpy as np


class Nav2DWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=512):
        self.size = size  # The size of the square grid
        self.window_size = size  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.num_obstacles = 10
        self.num_obs_considered = 3
        self.max_vel = 10 / self.size
        self.agent_size = 20 / self.size
        self.obs_min_size = 5 / self.size
        self.obs_max_size = 40 / self.size
        low_obs_p = np.array([-2.0] * 2 * self.num_obs_considered)
        high_obs_p = np.array([2.0] * 2 * self.num_obs_considered)
        #low_obs_v = np.array([-self.max_vel] * self.num_obstacles) #360 degree scan to a max of 4.5 meters
        #high_obs_v = np.array([self.max_vel] * self.num_obstacles)
        #low_obs_size = np.array([self.obs_max_size] * self.num_obstacles)
        #high_obs_size = np.array([self.obs_min_size] * self.num_obstacles)
        self.velocity = np.array([0.0, 0.0])
        self.num_steps = 0

        """self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "obstacles_pos": spaces.Box(low_obs_p, high_obs_p, dtype=np.int16),
                "obstacles_vel": spaces.Box(low_obs_v, high_obs_v, dtype=np.int16),
                "obstacles_size": spaces.Box(self.obs_min_size, self.obs_max_size, dtype=np.int16)
            }
        )"""

        # Observation space for static obstacles
        """# for examples/sac.py
	low = np.hstack((np.array([0, 0]), np.array([0, 0]), low_obs_p, low_obs_p, low_obs_size))
        high = np.hstack((np.array([size - 1, size - 1]), np.array([size - 1, size - 1]), high_obs_p, high_obs_p,
                          high_obs_size))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int16)"""

        # For examples/her/her_sac_gym_fetch_reach.py
        low = np.hstack((np.array([-2.0, -2.0]), low_obs_p, np.array([-self.max_vel]*2)))
        high = np.hstack((np.array([2.0, 2.0]), high_obs_p, np.array([-self.max_vel]*2)))
        """self.observation_space = spaces.Dict(
            {
                'observation': spaces.Box(low=low, high=high, dtype=np.float32),
                'desired_goal': spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32),
                'achieved_goal': spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            }
        )"""

        # Observation space for moving obstacles
        """low = np.hstack((np.array([0, 0]), np.array([0, 0]), low_obs_p, low_obs_p, low_obs_v, low_obs_v, low_obs_size))
        high = np.hstack((np.array([size - 1, size - 1]), np.array([size - 1, size - 1]), high_obs_p, high_obs_p, high_obs_v, high_obs_v, high_obs_size))"""
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        # self.action_space = spaces.Box(0, 4, shape=(1,), dtype=np.int16)
        self.action_space = spaces.Box(np.array([-self.max_vel]*2), np.array([self.max_vel]*2), dtype=np.float32)


        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        """self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]) # Should never happen
        }"""
        """self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([1, 1]),
            3: np.array([0, 1]),
            4: np.array([-1, 1]),
            5: np.array([-1, 0]),
            6: np.array([-1, -1]),
            7: np.array([0, -1]),
            8: np.array([-1, -1]),
        }"""
        

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        #return {"agent": self._agent_location}, "target": self._target_location, "obstacles_pos": self._obstacles_positions, "obstacles_vel": self._obstacles_vels}
        # Static obstacles
        """obs = np.hstack((self._agent_location, self._target_location, self._obstacles_positions.flatten(),
                         self._obstacles_size))"""

        # Calculate relative observations
        dir_goal = self._target_location - self._agent_location

        dir_obs = np.zeros((2))
        dir_obs_closest = 2*np.ones((2, self.num_obs_considered))
        for i in range(self._obstacles_positions.shape[0]):
            dir_obs = self._obstacles_positions[i] - self._agent_location
            # Find collision point
            dir_obs -= dir_obs / np.linalg.norm(dir_obs) * (self.agent_size + self._obstacles_size[i] / self.size)
            # Save obstacles with shortest distance
            for j in range(self.num_obs_considered-1, -1, -1):
                if np.linalg.norm(dir_obs) < np.linalg.norm(dir_obs_closest[:,j]):
                    if j+1 < self.num_obs_considered:
                        dir_obs_closest[:,j+1] = dir_obs_closest[:,j]
                    dir_obs_closest[:,j] = dir_obs

        # For examples/her/her_sac_gym_fetch_reach.py
        """obs = {
            'observation': np.hstack((dir_goal, dir_obs.flatten())),
            'achieved_goal': self._agent_location,
            'desired_goal': self._target_location
        }"""
        obs = np.hstack((dir_goal, dir_obs_closest.flatten(), self.velocity))

        # Moving obstacles
        # obs = np.hstack((self._agent_location, self._target_location, self._obstacles_positions.flatten(), self._obstacles_vels.flatten(), self._obstacles_size))
        return obs

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.uniform(-1.0, 1.0, size=2)
        # self._agent_location = np.array([self.size,0])


        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.linalg.norm(self._target_location - self._agent_location) < 2.0 * self.agent_size:
            self._target_location = self.np_random.uniform(-1.0, 1.0, size=2)

        # Sample the obstacles' radius randomly
        self._obstacles_size = self.np_random.uniform(self.obs_min_size, self.obs_max_size, size=self.num_obstacles)

        # Sample the obstacles' locations randomly until it does not coincide with the agent's location or the targe location
        obstacles = []
        for i in range(self.num_obstacles):
            obs_location = self._agent_location
            while((np.linalg.norm(obs_location - self._agent_location) < 2*(self.agent_size + self._obstacles_size[i])) or (np.linalg.norm(obs_location - self._target_location) < 2*(self.agent_size + self._obstacles_size[i]))):
                obs_location = self.np_random.uniform(-1.0, 1.0, size=2)
            obstacles.append(obs_location)
        self._obstacles_positions = np.array(obstacles)

        # Set obstacle velocities
        # Static obstacles
        self._obstacles_vels = np.zeros((self.num_obstacles,2))

        # Moving obstacles
        #self._obstacles_vels = self.np_random.integers(-self.max_vel, self.max_vel, size=(self.num_obstacles,2))

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = action
        self.velocity = action
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location += direction
        # self._agent_location = np.clip(
        #     self._agent_location, self.agent_size, self.size - self.agent_size
        # )
        # Move the obstacles
        """for i in range(self._obstacles_positions.shape[0]):
            self._obstacles_vels[i] += np.random.randint(low=-1, high=1, size=(2,))
            self._obstacles_vels[i] = np.clip(self._obstacles_vels[i], -self.max_vel, self.max_vel)
            # self._obstacles_positions[i] = np.clip(self._obstacles_positions[i]+np.int16(self._obstacles_vels[i]), self.agent_size, self.size - self.agent_size)
            self._obstacles_positions[i] += np.int16(self._obstacles_vels[i])

            if(self._obstacles_positions[i][0] < self._obstacles_size[i] or self._obstacles_positions[i][0]>self.size - self._obstacles_size[i]):
                self._obstacles_vels[i][0] *= -1
                self._obstacles_positions[i] = np.clip(self._obstacles_positions[i] + np.int16(self._obstacles_vels[i]), self._obstacles_size[i], self.size - self._obstacles_size[i])
            if(self._obstacles_positions[i][1] < self._obstacles_size[i] or self._obstacles_positions[i][1]>self.size - self._obstacles_size[i]):
                self._obstacles_vels[i][1] *= -1
                self._obstacles_positions[i] = np.clip(self._obstacles_positions[i] + np.int16(self._obstacles_vels[i]), self._obstacles_size[i], self.size - self._obstacles_size[i])
            # print("pos: {}".format(self._obstacles_positions[i]))
            # print("vel: {}".format(self._obstacles_vels[i]))
        """


        # An episode is done iff the agent has reached the target or crashed into an obstacle
        arrived_goal = np.linalg.norm(self._agent_location-self._target_location) < 2*self.agent_size
        # Check for collisions
        collision = False
        for i, obs in enumerate(self._obstacles_positions):
            if(np.linalg.norm(self._agent_location-obs) < 2*(self.agent_size+self._obstacles_size[i])):
                collision = True
                break
        # Check for out of boundaries
        if(self._agent_location[0] < -1.0 + self.agent_size or self._agent_location[0] > 1.0 - self.agent_size or
                self._agent_location[1] < -1.0 - self.agent_size or self._agent_location[1] > 1.0 - self.agent_size):
            collision = True
        # Check if max. number of steps is reached
        self.num_steps += 1
        max_steps_reached = False
        if self.num_steps >= 500:
            max_steps_reached = True

        # Sparse Reward
        if arrived_goal:
            reward = 1
        #elif collision:
        #    reward = -1
        #elif max_steps_reached:
        #    reward = -1
        else:
            reward = 0
        observation = self._get_obs()
        info = self._get_info()
        done = arrived_goal or collision or max_steps_reached

        # Reset steps
        if done:
            self.num_steps = 0
            self.velocity = np.array([0, 0])
        return observation, reward, done, info

    def compute_reward(self, archieved_goal, desired_goal, info):
        # Normalize distance to desired goal
        return 1 #(1 - np.linalg.norm(archieved_goal - desired_goal) / 724)

    def render(self, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.agent_size*self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                (self._target_location+1.0)/2*self.size-np.array([pix_square_size, pix_square_size]),
                (pix_square_size*2, pix_square_size*2),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            ((self._agent_location+1.0)/2*self.size),
            pix_square_size,
        )

        # Finally, add obstacles
        for i, obs_pos in enumerate(self._obstacles_positions):
            pygame.draw.circle(
            canvas,
            (255, 0, 0),
            ((obs_pos+1.0)/2*self.size),
            self._obstacles_size[i]*self.size,
        )

         # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, 50 * x),
                (self.window_size, 50 * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (50 * x, 0),
                (50 * x, self.window_size),
                width=3,
            )

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

