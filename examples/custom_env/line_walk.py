"""A five-position walk with a terminal reward at either end."""

import gymnasium as gym


class LineWalk(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(2)
        self.position = 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 2
        return self.position, {}

    def step(self, action):
        self.position += 1 if action == 1 else -1
        reward = {0: -1.0, 4: 1.0}.get(self.position, 0.0)
        return self.position, reward, self.position in (0, 4), False, {}


gym.register("LineWalk-v0", entry_point=LineWalk, max_episode_steps=10)
