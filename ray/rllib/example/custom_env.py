import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig

class MyDummyEnv(gym.Env):
    def  __init__(self,config=None):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        #return observation and info dict
        return np.array([1.0]),{}

    def step(self, action):
        #return next observation, reward, terminated, truncated, info dict
        return np.array([1.0]), 1.0, True, False, {}

config = (
    PPOConfig()
    .environment(
        MyDummyEnv,
        env_config={},
    )
)

algo = config.build()
print('-------------')
print(algo.train())