import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

from typing import Optional

from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment
)

from ray.tune.registry import get_trainable_cls, register_env

parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)
parser.add_argument(
    "--corridor-length",
    type=int,
    default=10,
    help="The length of the corridor in fields. Note that this number includes the "
    "starting- and goal states.",
)


class SimpleCorridor(gym.Env):
    """
    Simple corridor environment with a goal state.
    """
    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.end_pos = config.get("corridor_length", 10)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        random.seed(seed)
        self.cur_pos = 0
        return np.array([self.cur_pos], dtype=np.float32), {"env_state":"reset"}
    
    def step(self, action: int):
        assert action in[0,1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1

        terminated = self.cur_pos >= self.end_pos
        truncated = False
        
        reward = random.uniform(0.5, 1.5) if terminated else -0.01
        return np.array([self.cur_pos], dtype=np.float32), reward, terminated, truncated, {"env_state": "step"}
    
if __name__ == "__main__":
    args = parser.parse_args()
    
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(env=SimpleCorridor,
                     env_config={"corridor_length": args.corridor_length})
    )
    
    run_rllib_example_script_experiment(base_config, args)

            
        