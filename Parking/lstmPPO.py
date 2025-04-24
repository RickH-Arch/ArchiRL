from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from envs.simple_park import SimplePark
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import numpy as np
import gymnasium as gym

env_config = {
    "nrow": 10,
    "ncol": 12,
    "vision_range": 7,
    "disabled_states": [40,41,42,52,53,54,64,65,66,
                            94,95,106,107,118,119,
                            0,12,24,36,48,60],
    "entrances_states": [59,2,113],
    "max_step": 500,
}
env = SimplePark(env_config)
#env = gym.make('CartPole-v1')
vec_env = DummyVecEnv([lambda:env])
vec_env = VecMonitor(vec_env)
#check_env(env)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class AverageRewardCallback(BaseCallback):
    """
    A custom callback that logs average reward over a specified number of steps.
    """
    def __init__(self,check_freq:int,verbose:int=1):
        super(AverageRewardCallback,self).__init__(verbose)
        self.check_freq = check_freq
        self.rewards = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards',[])
        if len(rewards) > 0:
            reward = rewards[0]
            self.rewards.append(reward)
            self.total_steps += 1

            if self.total_steps % self.check_freq == 0:
                avg_reward = np.mean(self.rewards[-self.check_freq:])
                print(f"Average reward over last {self.check_freq} steps: {avg_reward:.2f},total steps: {self.total_steps}")
        return True
        

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[256,256,256],vf=[256,256,256]),
    lstm_hidden_size=128,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
)

model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log="./lstmPPO_tensorboard",
    device=device,
    n_steps = 1000,
    batch_size=256,
    n_epochs=8,
    gamma = 0.99,
    gae_lambda = 0.95,
    ent_coef=0.05,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    learning_rate=0.003
)
#print(model.policy)

total_timesteps = 500000

average_reward_callback = AverageRewardCallback(check_freq=300,verbose=0)
eval_callback = EvalCallback(vec_env,
                             best_model_save_path="./parking_best_model",
                             log_path="./parking_eval_log",
                             eval_freq=1000,
                             n_eval_episodes=8,
                             deterministic=False,
                             verbose=2
                             )
checkpoint_callback = CheckpointCallback(save_freq=1000,save_path="./parking_checkpoints",name_prefix="parking_model",
                                         save_replay_buffer=False,
                                         save_vecnormalize=False,
                                         verbose=0)

callbacks = CallbackList([average_reward_callback])

model.learn(total_timesteps=total_timesteps,progress_bar=True)
