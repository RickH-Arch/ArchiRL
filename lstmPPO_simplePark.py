from sb3_contrib import RecurrentPPO
#from stable_baselines3.common.env_checker import check_env
from envs.simple_park.simple_park import SimplePark
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import numpy as np
import torch
import glob
import os

# env_config = {
#     "nrow": 10, #y
#     "ncol": 12, #x
#     "vision_range": 7,
#     "disabled_states": [40,41,42,52,53,54,64,65,66,
#                             94,95,106,107,118,119,
#                             0,12,24,36,48,60],
#     "entrances_states": [59,2,113],
# }


env_config = {
    "nrow": 15, #y
    "ncol": 21, #x
    "vision_range": 7,
    "disabled_states": [0,21,42,63,
                            210,231,252,273,294,
                            288,289,290,291,292,293,
                            309,310,311,312,313,314],
    "entrances_states": [7,125,192], 
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
    lstm_hidden_size=256,
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
    n_steps = 1500,
    batch_size=256,
    n_epochs=8,
    gamma = 0.99,
    gae_lambda = 0.95,
    ent_coef=0.05,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    learning_rate=0.001
)

#find the most recent model in the parking_best_model folder
model_files = glob.glob("./parking_best_model/*.pth")
if model_files:
    most_recent_model = max(model_files, key=os.path.getctime)
    model.policy.load_state_dict(torch.load(most_recent_model, map_location=device))
else:
    print("No model files found in the parking_best_model folder.")

#print(model.policy)
env.model = model

total_timesteps = 500000

average_reward_callback = AverageRewardCallback(check_freq=300,verbose=0)
eval_callback = EvalCallback(vec_env,
                             best_model_save_path="./parking_best_model",
                             log_path="./parking_eval_log",
                             eval_freq=1000,
                             n_eval_episodes=3,
                             deterministic=False,
                             verbose=2,
                             render=False
                             )
checkpoint_callback = CheckpointCallback(save_freq=1000,save_path="./parking_checkpoints",name_prefix="parking_model",
                                         save_replay_buffer=False,
                                         save_vecnormalize=False,
                                         verbose=0)

callbacks = CallbackList([eval_callback])

model.learn(total_timesteps=total_timesteps,progress_bar=True)

#model.save("parking_model_1")
