from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from envs.AdvanceParkingEnv.advance_park import AdvancePark
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np
import torch
import glob
import sys
import os
sys.path.append(os.path.abspath(__file__))

from stable_baselines3.common.callbacks import EvalCallback, CallbackList,CheckpointCallback

from envs.AdvanceParkingEnv.park_reader import ParkReader


import datetime
TIMENOW = datetime.datetime.now().strftime("%Y%m%d_%H%M")

#data_name = '4街3'
data_name = '二办-2f'

data_path = f'data/{data_name}.csv'
reader = ParkReader()
units_pack = reader.read(data_path,[(7,12),(14,12)])
#units_pack = reader.read(data_path,[(5,4)])
config = {
    "units_pack": units_pack,
    "vision_range": 7,
    "save":True,
    "max_step_index": 2
}

env = AdvancePark(config)

vec_env = DummyVecEnv([lambda:env])
vec_env = VecMonitor(vec_env)
check_env(env)

seed = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def linear_schedule(initial_lr: float):
    """
    线性学习率调度。
    
    :param initial_lr: 初始学习率
    :return: 返回一个函数，根据 progress_remaining 计算当前学习率
    """
    def schedule(progress_remaining: float) -> float:
        """
        progress_remaining: 从 1.0（训练开始）到 0.0（训练结束）
        """
        lr = max(0.00001, initial_lr * progress_remaining)
        #print(f"Learning rate: {lr}")
        return lr
    
    return schedule

policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(pi=[256,256,256],vf=[256,256,256]),
    lstm_hidden_size=512,
    n_lstm_layers=1,
    shared_lstm=False,
    enable_critic_lstm=True,
)

model = RecurrentPPO(
    seed=seed,
    policy="MlpLstmPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log="./lstmPPO_tensorboard",
    device=device,
    n_steps = 1200,
    batch_size=256,
    n_epochs=8,
    gamma = 0.99,
    gae_lambda = 0.95,
    ent_coef=0.05,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    learning_rate=linear_schedule(0.001),
    clip_range=0.2
)

# model = RecurrentPPO.load(path = "./result_model/4街3_20250514/best_model.zip",
#                           seed=seed,
#     policy="MlpLstmPolicy",
#     env=vec_env,
#     policy_kwargs=policy_kwargs,
#     verbose=0,
#     tensorboard_log="./lstmPPO_tensorboard",
#     device=device,
#     n_steps = 1200,
#     batch_size=256,
#     n_epochs=8,
#     gamma = 0.99,
#     gae_lambda = 0.95,
#     ent_coef=0.02,
#     vf_coef= 0.5,
#     max_grad_norm=0.5,
#     learning_rate=linear_schedule(0.0005),
#     clip_range=0.2)
#model.set_env(vec_env)





eval_callback = EvalCallback(vec_env,
                             best_model_save_path=f"./result_model/{data_name}_{TIMENOW}",
                             log_path="./parking_eval_log",
                             eval_freq=3000,
                             n_eval_episodes=5,
                             deterministic=False,
                             verbose=0,
                             render=False
                             )
checkpoint_callback = CheckpointCallback(save_freq=6000,save_path=f"./result_model/checkpoint/c_{data_name}_{TIMENOW}")

env.model = model

callbacks = CallbackList([eval_callback])

total_timesteps = 2000000

model.learn(total_timesteps=total_timesteps,progress_bar=True,callback=callbacks)