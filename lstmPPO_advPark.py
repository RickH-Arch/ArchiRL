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

from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from envs.AdvanceParkingEnv.manual_park_reader import ManualParkReader

file_path = 'envs/AdvanceParkingEnv/manual_park_data.csv'

reader = ManualParkReader()
units_pack = reader.read(file_path)

config = {
    "units_pack": units_pack,
    "vision_range": 7,
    "save":True,
    "max_step_index": 1.5
}

env = AdvancePark(config)

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

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[256,512,256],vf=[256,512,256]),
    lstm_hidden_size=512,
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
    n_steps = 2000,
    batch_size=256,
    n_epochs=8,
    gamma = 0.99,
    gae_lambda = 0.95,
    ent_coef=0.05,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    learning_rate=0.005
)

eval_callback = EvalCallback(vec_env,
                             best_model_save_path="./parking_best_model",
                             log_path="./parking_eval_log",
                             eval_freq=3000,
                             n_eval_episodes=5,
                             deterministic=False,
                             verbose=0,
                             render=False
                             )

env.model = model

callbacks = CallbackList([eval_callback])

total_timesteps = 1000000

model.learn(total_timesteps=total_timesteps,progress_bar=True,callback=callbacks)