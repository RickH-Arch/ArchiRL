from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.AdvanceParkingEnv.advance_park import AdvancePark
from envs.AdvanceParkingEnv.park_reader import ParkReader
import time




data_name = '4街3'
data_path = f'data/{data_name}.csv'

reader = ParkReader()
units_pack = reader.read(data_path,[(5,4)])

config = {
    "units_pack": units_pack,
    "vision_range": 7,
    "save":True,
    "max_step_index": 1.5,
    "train":False,
    #"render_mode":"human"
}
env = AdvancePark(config)

#env = DummyVecEnv([lambda:env])
#env = VecMonitor(env)
obs,_ = env.reset()

model = RecurrentPPO.load("./parking_best_model/4街3_20250514/best_model_1.zip")

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    
    #env.render()
    #time.sleep(0.1)


