from envs.AdvanceParkingEnv.manual_park_reader import ManualParkReader
from envs.AdvanceParkingEnv.advance_park import AdvancePark
import keyboard
import time
import sys
import os
sys.path.append(os.path.abspath(__file__))


file_path = 'envs/AdvanceParkingEnv/manual2.csv'

def main():
    reader = ManualParkReader()
    units_pack = reader.read(file_path,[(6,4)])
    unit = units_pack.get_unit_byState(39)
    config = {
        "units_pack": units_pack,
        "vision_range": 7,
        "render_mode": "human"
    }

    park = AdvancePark(config)
    park.reset()
    park.render()

    done = False
    total_reward = 0

    while not done:
        action = None

        if keyboard.is_pressed("up"):
            action = 0#forward
        elif keyboard.is_pressed("down"):
            action = 1#backward
        elif keyboard.is_pressed("left"):
            action = 2#left
        elif keyboard.is_pressed("right"):
            action = 3#right

        if action is not None:
            observation, reward, terminated, truncated, info = park.step(action)
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
            total_reward += reward
            done = terminated or truncated

            park.render()
            time.sleep(0.1)

        if done:
            print(f"回合结束，总奖励: {total_reward}")
            observation, info = park.reset()
        

    park.close()

if __name__ == "__main__":
    main()