from envs.simple_park import SimplePark
import keyboard
import time
import sys
import os
sys.path.append(os.path.abspath(__file__))


def main():

    config = {
        "nrow": 10,
        "ncol": 12,
        "vision_range": 7,
        "disabled_states": [40,41,42,52,53,54,64,65,66,
                            94,95,106,107,118,119,
                            0,12,24,36,48,60],
        
        "entrances_states": [59,2,113],
        
        "render_mode": "human"
    }

    park = SimplePark(config)
    park.reset()
    
    park.render()
    done = False
    total_reward = 0
    while not done:
        action = None

        if keyboard.is_pressed("up"):
            action = 0
        elif keyboard.is_pressed("down"):
            action = 1
        elif keyboard.is_pressed("left"):
            action = 2
        elif keyboard.is_pressed("right"):
            action = 3

        if action is not None:
            observation, reward, terminated, truncated, info = park.step(action)
            print(f"reward: {reward}, terminated: {terminated}, truncated: {truncated}")
            #print(f"observation: {observation}")
            total_reward += reward
            done = terminated or truncated
            park.render()
            
            time.sleep(0.1)

        if done:
            print(f"回合结束，总奖励: {total_reward}")
            observation, info = park.reset()
            done = False
            total_reward = 0
    park.close()

if __name__ == "__main__":
    main()