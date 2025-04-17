from envs.simple_park_gym import SimplePark
import keyboard
import time

def main():

    config = {
        "nrow": 10,
        "ncol": 10,
        "vision_range": 5,
        "disabled_states": [9,19,29,39,8,18,28,38,
                            77,78,79,87,88,89,97,98,99,
                            23,24,33,34,43,44],
        "entrances_states": [59,2,91],
        "render_mode": "human"
    }

    park = SimplePark(config)
    park.render()
    done = False

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
            done = terminated or truncated
            park.render()
            time.sleep(0.1)

        if done:
            print(f"回合结束，总奖励: {reward}")
            observation, info = park.reset()
            done = False

    park.close()

if __name__ == "__main__":
    main()