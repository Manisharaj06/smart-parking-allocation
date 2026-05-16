from env.parking_env import ParkingEnv
import random

env = ParkingEnv()

state, _ = env.reset()

print("Initial State:")
env.render()

for i in range(5):

    action = random.randint(0, 4)

    print(f"\nAgent chose slot: {action}")

    state, reward, done, truncated, info = env.step(action)

    env.render()

    print("Reward:", reward)

    if done:
        print("\nAll slots occupied!")
        break