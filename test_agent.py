import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("."))

from stable_baselines3 import PPO
from env.parking_env import ParkingEnv

# Load environment
env = ParkingEnv()

# Load trained model
model = PPO.load("models/parking_agent")

# Reset environment
state, _ = env.reset()

print("Initial Parking State:")
env.render()

for step in range(10):

    # Agent predicts best action
    action, _states = model.predict(state)

    print(f"\nAI chose slot: {action}")

    state, reward, done, truncated, info = env.step(action)

    env.render()

    print("Reward:", reward)

    if done:
        print("\nAll parking slots occupied!")
        break