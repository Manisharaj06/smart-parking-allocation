import numpy as np
from env.parking_env import ParkingEnv

# Load environment
env = ParkingEnv()

# Load trained Q-table
q_table = np.load("models/q_table.npy")

# Reset environment
state, _ = env.reset()

print("Initial Parking State:")
env.render()

done = False

while not done:

    # Convert state to index
    state_index = int("".join(map(str, state)), 2)

    # Choose best action
    action = np.argmax(q_table[state_index])

    print(f"\nAI chose slot: {action}")

    # Perform action
    state, reward, done, _, _ = env.step(action)

    env.render()

    print("Reward:", reward)

print("\nAll parking slots occupied!")