import sys
import os
import time

# Add project root to path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import numpy as np
import random
import mlflow
import matplotlib.pyplot as plt

from env.parking_env import ParkingEnv

# Create environment
env = ParkingEnv()

# Q-table
q_table = np.zeros((32, env.total_slots))

# Hyperparameters
alpha = 0.05
gamma = 0.85
epsilon = 1.0
epsilon_decay = 0.998
min_epsilon = 0.01

episodes = 1000

# MLflow experiment
mlflow.set_experiment("SmartParkingQLearning")

# Timestamp for versioning
timestamp = int(time.time())

# Professional MLflow run name
run_name = f"Parking_Run_{timestamp}"

with mlflow.start_run(run_name=run_name):

    # MLflow tags
    mlflow.set_tag("project", "Smart Parking RL")
    mlflow.set_tag("algorithm", "Q-Learning")
    mlflow.set_tag("developer", "Roshini")

    # Log parameters
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("episodes", episodes)
    mlflow.log_param("epsilon_decay", epsilon_decay)

    rewards_per_episode = []

    print("\nTraining Started...\n")

    # Training loop
    for episode in range(episodes):

        state, _ = env.reset()

        # Convert binary state to integer
        state_index = int(
            "".join(map(str, state)),
            2
        )

        done = False
        total_reward = 0

        while not done:

            # Exploration vs Exploitation
            if random.uniform(0, 1) < epsilon:

                action = env.action_space.sample()

            else:

                action = np.argmax(
                    q_table[state_index]
                )

            next_state, reward, done, _, _ = env.step(action)

            next_state_index = int(
                "".join(map(str, next_state)),
                2
            )

            # Q-learning update
            q_table[state_index, action] = (
                q_table[state_index, action]
                + alpha * (
                    reward
                    + gamma * np.max(q_table[next_state_index])
                    - q_table[state_index, action]
                )
            )

            state_index = next_state_index

            total_reward += reward

        # Reduce exploration gradually
        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

        rewards_per_episode.append(total_reward)

        # Log metrics every 100 episodes
        if episode % 100 == 0:

            print(
                f"Episode: {episode} | Reward: {total_reward}"
            )

            mlflow.log_metric(
                "reward",
                total_reward,
                step=episode
            )

    # =========================
    # MODEL VERSIONING
    # =========================

    model_path = f"models/q_table_{timestamp}.npy"

    np.save(model_path, q_table)

    # Log model artifact
    mlflow.log_artifact(model_path)

    # =========================
    # REWARD GRAPH
    # =========================

    plt.figure(figsize=(10, 5))

    plt.plot(rewards_per_episode)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Q-Learning Training Rewards")

    graph_path = f"logs/reward_plot_{timestamp}.png"

    plt.savefig(graph_path)

    # Log graph artifact
    mlflow.log_artifact(graph_path)

    plt.close()

    print("\nQ-learning training completed!")

    print(f"\nModel saved at: {model_path}")

    print(f"Reward graph saved at: {graph_path}")