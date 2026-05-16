import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class ParkingEnv(gym.Env):

    def __init__(self):

        super(ParkingEnv, self).__init__()

        # Total parking slots
        self.total_slots = 5

        # Action space:
        # Agent can choose slot 0 to 4
        self.action_space = spaces.Discrete(self.total_slots)

        # Observation space:
        # 0 = free
        # 1 = occupied
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.total_slots,),
            dtype=np.int32
        )

        # Initial parking state
        self.state = np.zeros(self.total_slots, dtype=np.int32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Random parking occupancy
        self.state = np.random.randint(0, 2, size=(self.total_slots,))

        return self.state, {}

    def step(self, action):

        reward = 0
        done = False

        # If chosen slot is free
        if self.state[action] == 0:

            reward = 10

            # Occupy the slot
            self.state[action] = 1

        else:
            reward = -10

        # End episode if all slots occupied
        if np.all(self.state == 1):
            done = True

        return self.state, reward, done, False, {}

    def render(self):

        print("\nParking Slots:")
        print(self.state)