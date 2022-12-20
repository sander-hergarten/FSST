from tf_agents.trajectories import time_step as ts
import numpy as np
import tensorflow as tf


class NormalizedReturn:
    def __init__(self, environment, batch_size=1):
        norm_values_easy = {
            "coinrun": (5, 10),
            "dtarpilot": (2.5, 64),
            "caveflyer": (3.5, 12),
            "dodgeball": (1.5, 19),
            "fruitbot": (-1.5, 32.4),
            "chaser": (0.5, 13),
            "miner": (1.5, 13),
            "jumper": (3, 10),
            "leaper": (3, 10),
            "maze": (5, 10),
            "bigfish": (1, 40),
            "heist": (3.5, 10),
            "climber": (2, 12.6),
            "plunder": (4.5, 30),
            "ninja": (3.5, 10),
            "bossfight": (0.5, 13),
        }

        self.r_min, self.r_max = norm_values_easy[environment]
        self.batch_size = batch_size
        self.storage = np.zeros((batch_size))

    def __call__(self, batch):
        self.storage += batch.reward.numpy()

    def result(self):
        print(self.storage)
        return np.mean(self.r_mean(self.storage))

    def r_mean(self, storage):
        return (np.array(storage) - self.r_min) / (self.r_max - self.r_min)

class VideoSaver:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.storage = []

    def __call__(self, batch):
        self.storage.append(batch.observation.numpy())

    def result
