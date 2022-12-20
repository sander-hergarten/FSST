# Imports
from dataclasses import dataclass

import gym
import procgen

from environment import GymEvironment

import tensorflow as tf

from tf_agents import specs
from tf_agents.trajectories import time_step as ts

# %%
# List Environmnents
procgen_names = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# %%
# Environments Class

t_spec = specs.TensorSpec | specs.BoundedTensorSpec


@dataclass
class EnvironmentSpec:
    observation_spec: t_spec
    action_spec: t_spec
    reward_spec: t_spec
    discount_spec: t_spec
    step_type_spec: t_spec
    time_step_spec: ts.TimeStep


class ProcgenEnvironment(GymEvironment):
    def __init__(self, name: str="coinrun"):
        self.name = f"procgen-{name}-v0"

        # Interface initialization
        self._recording_env = gym.make(self.name, render_mode="rgb_array")
        self._primary_env = gym.make(self.name)

        self.tracker_interface["switch_to_render_env"] = self.switch_to_render_env
        self.tracker_interface["switch_to_default_env"] = self.switch_to_default_env

        # find pretty way to parse this
        spec = EnvironmentSpec(
            reward_spec=specs.TensorSpec(shape=(), dtype=tf.float32, name="reward"),  # type: ignore
            step_type_spec=specs.TensorSpec(shape=(), dtype=tf.int32, name="step_type"),  # type: ignore
            action_spec=specs.BoundedTensorSpec((), tf.int32, minimum=0, maximum=14),  # type: ignore
            discount_spec=specs.BoundedTensorSpec(
                shape=(), dtype=tf.float32, minimum=0, maximum=1, name="discount"
            ),  # type: ignore
            observation_spec=specs.TensorSpec(
                shape=(64, 64, 3), dtype=tf.int32, name="observation"
            ),  # type: ignore
            time_step_spec=ts.time_step_spec(
                observation_spec=specs.TensorSpec((64, 64, 3), tf.int32),
                reward_spec=specs.TensorSpec((), tf.float32),
            ),  # type: ignore
        )

        super().__init__(spec, self._primary_env)

    def switch_to_render_env(self):
        if not self._episode_ended:
            raise Exception("cannot switch environment while episode has not ended")

        self.env = self._recording_env

    def switch_to_default_env(self):
        if not self._episode_ended:
            raise Exception("cannot switch environment while episode has not ended")

        self.env = self._primary_env
