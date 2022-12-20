# Imports
from typing import Any, Callable, Iterable

import tensorflow as tf

from tf_agents.environments import PyEnvironment, tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


import numpy as np

# Classes
class GymEvironment(PyEnvironment):
    trackers: list = []
    tracker_interface: dict[str, Callable] = {}

    current_episode: int = 0
    current_step: int = 0

    def __init__(self, environment_spec, env):
        self._action_spec = environment_spec.action_spec
        self._observation_spec = environment_spec.observation_spec
        self._time_step_spec = environment_spec.time_step_spec
        self._discount_spec = environment_spec.discount_spec
        self._reward_spec = environment_spec.reward_spec
        self.env = env
        self._episode_ended = True

        self.tracker_interface["get_current_episode"] = lambda: self.current_episode
        self.tracker_interface["get_current_step"] = lambda: self.current_step
        self.tracker_interface["episode_in_progress"] = lambda: not self._episode_ended

        self.tracker_calls = {
            getattr(callable, "__name__", repr(callable)): 0
            for callable in self.tracker_interface
        }

    def _reset(self):
        self._episode_ended = True

        # self.reset_tracker()

        self._state = self.env.reset()

        self._episode_ended = False

        self.current_step = 0
        self.current_episode += 1
        self.current_step_type = ts.StepType.FIRST

        return ts.restart(np.array(self._state, dtype=np.int32))

    def reward_spec(self):
        return self._reward_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:

        self.reset() if self._episode_ended or self.current_step >= 4000 else None

        self.current_step += 1

        step_data: tuple[Iterable, int, bool, dict] = self.env.step(action)

        self._state, self.reward, self._episode_ended, self.etc = step_data

        # self.run_tracker(step_data)

        if self._episode_ended:
            self.current_step_type = ts.StepType.LAST
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        self.current_step_type = ts.StepType.MID

        return ts.transition(
            np.array(self._state, dtype=np.int32), reward=self.reward, discount=1.0
        )

    def set_discount_factor(self, discount_factor: float):
        self.discount_factor = discount_factor

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def add_tracker(self, tracker):
        self.trackers.append(tracker)

    def run_tracker(self, step_data):
        for tracker in self.trackers:
            tracker(step_data=step_data, **self.tracker_interface)

    def reset_tracker(self):
        for tracker in self.trackers:
            tracker.reset(**self.tracker_interface)

    def to_tf_env(self):
        return tf_py_environment.TFPyEnvironment(self)
