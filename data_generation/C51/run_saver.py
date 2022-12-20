import tensorflow as tf
import pandas as pd
import pandas_gbq
import time
from threading import Thread
from collections import deque, defaultdict
from sqlalchemy.engine import create_engine
from sqlalchemy import insert, table, column
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import Trajectory
from uuid import uuid1
import numpy as np


class StepSaver:
    def __init__(self, batch_size, environment_name="Coinrun", dataset=None):

        self.environment_name = environment_name

        self.engine = create_engine("bigquery://deplearn")
        self.batch_size = batch_size
        self.queues = [deque([]) for _ in range(self.batch_size)]

        self.dataset = (
            f"dataset_run_{int(time.time() *1000)}" if not dataset else dataset
        )

        print(f"generating dataset: {self.dataset}")

        with self.engine.connect() as conn:

            conn.execute(f"CREATE SCHEMA {self.dataset}")

        # with self.engine.connect() as conn:
        #     conn.execute(
        #         f"CREATE TABLE {self.dataset}.observations (step int, episode_id string, environment int, observation bytes)"
        #     )

        #     conn.execute(
        #         (
        #             f"CREATE TABLE {self.dataset}.step_data "
        #             "(step int, "
        #             " episode_id string,"
        #             " environment int,"
        #             " action int,"
        #             " step_type int,"
        #             " next_step_type int,"
        #             " reward numeric,"
        #             " discount numeric)"
        #         )
        #     )

        self._helper_loop_trajectory = lambda batch_trajectory, keyword: tf.unstack(
            getattr(batch_trajectory, keyword)
        )

        self.processing_queue = deque([])

    def add_data_to_queue(self, batch_trajectory):
        trajectories = self.loop_trajectory(batch_trajectory)

        for environment_id, (queue, trajectory) in enumerate(
            zip(self.queues, trajectories)
        ):
            queue.append(trajectory)

            if trajectory.step_type != ts.StepType.LAST:
                continue

            queue_copy = queue.copy()
            queue.clear()

            observation, step_data = self.format_episode(environment_id, queue_copy)

            self.processing_queue.append((observation, step_data))

    def loop_trajectory(self, batch_trajectory):

        trajectory_data = [
            self._helper_loop_trajectory(batch_trajectory, attr)
            if attr != "policy_info"
            else [batch_trajectory.policy_info for _ in range(self.batch_size)]
            for attr in [
                "step_type",
                "observation",
                "action",
                "policy_info",
                "next_step_type",
                "reward",
                "discount",
            ]
        ]

        return [Trajectory(*traj) for traj in np.array(trajectory_data).T]

    def commit_episodes(self):
        Thread(target=self._commit_episode).start()

    def _commit_episode(self):

        observations_list = []
        step_data_list = []

        queue_len = len(self.processing_queue)

        for _ in range(queue_len):
            observations_dict, step_data_dict = self.processing_queue.pop()

            observations_list.append(pd.DataFrame(observations_dict))
            step_data_list.append(pd.DataFrame(step_data_dict))

        pandas_gbq.to_gbq(
            pd.concat(observations_list),
            f"{self.dataset}.{self.environment_name}_observations",
            project_id="deplearn",
            if_exists="append",
        )

        pandas_gbq.to_gbq(
            pd.concat(step_data_list),
            f"{self.dataset}.step_data",
            project_id="deplearn",
            if_exists="append",
        )

        print("commited data to db")

    def format_episode(self, environment_id: int, queue: deque):
        time_0 = time.perf_counter()

        episode_length = len(queue)
        episode_id = str(uuid1())

        observations = defaultdict(list)
        step_data = defaultdict(list)

        for step in range(episode_length):
            # step counter
            # episode id

            trajectory = queue.pop()

            observation = {
                "step": step,
                "episode_id": episode_id,
                "environment": environment_id,
                "observation": tf.io.encode_base64(
                    tf.io.encode_jpeg(tf.cast(trajectory.observation, tf.uint8))
                )
                .numpy()
                .decode("utf-8"),
            }

            for key, value in observation.items():
                observations[key].append(value)

            ts_data = {
                "step": step,
                "episode_id": episode_id,
                "environment": environment_id,
                "action": trajectory.action.numpy(),
                "step_type": trajectory.step_type.numpy(),
                "next_step_type": trajectory.next_step_type.numpy(),
                "reward": trajectory.reward.numpy(),
                "discount": trajectory.discount.numpy(),
            }

            for key, value in ts_data.items():
                step_data[key].append(value)
        return observations, step_data

        df = pd.DataFrame(observations)

        df = pd.DataFrame(step_data)
