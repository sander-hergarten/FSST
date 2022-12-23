from typing import Callable
import wandb
import tracemalloc
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Lambda, Resizing

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


from run_saver import StepSaver
import subprocess

from procgen_env import ProcgenEnvironment
from itertools import product
import argparse


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


tracemalloc.start()


from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import tf_py_environment, BatchedPyEnvironment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from metrics import NormalizedReturn

parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", help="increase output verbosity")
parser.add_argument("--environment", help="increase output verbosity")
args = parser.parse_args()


sweep_threads = 6
parallel_envs = 20

# train_summary_writer = tf.summary.create_file_writer("logs/")


def main(env):
    def evaluate_agent(policy, observers, batch_size=10):
        eval_env_py = BatchedPyEnvironment(
            [ProcgenEnvironment(env) for _ in range(batch_size)]
        )

        eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)

        dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            policy,
            observers=observers,
            num_episodes=20,
        ).run()

    wandb.init()

    class StepCounter:
        step_count = 0

        def step_increment(self, batch):
            self.step_count += 20

    step_counter = StepCounter()

    # with tf.profiler.experimental.Profile("logs/"):
    eval_env_py = ProcgenEnvironment()
    train_py_env = BatchedPyEnvironment(
        [ProcgenEnvironment(env) for _ in range(parallel_envs)]
    )

    eval_env = tf_py_environment.TFPyEnvironment(eval_env_py)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    # Input encoding
    preprocessing_layers = tf.keras.models.Sequential(
        [Lambda(lambda x: tf.divide(x, 255))]
    )
    conv_layer_params = [
        (32, (3, 3), 1),
        (64, (3, 3), 1),
        (64, (3, 3), 1),
    ]
    fc_layer_params = [512 for _ in range(wandb.config.fc_layers)]

    actor_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layers,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    train_step_counter = tf.Variable(0)

    tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=actor_net,
        optimizer=optimizer,
        min_q_value=-10,
        max_q_value=10,
        n_step_update=2,
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=wandb.config.gamma,
        train_step_counter=train_step_counter,
    )

    tf_agent.initialize()

    returns = []
    last_avg_reward = 0
    buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=1000,
        dataset_drop_remainder=True,
    )

    state_saver = StepSaver(parallel_envs)

    for n in range(100):
        # with tf.profiler.experimental.Trace("logs/", step_num=n, _r=1):

        # Observers
        max_rewards_metric = tf_metrics.MaxReturnMetric(batch_size=parallel_envs)

        normalized_return_metric = NormalizedReturn(env, batch_size=parallel_envs)

        average_episode_length_metric = tf_metrics.AverageEpisodeLengthMetric(
            batch_size=parallel_envs
        )

        average_rewards_metric = tf_metrics.AverageReturnMetric(
            batch_size=parallel_envs
        )

        replay_observer = [
            buffer.add_batch,
            max_rewards_metric,
            average_rewards_metric,
            normalized_return_metric,
            average_episode_length_metric,
            # state_saver.add_data_to_queue,
            step_counter.step_increment,
        ]

        dynamic_step_driver.DynamicStepDriver(
            train_env,
            tf_agent.collect_policy,
            observers=replay_observer,
            num_steps=20000,
        ).run()

        state_saver.commit_episodes()

        print(buffer.num_frames())
        dataset = buffer.as_dataset(
            sample_batch_size=10,
            num_steps=10,
            num_parallel_calls=16,
            single_deterministic_pass=True,
        )

        iterator = iter(dataset)

        num_train_steps = wandb.config.epochs

        for _ in range(num_train_steps):
            trajectories, _ = next(iterator)
            tf_agent.train(experience=trajectories)

        returns.append(max_rewards_metric.result().numpy())
        last_avg_reward = average_rewards_metric.result().numpy()

        max_reward_eval, normalized_return_eval, video = evaluate_agent(
            eval_env,
            tf_agent.policy,
        )

        wandb.log(
            {
                "average_reward_training": last_avg_reward,
                "max_reward_training": returns[-1],
                "average_episode_length": average_episode_length_metric.result().numpy(),
                "normalized_return_training": normalized_return_metric.result(),
                "max_reward_eval": max_reward_eval,
                "normalized_return_eval": normalized_return_eval,
                "total_steps": step_counter.step_count,
                "video": wandb.Video(video, fps=15),
            }
        )

    wandb.log(
        {
            "final_normalized_return": np.mean(
                [
                    evaluate_agent(
                        tf_agent.policy,
                    )[1]
                    for _ in range(10)
                ]
            )
        }
    )


if __name__ == "__main__":

    sweep_config = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "final_normalized_return"},
        "parameters": {
            "epochs": {"min": 1, "max": 15},
            "lr": {"max": 0.1, "min": 0.0001},
            "fc_layers": {"min": 1, "max": 12},
            "gamma": {"max": 0.9999, "min": 0.85},
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 20,
        },
    }

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    # main()

    if args.sweep_id:
        sweep_id = args.sweep_id
        wandb.agent(sweep_id=sweep_id, function=main, count=100, project="C51-Coinrun")
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project="C51-Coinrun")
        wandb.agent(sweep_id=sweep_id, function=main, count=100)

    # with ThreadPoolExecutor(max_workers=sweep_threads) as thread_pool:

    #     futures = [
    #         thread_pool.submit(
    #             lambda: wandb.agent(sweep_id=sweep_id, function=main, count=100)
    #         )
    #         for _ in range(sweep_threads)
    #     ]

    #     _, _ = wait(futures, return_when=ALL_COMPLETED)
