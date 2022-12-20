import wandb
import time
import tensorflow as tf

from skopt import gp_minimize
from skopt.space.space import Integer

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, BatchedPyEnvironment
from tf_agents.networks import categorical_q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from procgen_env import ProcgenEnvironment
from memory_watcher_daemon import MemoryWatcher

num_iterations = 250  # @param {type:"integer"}
collect_episodes_per_iteration = 2  # @param {type:"integer"}
replay_buffer_capacity = 2000  # @param {type:"integer"}

learning_rate = 1e-3  # @param {type:"number"}
log_interval = 25  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 50  # @param {type:"integer"}

memory_watcher = MemoryWatcher()


def main(parallel_envs):
    try:
        train_py_env = BatchedPyEnvironment(
            [ProcgenEnvironment() for _ in range(parallel_envs[0])]
        )
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)

        # Input encoding
        preprocessing_layers = tf.keras.layers.Lambda(lambda x: tf.divide(x, 255))
        conv_layer_params = [
            (32, 4, 1),
            (64, 8, 1),
            (128, 8, 1),
        ]
        fc_layer_params = [256, 256, 128]

        actor_net = categorical_q_network.CategoricalQNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            preprocessing_layers=preprocessing_layers,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
            gamma=0.99,
            train_step_counter=train_step_counter,
        )

        tf_agent.initialize()

        buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=10000,
        )

        average_rewards_metric = tf_metrics.AverageReturnMetric(train_env.reward_spec())

        replay_observer = [buffer.add_batch, average_rewards_metric]

        # Begin training loop
        time_0 = time.perf_counter()

        # with tf.profiler.experimental.Trace('train', step_num=n, _r=1):
        collect_op = dynamic_step_driver.DynamicStepDriver(
            train_env,
            tf_agent.collect_policy,
            observers=replay_observer,
            num_steps=10000,
        ).run()

        dataset = buffer.as_dataset(sample_batch_size=10, num_steps=100)

        iterator = iter(dataset)

        num_train_steps = 10

        for _ in range(num_train_steps):
            trajectories, _ = next(iterator)
            tf_agent.train(experience=trajectories)
    except MemoryError:
        time_0 = time.perf_counter() - 100

    if any(memory_watcher.is_out_of_memory()):
        memory_watcher.clear_queue()

        time_0 -= 100

    return time.perf_counter() - time_0


if __name__ == "__main__":
    res_gp = gp_minimize(main, [Integer(1, 25)], n_calls=10)

    print("bayesian minimize: ", res_gp)
