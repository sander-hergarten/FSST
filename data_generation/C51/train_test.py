from tf_agents.environments import random_py_environment, tf_py_environment
from tf_agents.networks import encoding_network
from tensorflow.keras.layers import Lambda, UnitNormalization
from tf_agents.specs import array_spec
import tensorflow as tf
import numpy as np
from model import ReinforceNetwork

# Environment definition
observation_spec = array_spec.BoundedArraySpec(
    (64, 64, 3), dtype=np.float32, minimum=0, maximum=255
)

action_spec = array_spec.BoundedArraySpec((15,), dtype=np.float32, minimum=0, maximum=1)

random_env = random_py_environment.RandomPyEnvironment(
    observation_spec, action_spec=action_spec
)

tf_env = tf_py_environment.TFPyEnvironment(random_env)


timestep = tf_env.reset()

coolnet = ReinforceNetwork(observation_spec, action_spec)

# # Input encoding
# preprocessing_layers = Lambda(lambda x: tf.divide(x, 255))
#
# conv_layer_params = [
#     (32, 4, 1),
#     (64, 8, 1),
#     (128, 8, 1),
# ]
# fc_layer_params = [256, 128]
# dropout_layer_params = [0.2, 0.1]
# preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
#
# print("gugu", observation_spec)
# print("gaga", action_spec)
#
# _encoder = encoding_network.EncodingNetwork(
#     observation_spec,
#     preprocessing_layers=preprocessing_layers,
#     conv_layer_params=conv_layer_params,
#     fc_layer_params=fc_layer_params,
#     dropout_layer_params=dropout_layer_params,
#     activation_fn=tf.keras.activations.relu,
#     batch_squash=False,
# )

# print(_encoder(timestep.observation))
# print(coolnet_encoder(timestep.observation.numpy()[0], step_type=(), network_state=()))
print(coolnet(timestep.observation, timestep.step_type))
