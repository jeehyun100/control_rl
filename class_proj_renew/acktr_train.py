import tensorflow as tf
from manipulator_2d import Manipulator2D
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.policies import LnMlpPolicy
#from stable_baselines import PPO2
from stable_baselines import ACKTR
import os
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env
#
# normalize: true
# n_envs: 16
# n_timesteps: !!float
# 3e5
# policy: 'MlpPolicy'
# ent_coef: 0.0

# Log dir
log_dir = "./tmp10/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, save_name = "acktr")

env = Manipulator2D()

# multiprocess environment
#env = make_vec_env('CartPole-v1', n_envs=4)
env = Monitor(env, log_dir)
# Custom MLP policy of two layers of size 32 each with tanh activation function
#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])

# Create the agent

#model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs,)
#model = PPO2(MlpPolicy, env, verbose=1)
# Train the agent
model = ACKTR(MlpPolicy, env, verbose=1, ent_coef = 0.0)
# 3e5
# policy: 'MlpPolicy'
# ent_coef: 0.0)
model.learn(total_timesteps=20000000, callback=callback)
# Save the agent
model.save("acktr-man1")

# del model
# # the policy_kwargs are automatically loaded
# model = PPO2.load("ppo2-cartpole")