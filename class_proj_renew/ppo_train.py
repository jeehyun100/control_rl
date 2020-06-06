import tensorflow as tf
from manipulator_2d import Manipulator2D
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.policies import LnMlpPolicy
from stable_baselines import PPO2
import os
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines.bench import Monitor


# Log dir
log_dir = "./tmp5/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

env = Manipulator2D()
env = Monitor(env, log_dir)
# Custom MLP policy of two layers of size 32 each with tanh activation function
#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])

# Create the agent

#model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs,)
model = PPO2(MlpPolicy, env, verbose=1)
# Train the agent
model.learn(total_timesteps=20000000, callback=callback)
# Save the agent
model.save("ppo2-mani12")

# del model
# # the policy_kwargs are automatically loaded
# model = PPO2.load("ppo2-cartpole")