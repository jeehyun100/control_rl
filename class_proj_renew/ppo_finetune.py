import tensorflow as tf
from manipulator_2d import Manipulator2D
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.policies import LnMlpPolicy
from stable_baselines import PPO2
import os
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines.bench import Monitor


# Log dir
log_dir = "./tmp2/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

env = Manipulator2D()
#env = Monitor(env, log_dir)
# Create the agent
# Gym Environment 호출


load_model_path = "tmp/ppo_15207000.zip"
#load_model_path = "ppo2-mani7.zip"
#저장된 학습 파일로부터 weight 등을 로드
model = PPO2.load(load_model_path)
env = model.get_env()
# change env
model.set_env(env)
model.learn(total_timesteps=16000000, callback=callback)
# Save the agent
model.save("ppo2-mani9")

# del model
# # the policy_kwargs are automatically loaded
# model = PPO2.load("ppo2-cartpole")