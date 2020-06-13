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


# Gym Environment 호출
env = Manipulator2D()

load_model_path = "tmp9/acktr_16110000.zip"
#load_model_path = "ppo2-mani7.zip"
#저장된 학습 파일로부터 weight 등을 로드
model = ACKTR.load(load_model_path)

# 시뮬레이션 환경을 초기화
obs = env.reset()

points = 0
total_time = 0

while(total_time <= 120):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:

        total_time += env.t

        # env.buffer = env.buffer_csv
        if "status" in info:
            if info["status"] == "0":
                points += 1
                print("get {0} in time {1}".format(points, env.t))
                #print("robot bound")
        #         #break
        obs = env.reset()
        #break

env.render()