import tensorflow as tf
from manipulator_2d import Manipulator2D
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common.policies import LnMlpPolicy
from stable_baselines import PPO2
import os
from callback import SaveOnBestTrainingRewardCallback
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from multiprocessing import Process, freeze_support, set_start_method
#from stable_baselines.common.vec_env import VecMonitor
from vec_monitor import VecMonitor

#
# normalize: true
# n_envs: 16
# n_timesteps: !!float
# 3e5
# policy: 'MlpPolicy'
# ent_coef: 0.0

# Log dir
log_dir = "./tmp11/"
os.makedirs(log_dir, exist_ok=True)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

def make_env( rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        #env = gym.make(env_id)
        env = Manipulator2D()
        #env = Monitor(env, log_dir)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def train_ppo():

    env = Manipulator2D()
    env = Monitor(env, log_dir)
    # Custom MLP policy of two layers of size 32 each with tanh activation function
    #policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])

    # Create the agent
    # env = SubprocVecEnv([make_env( i) for i in range(8)])
    # env = VecMonitor(env, log_dir)
    #model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs,)
    model = PPO2(MlpPolicy, env, verbose=1, nminibatches=32, noptepochs = 10, ent_coef= 0.0)
    # Train the agent
    model.learn(total_timesteps=20000000, callback=callback)
    # Save the agent
    model.save("ppo2-mani14")
    #
    # n_timesteps: !!float
    # 4e6
    # policy: 'MlpPolicy'
    # n_steps: 512
    # nminibatches: 32
    # lam: 0.95
    # gamma: 0.99
    # noptepochs: 10
    # ent_coef: 0.0
    # learning_rate: 2.5e-4
    # cliprange: 0.2

# del model
# # the policy_kwargs are automatically loaded
# model = PPO2.load("ppo2-cartpole")

if __name__ == '__main__':
    freeze_support()
    set_start_method('spawn')
    train_ppo()