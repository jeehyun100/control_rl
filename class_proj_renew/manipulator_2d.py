import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import random

from ou_noise import OUNoise



class Transformation:
    """
    Transformation class for SE(2)
    """

    def __init__(self, matrix=None, translation=(0, 0), rotation=0):
        if isinstance(matrix, None.__class__):
            self._matrix = self.compute_matrix(translation, rotation)
        else:
            self._matrix = matrix.copy()
    
    def __mul__(self, other):
        if isinstance(other, self.__class__):
            tmp = Transformation()
            tmp._matrix = np.matmul(self._matrix, other._matrix)
            return tmp
        elif isinstance(other, np.ndarray):
            if other.shape==(2,):
                return np.matmul(self._matrix, np.concatenate((other, [1])))[:2]
            else:
                return np.matmul(self._matrix, other)
        else:
            return self._matrix * other

    def __str__(self):
        return "Translation: %s\nRotation: %s\nTransfromation matrix:\n%s"%(
            self.get_translation(), self.get_rotation(), self._matrix
        )

    def transform(self, translation=(0, 0), rotation=0):
        self._matrix = np.matmul(self._matrix, self.compute_matrix(translation, rotation))

    def reset(self):
        self._matrix = self.compute_matrix((0, 0), 0)

    def get_translation(self):
        return self._matrix[0:2, 2]

    def get_rotation(self):
        return self._matrix[0:2, 0:2]

    def get_transformation(self):
        return self._matrix

    def x(self, x=None):
        if isinstance(x, None.__class__):
            return self._matrix[0, 2]
        else:
            self._matrix[0, 2] = x

    def y(self, y=None):
        if isinstance(y, None.__class__):
            return self._matrix[1, 2]
        else:
            self._matrix[1, 2] = y

    def euler_angle(self, angle=None):
        if isinstance(angle, None.__class__):
            return np.arctan2(self._matrix[1, 0], self._matrix[0, 0])
        elif isinstance(angle, float):
            self._matrix[0:2, 0:2] = self.compute_matrix((0, 0), angle)[0:2, 0:2]

    def inv(self, return_class=True):
        if return_class:
            tmp = Transformation()
            tmp._matrix = np.linalg.inv(self._matrix)
            return tmp
        else:
            return np.linalg.inv(self._matrix)

    def copy(self):
        tmp = Transformation(matrix=self._matrix)
        return tmp

    @staticmethod
    def compute_matrix(translation, rotation):
        c = np.cos(rotation)
        s = np.sin(rotation)
        return np.array(
            [
                [c, -s, translation[0]],
                [s,  c, translation[1]],
                [0,  0,              1]
            ]
        )



class Manipulator2D(gym.Env):
    
    def __init__(self, arm1=1, arm2=1, dt=0.01, tol=0.1):
        self.env_boundary = 5

        # Observation space를 구성하는 state의 최대, 최소를 지정한다.
        # self.obs_high = np.array([
        #
        #     self.env_boundary, self.env_boundary,
        #     self.env_boundary + arm1, self.env_boundary + arm1,
        #     self.env_boundary + arm2, self.env_boundary + arm2,
        #     self.env_boundary, self.env_boundary,
        #     np.pi, 14.21, np.pi, np.pi, np.pi
        # ])
        #
        # #self.obs_low = -self.obs_high
        # self.obs_low = np.array([
        #     -1*self.env_boundary, -1*self.env_boundary,
        #     -1*(self.env_boundary + arm1), -1*(self.env_boundary + arm1),
        #     -1*(self.env_boundary + arm2), -1*(self.env_boundary + arm2),
        #     -1*(self.env_boundary), -1*(self.env_boundary),
        #     -np.pi, 0, -np.pi, -np.pi, -np.pi
        # ])
        # self.obs_high = np.array([
        # 
        #     np.pi, np.pi, np.pi
        # ])
        # 
        # #self.obs_low = -self.obs_high
        # self.obs_low = np.array([
        #     -np.pi, -np.pi, -np.pi
        # ])

        self.obs_high = np.array([

            self.env_boundary*3, self.env_boundary*3,
            self.env_boundary * 3, self.env_boundary * 3,
            self.env_boundary * 3, self.env_boundary * 3,
            self.env_boundary * 3, self.env_boundary * 3,

        ])

        self.obs_low = -self.obs_high


        #좌표는 더 크게 해야함.. 근데 너무 작게 해서 안된듯 약 15로 노말라이제이션

       # Action space를 구성하는 action의 최대, 최소를 지정한다.
        self.action_high = np.array([1, np.pi, np.pi, np.pi])
        self.action_low = np.array([0, -np.pi, -np.pi, -np.pi])

        # GYM environment에서 요구하는 변수로, 실제 observation space와 action space를 여기에서 구성한다.
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        # 로봇암의 요소를 결정하는 변수
        self.link1_len = arm1 # 로봇팔 길이
        self.link2_len = arm2
        self.dt = dt # Timestep
        self.tol = tol # 목표까지 거리

        # 학습 환경에서 사용할 난수 생성에 필요한 seed를 지정한다.
        self.seed()

        self.target_speed = 1.2

        self.timeout = self.dt * 7000

        # 변수를 초기화한다.
        self.reset()

        
    def step(self, action):
        # 움직이지마
        self._move_target()

        # 강화학습이 만들어낸 action을 위에서 지정한 최대, 최소 action으로 클리핑한다.
        action = np.clip(action, self.action_low, self.action_high)

        # Action으로부터 로봇암 kinematics를 계산하는 부분
        # 여기에서 action은 각 로봇팔1이 x축과 이루는 각의 변화, 로봇팔1과 로봇팔2이 이루는 각의 변화를 말함
        self.robot_tf.transform(
            translation=(action[0]*self.dt, 0),
            rotation=action[1]*self.dt
        )

        self.joint1_tf.transform(rotation=action[2] * self.dt)
        self.joint2_tf.transform(rotation=action[3] * self.dt)

        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        self.t += self.dt

        # Reward와 episode 종료 여부를 확인
        reward, done = self._get_reward()

        # 기타 목적으로 사용할 데이터를 담아둠
        info = {}

        # 시각화 목적으로 사용할 데이터를 self.buffer 에 저장
        self.buffer.append(
            dict(
                robot=self.robot_tf.copy(),
                link1=self.link1_tf_global.copy(),
                link2=self.link2_tf_global.copy(),
                target=self.target_tf.copy(),
                time=self.t,
                reward=reward
            )
        )
        # 시각화 목적으로 사용할 데이터를 self.buffer 에 저장


        # 일반적으로 Gym environment의 step function은 
        # State(observation), 현재 step에서의 reward, episode 종료 여부, 기타 정보로 구성되어있음
        return self._get_state_r(), reward, done, info


    def reset(self):
        # 매 episode가 시작될때 사용됨.
        # 사용 변수들 초기화

        #매 에피소드마다 로봇을 원점으로 이동시키려면 아래 주석을 해제한다.
        #self.robot_tf = Transformation(translation=(3, -5))
        self.robot_tf = Transformation()
        self.joint1_tf = Transformation()
        self.link1_tf = Transformation(translation=(self.link1_len, 0))
        self.joint2_tf = Transformation()
        self.link2_tf = Transformation(translation=(self.link2_len, 0))
        self.link1_tf_global = self.robot_tf * self.joint1_tf * self.link1_tf
        self.link2_tf_global = self.link1_tf_global * self.joint2_tf * self.link2_tf

        # # #목표 지점 생성
        self.target_tf = Transformation(
            translation=(
                random.randrange(-self.env_boundary, self.env_boundary),
                random.randrange(-self.env_boundary, self.env_boundary)
            )
        )

        # #목표 지점 고
        # self.target_tf = Transformation(
        #     translation=(
        #         -5.0,
        #         5.0
        #     )
        # )
        self.ou = OUNoise(dt=self.dt, theta=0.1, sigma=0.2)

        self.done = False
        self.t = 0
        self.buffer = []    # 시각화를 위한 버퍼. episode가 리셋될 때마다 초기화.
        self.buffer_detail = []

        # Step 함수와 다르게 reset함수는 초기 state 값 만을 반환합니다.
        return self._get_state_r()


    def _move_target(self):
        self.target_tf.transform(
            translation = (self.target_speed * self.dt, 0),
            rotation = self.ou.evolve() * self.dt
        )
        if self.target_tf.x() > self.env_boundary:
            self.target_tf.x(self.env_boundary)
        if self.target_tf.x() < -self.env_boundary:
            self.target_tf.x(-self.env_boundary)
        if self.target_tf.y() > self.env_boundary:
            self.target_tf.y(self.env_boundary)
        if self.target_tf.y() < -self.env_boundary:
            self.target_tf.y(-self.env_boundary)


    def _get_reward(self):
        # # 해당 step의 reward를 계산합니다.
        # done = False
        #
        # l = np.linalg.norm(
        #     self.target_tf.get_translation() - self.link2_tf_global.get_translation()
        # )
        # # 목표점을 반지름 sqrt(self.tol)인 원으로 설정
        # if l < self.tol:
        #     reward = 1.
        #     # 목표 근처에 도달하면 episode 종료를 알립니다.
        #     done = True
        # else:
        #     # 아직 목표 근처에 도달하지 않았을 때는 목표에 가까워질수록 리워드가 커지게 설정
        #     reward = -l**2
        #
        # x0, y0 = self.robot_tf.get_translation()
        # if abs(x0) > self.env_boundary:
        #     print("Robot이 Boundary를 벗어남.")
        #     done = True
        #     reward = -100
        # elif abs(y0) > self.env_boundary:
        #     print("Robot이 Boundary를 벗어남.")
        #     done = True
        #     reward = -100
        #
        # return reward, done
        # 해당 step의 reward를 계산합니다.
        done = False

        l = np.linalg.norm(
            self.target_tf.get_translation() - self.link2_tf_global.get_translation()
        )
        # 목표점을 반지름 sqrt(self.tol)인 원으로 설정
        if l < self.tol:
            reward = 1.
            # 목표 근처에 도달하면 episode 종료를 알립니다.
            done = True
        else:
            # 아직 목표 근처에 도달하지 않았을 때는 목표에 가까워질수록 리워드가 커지게 설정
            reward = -l**2

        # 시간 고려
        if self.timeout <= self.t:
            reward = -100
            done = True
            print("시간 초과")

        time_panalty = -10 *( self.t/self.timeout)

        reward = reward + time_panalty

        x0, y0 = self.robot_tf.get_translation()
        if abs(x0) > self.env_boundary:
            print("Robot이 Boundary를 벗어남.")
            done = True
            reward = -100
        elif abs(y0) > self.env_boundary:
            print("Robot이 Boundary를 벗어남.")
            done = True
            reward = -100


        return reward, done


    
    def _get_state(self):
        # State(Observation)를 반환합니다.
        return np.concatenate(
            [
                tf.get_translation() for tf in [
                    self.robot_tf,
                    self.link1_tf_global,
                    self.link2_tf_global,
                    self.target_tf
                ]
            ]
        )
    
    
    def _get_state_r(self):
        # # State(Observation)를 반환합니다.
        # r_np = np.concatenate([
        #     Transformation(
        #         matrix=self.target_tf.get_transformation() - self.robot_tf.get_transformation()).get_translation()
        #     , Transformation(
        #         matrix=self.target_tf.get_transformation() - self.link2_tf_global.get_transformation()).get_translation()
        #     , Transformation(
        #         matrix=self.target_tf.get_transformation() - self.link1_tf_global.get_transformation()).get_translation()
        #     , self.target_tf.get_translation()
        #
        # ])
        # # State(Observation)를 반환합니다.
        # r_np = np.concatenate([
        #     self.target_tf.get_translation()
        #     ,Transformation(
        #         matrix=self.target_tf.get_transformation() - self.link2_tf_global.get_transformation()).get_translation()
        #     , Transformation(
        #         matrix=self.target_tf.get_transformation() - self.link1_tf_global.get_transformation()).get_translation()
        #     , Transformation(
        #         matrix=self.target_tf.get_transformation() - self.robot_tf.get_transformation()).get_translation()
        #         # State(Observation)를 반환합니다.
        #         r_np = np.concatenate([
        #             self.target_tf.get_translation()
        #             ,Transformation(
        #                 matrix=self.target_tf.get_transformation() - self.link2_tf_global.get_transformation()).get_translation()
        #             , Transformation(
        #                 matrix=self.target_tf.get_transformation() - self.link1_tf_global.get_transformation()).get_translation()
        #             , Transformation(
        #                 matrix=self.target_tf.get_transformation() - self.robot_tf.get_transformation()).get_translation()
        #
        #
        #         ])
        #
        # ])
        # State(Observation)를 반환합니다.
        # r_np = np.concatenate([
        #     self.target_tf.get_translation()
        #     ,Transformation(
        #         matrix=self.target_tf.get_transformation() - self.link2_tf_global.get_transformation()).get_translation()
        #     , Transformation(
        #                 matrix=self.target_tf.get_transformation() - self.link1_tf_global.get_transformation()).get_translation()
        #     , Transformation(
        #              matrix=self.target_tf.get_transformation() - self.robot_tf.get_transformation()).get_translation()
        #
        # ])
        #
        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # r_np = np.concatenate([link2_to_target, err1, err2, err3])


        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        # r_np = np.concatenate([link2_to_target, err1, err2, err3, [r_rad], [dis_tar_link] ])
        #

        #
        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        #
        # link2_rad_2 = np.degrees(np.arctan2(link2_to_target[0],link2_to_target[1]))
        # j1_rad = self.joint1_tf.euler_angle()
        # j2_rad = self.joint2_tf.euler_angle()
        # link2_rad = tar_rad - j1_rad - j2_rad
        # dis_link2 = np.linalg.norm(self.target_tf.get_translation() - self.link2_tf_global.get_translation())
        # r_np = np.concatenate([link2_to_target, err1, err2, err3, [r_rad], [dis_tar_link], [link2_rad], [dis_link2]])


        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        #
        # link2_rad_2 = np.degrees(np.arctan2(link2_to_target[0],link2_to_target[1]))
        # j1_rad = self.joint1_tf.euler_angle()
        # j2_rad = self.joint2_tf.euler_angle()
        # link2_rad = j1_rad - j2_rad - tar_rad
        # dis_link2 = np.linalg.norm(self.target_tf.get_translation() - self.link2_tf_global.get_translation())
        # r_np = np.concatenate([link2_to_target, err1, err2, err3, [r_rad], [dis_tar_link]])

        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        # r_np = np.concatenate([link2_to_target, err1, err2, err3, [r_rad], [dis_tar_link] ])
        #

        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        #
        #
        # e_rad3 = np.arctan2(err3[1], err3[0])
        # e_rad4 = np.arctan2(err2[1], err2[0])
        # e_rad5 = np.arctan2(err1[1], err1[0])
        #
        # r_np = np.concatenate([link2_to_target, err1, err2, err3, [r_rad], [dis_tar_link]
        #                           ,[e_rad3], [e_rad4], [e_rad5] ])

        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        #
        #
        # e_rad3 = np.arctan2(err3[1], err3[0])
        # e_rad4 = np.arctan2(err2[1], err2[0])
        # e_rad5 = np.arctan2(err1[1], err1[0])
        #
        # r_np = np.concatenate([[e_rad3], [e_rad4], [e_rad5] ])
        #
        # self.buffer_detail.append(
        #     dict(
        #         link2_to_target = link2_to_target, # x,y
        #         err1 = err1, # x, y
        #         err2 = err2, # x, y
        #         err3 = err3, # x, y
        #         e_rad3=e_rad3,
        #         e_rad4=e_rad4,
        #         e_rad5=e_rad5
        #     )
        # )

        # link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        # err1 = self.link2_tf * link2_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self.joint1_tf * err2
        # tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        # rot_rad = self.robot_tf.euler_angle()
        # r_rad = rot_rad-tar_rad
        # dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())
        #
        #
        # e_rad3 = np.arctan2(err3[1], err3[0])
        # e_rad4 = np.arctan2(err2[1], err2[0])
        # e_rad5 = np.arctan2(err1[1], err1[0])
        #
        # r_np = np.concatenate([link2_to_target,err1,err2,err3])
        #
        # self.buffer_detail.append(
        #     dict(
        #         link2_to_target = link2_to_target, # x,y
        #         err1 = err1, # x, y
        #         err2 = err2, # x, y
        #         err3 = err3, # x, y
        #         e_rad3=e_rad3,
        #         e_rad4=e_rad4,
        #         e_rad5=e_rad5
        #     )
        # )
        # robot_to_target = self.robot_tf.inv() * self.target_tf.get_translation()
        # err1 = self.joint1_tf * robot_to_target
        # err2 = self.link1_tf * self.joint2_tf * err1
        # err3 = self. link2_tf * err2
        #
        # e_rad3 = np.arctan2(err3[1], err3[0])
        # e_rad4 = np.arctan2(err2[1], err2[0])
        # e_rad5 = np.arctan2(err1[1], err1[0])
        #
        #
        # r_np = np.concatenate([robot_to_target,err1,err2,err3])
        #
        # self.buffer_detail.append(
        #     dict(
        #         link2_to_target = robot_to_target, # x,y
        #         err1 = err1, # x, y
        #         err2 = err2, # x, y
        #         err3 = err3, # x, y
        #         e_rad3=e_rad3,
        #         e_rad4=e_rad4,
        #         e_rad5=e_rad5
        #     )
        # )

        link2_to_target = self.link2_tf_global.inv() * self.target_tf.get_translation()
        err1 = self.link2_tf * link2_to_target
        err2 = self.link1_tf * self.joint2_tf * err1
        err3 = self.joint1_tf * err2
        tar_rad = np.arctan2(self.target_tf.y(), self.target_tf.x())
        rot_rad = self.robot_tf.euler_angle()
        r_rad = rot_rad-tar_rad
        dis_tar_link = np.linalg.norm(self.target_tf.get_translation()-self.robot_tf.get_translation())


        e_rad3 = np.arctan2(err3[1], err3[0])
        e_rad4 = np.arctan2(err2[1], err2[0])
        e_rad5 = np.arctan2(err1[1], err1[0])

        r_np = np.concatenate([link2_to_target,err1,err2,err3])

        self.buffer_detail.append(
            dict(
                link2_to_target = link2_to_target, # x,y
                err1 = err1, # x, y
                err2 = err2, # x, y
                err3 = err3, # x, y
                e_rad3=e_rad3,
                e_rad4=e_rad4,
                e_rad5=e_rad5
            )
        )

        return r_np



    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def render(self):
        # Episode 동안의 로봇암 trajectory plot
        buffer = np.array(self.buffer)
        
        # set up figure and animation
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                            xlim=(-self.env_boundary, self.env_boundary), ylim=(-self.env_boundary, self.env_boundary))
        ax.grid()

        robot, = ax.plot([], [], 'g', lw=2)
        link1, = ax.plot([], [], 'ko-', lw=2)
        link2, = ax.plot([], [], 'k', lw=2)
        gripper, = ax.plot([], [], 'k', lw=1)
        target, = ax.plot([], [], 'bo', ms=6)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        robot_geom = np.array(
            [
                [0.3, -0.2, -0.2, 0.3],
                [  0,  0.2, -0.2,   0],
                [  1,    1,    1,   1]
            ]
        )
        link2_geom = np.array(
            [
                [-self.link2_len, -0.1],
                [              0,    0],
                [              1,    1]
            ]
        )
        gripper_geom = np.array(
            [
                [0.1, -0.1, -0.1,  0.1],
                [0.1,  0.1, -0.1, -0.1],
                [   1,    1,   1,    1]
            ]
        )

        def init():
            """initialize animation"""
            robot.set_data([], [])
            link1.set_data([], [])
            link2.set_data([], [])
            gripper.set_data([], [])
            target.set_data([], [])
            time_text.set_text('')
            reward_text.set_text('')
            return robot, link1, link2, gripper, target, time_text, reward_text

        def animate(i):
            """perform animation step"""
            robot_points = buffer[i]['robot'] * robot_geom
            link2_points = buffer[i]['link2'] * link2_geom
            gripper_points = buffer[i]['link2'] * gripper_geom

            robot.set_data((robot_points[0, :], robot_points[1, :]))
            link1.set_data((
                [buffer[i]['robot'].x(), buffer[i]['link1'].x()],
                [buffer[i]['robot'].y(), buffer[i]['link1'].y()]
            ))
            link2.set_data((link2_points[0, :], link2_points[1, :]))
            gripper.set_data((gripper_points[0, :], gripper_points[1, :]))
            target.set_data([buffer[i]['target'].x(), buffer[i]['target'].y()])
            time_text.set_text('time = %.1f' % buffer[i]['time'])
            reward_text.set_text('reward = %.3f' % buffer[i]['reward'])
            return robot, link1, link2, gripper, target, time_text, reward_text

        interval = self.dt * 1000
        ani = animation.FuncAnimation(fig, animate, frames=len(self.buffer),
                                        interval=interval, blit=True, init_func=init)

        plt.show()
        plt.pause(50)  # Note this correction
        plt.close()

    def render2(self):
        # Episode 동안의 로봇암 trajectory plot
        buffer = np.array(self.buffer)
        buffer_detail = np.array(self.buffer_detail)

        # set up figure and animation
        fig = plt.figure()
        # ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
        #                      xlim=(-self.env_boundary, self.env_boundary), ylim=(-self.env_boundary, self.env_boundary))

        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-15, 15), ylim=(-15, 15))

        ax.grid()

        robot, = ax.plot([], [], 'g', lw=2)
        link1, = ax.plot([], [], 'ko-', lw=2)
        link2, = ax.plot([], [], 'k', lw=2)
        gripper, = ax.plot([], [], 'k', lw=1)
        target, = ax.plot([], [], 'bo', ms=6)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        # details
        d1 = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        d2 = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        d3 = ax.text(0.02, 0.75, '', transform=ax.transAxes)
        d4 = ax.text(0.02, 0.70, '', transform=ax.transAxes)
        d5 = ax.text(0.02, 0.65, '', transform=ax.transAxes)
        d6 = ax.text(0.02, 0.60, '', transform=ax.transAxes)
        d7 = ax.text(0.02, 0.55, '', transform=ax.transAxes)
        d8 = ax.text(0.02, 0.50, '', transform=ax.transAxes)
        d9 = ax.text(0.02, 0.35, '', transform=ax.transAxes)
        d10 = ax.text(0.02, 0.20, '', transform=ax.transAxes)

        inv_link3, = ax.plot([], [], 'ro', ms=6)
        inv_link2, = ax.plot([], [], 'bo', ms=6)
        inv_link1, = ax.plot([], [], 'go', ms=6)
        inv_link0, = ax.plot([], [], 'ko', ms=6)


        robot_geom = np.array(
            [
                [0.3, -0.2, -0.2, 0.3],
                [0, 0.2, -0.2, 0],
                [1, 1, 1, 1]
            ]
        )
        link2_geom = np.array(
            [
                [-self.link2_len, -0.1],
                [0, 0],
                [1, 1]
            ]
        )
        gripper_geom = np.array(
            [
                [0.1, -0.1, -0.1, 0.1],
                [0.1, 0.1, -0.1, -0.1],
                [1, 1, 1, 1]
            ]
        )

        def init():
            """initialize animation"""
            robot.set_data([], [])
            link1.set_data([], [])
            link2.set_data([], [])
            gripper.set_data([], [])
            target.set_data([], [])
            time_text.set_text('')
            reward_text.set_text('')
            # detail
            d1.set_text('')
            d2.set_text('')
            d3.set_text('')
            d4.set_text('')
            d5.set_text('')
            d6.set_text('')
            d7.set_text('')
            d8.set_text('')
            d9.set_text('')

            inv_link3.set_data([], [])
            inv_link2.set_data([], [])
            inv_link1.set_data([], [])
            inv_link0.set_data([], [])

            return robot, link1, link2, gripper, target, time_text, reward_text, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, inv_link3, inv_link2, inv_link1, inv_link0

        def animate(i):
            """perform animation step"""
            robot_points = buffer[i]['robot'] * robot_geom
            link2_points = buffer[i]['link2'] * link2_geom
            gripper_points = buffer[i]['link2'] * gripper_geom

            robot.set_data((robot_points[0, :], robot_points[1, :]))
            link1.set_data((
                [buffer[i]['robot'].x(), buffer[i]['link1'].x()],
                [buffer[i]['robot'].y(), buffer[i]['link1'].y()]
            ))
            link2.set_data((link2_points[0, :], link2_points[1, :]))
            gripper.set_data((gripper_points[0, :], gripper_points[1, :]))
            target.set_data([buffer[i]['target'].x(), buffer[i]['target'].y()])
            time_text.set_text('time = %.1f' % buffer[i]['time'])
            reward_text.set_text('reward = %.3f' % buffer[i]['reward'])
            #details
            d1.set_text('link2_to_target = %s' % buffer_detail[i]['link2_to_target'])
            d2.set_text('err1 = %s' % buffer_detail[i]['err1'])
            d3.set_text('err2 = %s' % buffer_detail[i]['err2'])
            d4.set_text('err3 = %s' % buffer_detail[i]['err3'])
            d5.set_text('e_rad3 = %s' % buffer_detail[i]['e_rad3'])
            d6.set_text('e_rad4 = %s' % buffer_detail[i]['e_rad4'])
            d7.set_text('e_rad5 = %s' % buffer_detail[i]['e_rad5'])
            d8.set_text('target = %s' % buffer[i]['target'].get_translation())
            d9.set_text('link2 = %s' % buffer[i]['link2'].get_transformation())
            d10.set_text('link2_inv = %s' % buffer[i]['link2'].inv().get_transformation())

            inv_link3.set_data([buffer_detail[i]['link2_to_target'][0], buffer_detail[i]['link2_to_target'][1]])
            inv_link2.set_data([buffer_detail[i]['err1'][0], buffer_detail[i]['err1'][1]])
            inv_link1.set_data([buffer_detail[i]['err2'][0], buffer_detail[i]['err2'][1]])
            inv_link0.set_data([buffer_detail[i]['err3'][0], buffer_detail[i]['err3'][1]])
            #
            # link2_to_target = link2_to_target,  # x,y
            # err1 = err1,  # x, y
            # err2 = err2,  # x, y
            # err3 = err3,  # x, y
            # e_rad3 = e_rad3,
            # e_rad4 = e_rad4,
            # e_rad5 = e_rad5

            return robot, link1, link2, gripper, target, time_text, reward_text, inv_link3, inv_link2, inv_link1, inv_link0 #d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, inv_link2

        # link2_mat1 =  buffer[0]['link2'].get_transformation()
        # link2_mat2 = buffer[0]['link2'].inv().get_transformation()
        link_coord1 = (buffer_detail[0]['link2_to_target'][0], buffer_detail[0]['link2_to_target'][1])
        link_coord2 = (buffer_detail[1]['link2_to_target'][0], buffer_detail[1]['link2_to_target'][1])
        print("get mat")
        #interval = self.dt * 1000
        dd_max = 0
        dd_min = 0
        for row in buffer_detail:
            dd = row['link2_to_target']

            if dd.max() > dd_max:
                dd_max = dd.max()
            if dd.min() < dd_min:
                dd_min = dd.min()

        print(dd_max)
        print(dd_min)
        interval = 100
        ani = animation.FuncAnimation(fig, animate, frames=len(self.buffer),
                                     interval=interval, blit=True, init_func=init)
        # ani = animation.FuncAnimation(fig, animate, frames=10,
        #                               interval=interval, blit=True, init_func=init)

        plt.show()
        plt.pause(50)  # Note this correction
        plt.close()




def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''

    # 환경 초기화
    env.reset()

    # 10초 동안의 움직임을 관찰
    for t in np.arange(0, 10, env.dt):
        # 강화학습이 아닌 위에서 계산한 값을 이용하여 목표 각도에 가까워지도록 피드백 제어

        # position error를 이용해 control input 계산
        link2_to_target = env.link2_tf_global.inv() * env.target_tf.get_translation()
        err1 = env.link2_tf * link2_to_target
        err2 = env.link1_tf * env.joint2_tf * err1
        err3 = env.joint1_tf * err2
        action = [
            np.linalg.norm(err3),
            np.arctan2(err3[1], err3[0]),
            np.arctan2(err2[1], err2[0]),
            np.arctan2(err1[1], err1[0])
        ]

        # Environment의 step 함수를 호출하고, 
        # 변화된 state(observation)과 reward, episode 종료여부, 기타 정보를 가져옴
        next_state, reward, done, info = env.step(action)

        # episode 종료
        if done:
            break

    # Episode 동안의 로봇암 trajectory plot
    #env.render()
    env.render2()

if __name__=='__main__':

    test(Manipulator2D(tol=0.01))
