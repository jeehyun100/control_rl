import gym
from gym import core, spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

def sum_angle(a, b):
    c = a + b
    if c >= np.pi:
        c -= 2 * np.pi
    elif c < -np.pi:
        c += 2 * np.pi
    return c

class Manipulator2D(gym.Env):
    
    def __init__(self, arm1=1, arm2=1, dt=0.01, tol=0.1):
        # Observation space를 구성하는 state의 최대, 최소를 지정한다.
        self.obs_high = np.array([1, 1, 2, 2, 2, 2]) # x1, y1, x2, y2, xd, yd
        self.obs_low = -self.obs_high

        # Action space를 구성하는 action의 최대, 최소를 지정한다.
        self.action_high = np.array([np.pi, np.pi])
        self.action_low = -self.action_high

        # GYM environment에서 요구하는 변수로, 실제 observation space와 action space를 여기에서 구성한다.
        self.observation_space = spaces.Box(low = self.obs_low, high = self.obs_high, dtype = np.float32)
        self.action_space = spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)

        # 로봇암의 요소를 결정하는 변수
        self.arm1 = arm1 # 로봇팔 길이
        self.arm2 = arm2
        self.dt = dt # Timestep
        self.tol = tol # 목표까지 거리

        # 변수를 초기화한다.
        self.reset()
        
        # 학습 환경에서 사용할 난수 생성에 필요한 seed를 지정한다.
        self.seed()

        
    def step(self, action):
        # 강화학습이 만들어낸 action을 위에서 지정한 최대, 최소 action으로 클리핑한다.
        action = np.clip(action, self.action_low, self.action_high)

        # Action으로부터 로봇암 kinematics를 계산하는 부분
        # 여기에서 action은 각 로봇팔1이 x축과 이루는 각의 변화, 로봇팔1과 로봇팔2이 이루는 각의 변화를 말함
        self.theta1 += action[0] * self.dt
        self.theta21 += action[1] * self.dt
        self.theta2 = sum_angle(self.theta21, self.theta1)

        # 로봇암 각속도로부터 현재 step에서의 로봇암 좌표 계산하기
        self.x1 = np.cos(self.theta1) * self.arm1
        self.y1 = np.sin(self.theta1) * self.arm1
        self.x2 = self.x1 + np.cos(self.theta2) * self.arm2
        self.y2 = self.y1 + np.sin(self.theta2) * self.arm2
        self.t += self.dt
        
        # Reward와 episode 종료 여부를 확인
        reward, done = self._get_reward(
            # Reward는 end-effector와 목표 지점까지의 거리
            np.linalg.norm([self.xd-self.x2, self.yd-self.y2])
        )

        # 기타 목적으로 사용할 데이터를 담아둠
        info = {'dist' :  np.linalg.norm([self.xd-self.x2, self.yd-self.y2])}

        # 시각화 목적으로 사용할 데이터를 self.buffer 에 저장
        self.buffer.append(
            [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.t,
                reward
            ]
        )

        # 일반적으로 Gym environment의 step function은 
        # State(observation), 현재 step에서의 reward, episode 종료 여부, 기타 정보로 구성되어있음
        return self._get_state(), reward, done, info


    def reset(self):
        # 매 episode가 시작될때 사용됨.
        # 사용 변수들 초기화
        self.theta1 = 0
        self.theta21 = 0
        self.theta2 = sum_angle(self.theta21, self.theta1)
        self.x1 = np.cos(self.theta1) * self.arm1
        self.y1 = np.sin(self.theta1) * self.arm1
        self.x2 = self.x1 + np.cos(self.theta2) * self.arm2    #state구하기
        self.y2 = self.y1 + np.sin(self.theta2) * self.arm2    #state구하기

        # rd = 1.7
        # alphad = 3 * np.pi / 4

        # 연습문제 : 목표 지점이 고정되어있지 않고 랜덤하게 1사분면에서 움직일때에 대하여 학습해보세요.
        # Tip : 문제를 더 쉽게 만들기 위해 State를 변경해야할 수 있습니다.
        rd = np.random.uniform(low=1.5, high=1.99)
        alphad = np.random.uniform(low=0, high=np.pi / 2)

        # 목표 지점 생성
        self.xd = rd * np.cos(alphad)
        self.yd = rd * np.sin(alphad)

        self.done = False
        self.t = 0
        self.buffer = []    # 시각화를 위한 버퍼. episode가 리셋될 때마다 초기화.

        # Step 함수와 다르게 reset함수는 초기 state 값 만을 반환합니다.
        return self._get_state()


    def _get_reward(self, l):
        # 해당 step의 reward를 계산합니다.
        done = False

        # 목표점을 반지름 sqrt(self.tol)인 원으로 설정
        if l < self.tol:
            reward = 1.
            # 목표 근처에 도달하면 episode 종료를 알립니다.
            done = True 
        else:
            # 아직 목표 근처에 도달하지 않았을 때는 목표에 가까워질수록 리워드가 커지게 설정
            reward = -l**2

        return reward, done

    
    def _get_state(self):
        # State(Observation)를 반환합니다.
        # 로봇암1의 끝쪽 좌표, 로봇암2의 끝쪽 좌표, 목표 지점의 좌표로 이루어져 있습니다.

        return np.array(
            [
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.xd,
                self.yd
            ]
        )
    

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    
    def render(self, plot_reward=False, last = False):
        # Episode 동안의 로봇암 trajectory plot
        buffer = np.array(self.buffer)
        plt.figure(1)
        plt.scatter([self.xd], [self.yd], c='r', marker='x', s=300)
        plt.plot(buffer[:, 0], buffer[:, 1], c='g')
        if last == True:
            plt.plot(
                buffer[-1, 2] - 0.1*np.cos(self.theta2),
                buffer[-1, 3] - 0.1*np.sin(self.theta2),
                c='b'
            )
        else:
            plt.plot(
                buffer[:, 2] - 0.1*np.cos(self.theta2),
                buffer[:, 3] - 0.1*np.sin(self.theta2),
                c='b'
            )
        plt.plot(
            [0, self.x1, self.x2 - 0.1*np.cos(self.theta2)],
            [0, self.y1, self.y2 - 0.1*np.sin(self.theta2)],
            marker='o',
            c='k'
        )
        plt.plot(
            [
                self.x2 + 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) - 0.1*np.sin(self.theta2),
                self.x2 - 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2),
                self.x2 + 0.1*np.cos(self.theta2) + 0.1*np.sin(self.theta2)
            ],
            [
                self.y2 + 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) + 0.1*np.cos(self.theta2),
                self.y2 - 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2),
                self.y2 + 0.1*np.sin(self.theta2) - 0.1*np.cos(self.theta2)
            ],
            c='k'
        )
        plt.axis('square')
        plt.title('Trajectory')
        if plot_reward:
            # Episode 동안 획득한 reward plot
            plt.figure(2)
            plt.plot(buffer[:, 4], buffer[:, 5])
            plt.title('Rewards')
        #plt.gcf().canvas.flush_events()

        plt.show(block=False)
        plt.pause(0.0001)  # Note this correction
        plt.clf()
        plt.cla()
        plt.close()



def test(env):
    '''
    Test script for the environment "Manipulator2D"
    '''

    # 환경 초기화
    env.reset()

    # 현재 로봇암의 위치로부터 목표지점까지 도달하려면 필요한 로봇암 팔의 각도를 계산
    c = (env.xd**2 + env.yd**2 - env.arm1**2 - env.arm2**2) / env.arm1 / 2
    s = np.sqrt(env.arm2**2 - c**2)
    theta21d = np.arctan2(s, c)
    theta1d = sum_angle(np.arctan2(env.yd, env.xd), - np.arctan2(s, env.arm1 + c))

    # 10초 동안의 움직임을 관찰
    for t in np.arange(0, 10, env.dt):
        # 강화학습이 아닌 위에서 계산한 값을 이용하여 목표 각도에 가까워지도록 피드백 제어
        action = [theta1d - env.theta1, theta21d - env.theta21]

        # Environment의 step 함수를 호출하고, 
        # 변화된 state(observation)과 reward, episode 종료여부, 기타 정보를 가져옴
        next_state, reward, done, info = env.step(action)

        # episode 종료
        if done:
            break

    # Episode 동안의 로봇암 trajectory plot
    env.render(plot_reward=True)


if __name__=='__main__':
    test(Manipulator2D(tol=0.01))
