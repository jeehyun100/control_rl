import numpy as np
from grid_world import GridWorld
from settings import *
from matplotlib import pyplot as plt
import pandas as pd

class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.grid_world = GridWorld()
        self.isEnd = self.grid_world.isEnd
        self.lr = 0.2
        self.exp_rate = -1#-1
        self.decay_gamma = 0.99

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict
        self.state_values_vec = np.zeros((len(self.Q_values)))
        self.policy_list = list()

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -np.inf
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
            print("random")
        else:
            # greedy action
            for a in self.actions:
                current_position = self.grid_world.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.grid_world.state, action))
        return action

    def takeAction(self, action):
        position = self.grid_world.nxtPosition(action)
        # update GridWorld
        return GridWorld(state=position)

    def reset(self):
        self.states = []
        self.grid_world = GridWorld()
        self.isEnd = self.grid_world.isEnd

    def play(self, rounds=10):
        i = 0
        histories = list()
        before_q_value_numpy = 0
        data_sample = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.grid_world.isEnd:
                # back propagate
                reward = self.grid_world.giveReward()
                for a in self.actions:
                    self.Q_values[self.grid_world.state][a] = reward
                #print("Game End Reward", reward)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                #if i % 100 == 0:
                self.policy_list.append(self.make_policy_table_from_q())
                self.reset()
                i += 1
                aa = [[k for e, k in e.items()] for e in [v for k, v in self.Q_values.items()]]
                ab = np.array(aa, dtype=float)
                q_value_numpy  = np.max(ab, axis=1, keepdims=False)
                diff = np.linalg.norm(self.state_values_vec[:] - q_value_numpy[:])
                self.state_values_vec = q_value_numpy
                #print("iter {0}".format(i))
                #print("diff {0}".format(diff))
                #print("qvalue {0}".format(np.linalg.norm(q_value_numpy[:])))
                print("{0}".format(data_sample))
                value_scalar= np.linalg.norm(q_value_numpy[:])
                histories.append((value_scalar, diff, int(data_sample)))
                data_sample = 0

            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.grid_world.state), action]) #s, a
                #print("current position {} action {}".format(self.grid_world.state, action))
                # by taking the action, it reaches the next state
                self.grid_world = self.takeAction(action)
                # mark is end
                self.grid_world.isEndFunc()
                #print("nxt state", self.grid_world.state)
                #print("---------------------")
                self.isEnd = self.grid_world.isEnd
                data_sample += 1
        return histories, self.policy_list

    def make_policy_table_from_q(self):
        convert_list = list()
        for k, v in self.Q_values.items():
            convert_list.append(np.argmax(np.array([val for (key, val) in v.items()], float)))
        return convert_list


def draw_plot(history, max_n=100, exp_rate = 0.3):
    plt.figure(figsize=(16, 10))

    #for history, diff in histories:
    # val = plt.plot(history.epoch, history['val_' + key],
    #                '--', label=name.title() + ' Val')
    label_text = "q-learning non epsilon-greedy" if exp_rate == -1 else "q-learning epsilon-greedy"
    plt.plot(range(len(history)), np.array(history)[:,0], label="q-learning q value")
    plt.plot(range(len(history)), np.array(history)[:,1], label="q-learning diff")
    plt.plot(range(len(history)), np.array(history)[:,2], label=label_text)
    #
    plt.xlabel('Epochs')
    #plt.ylabel('Diff')
    plt.ylabel('Diff/Qvalue/#Sample')
    plt.legend()
    #
    plt.xlim([0, max_n])
    # plt.title(name)
    plt.savefig(label_text+".png")
    plt.show()



if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    histories, policy_list = ag.play(50)
    draw_plot(histories, 60, ag.exp_rate)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    df_p = pd.DataFrame(policy_list)
    df_p.to_csv("q-learning.csv")