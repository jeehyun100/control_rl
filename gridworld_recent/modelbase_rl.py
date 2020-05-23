import numpy as np
from grid_world import GridWorld
from settings import *
from matplotlib import pyplot as plt
import pandas as pd


# Agent of player

class Agent:

    def __init__(self):
        self.actions = ["up", "down", "left", "right"]
        self.num_actions = len(self.actions)
        self.grid_world = GridWorld()

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0
        self.state_indices = {}
        k = 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_indices[(i, j)] = k  # set initial value to 0
                k += 1

        self.num_states = len(self.state_values)
        self.state_values_vec = np.zeros((self.num_states))
        self.rewards = np.zeros((self.num_states))

        self.state_transition_prob = np.zeros((self.num_states, self.num_actions, self.num_states))
        #self.get_observation_by_random(20000)

        self.discount = 0.99
    def count_base_reward(self, state):
        """
        To count state, reward in observation sequence

        """
        count_s = [exp for exp in self.observation if exp[0] == state]
        count_s_r = [exp[2] for exp in self.observation if exp[0] == state]
        try:
            r = sum(count_s_r)/len(count_s)
        except Exception as e:
            r = 0
        print(sum(count_s_r))
        return r

    def count_base_prob(self, state, action, nxtState):
        """
        To count state, action nxtState in observation sequence

        """
        # m -1 loop
        count_s_a = [ exp for exp in self.observation[:len(self.observation)-1] if exp[0] == state and exp[1] == action]
        count_s_a_nxt = [ (i,exp, self.observation[i+1][0]) for i,exp in enumerate(self.observation[:len(self.observation)-1]) if exp[0] == state and exp[1] == action and self.observation[i+1][0] == nxtState ]
        if action == 'down' and nxtState == (1,0):
            print("down and (1,0)")
        try:
            p = len(count_s_a_nxt)/len(count_s_a)
        except Exception as e:
            p = 0
        print(count_s_a)
        return p


    def get_observation_by_random(self, num_of_samples):
        """
        Get observation sequecne with random moving agent

        """
        observation = [] #Observe experience s1, r1, a1, s2, r2, a2, sm, rm, am
        for _ in range(num_of_samples):
            s_a_r = list()
            r_action = self.choose_random_action()
            s_a_r.append(self.grid_world.state)
            s_a_r.append(r_action)
            reward = self.grid_world.giveReward()
            s_a_r.append(reward)

            print("current position {} action {}".format(self.grid_world.state, r_action))
            # by taking the action, it reaches the next state
            self.grid_world = self.takeAction(r_action)
            #
            if(reward == 1):
                print("reward 1")
            observation.append(s_a_r)

        self.observation = observation
        self.make_transition_matrix()

    def make_transition_matrix(self):
        """
        To calulate Transition matrix p(s'|s,a)

        """
        for state in self.state_values.keys():
            self.rewards[self.state_indices[state]] = self.count_base_reward(state)
            for action in self.actions:
                for nxtState in self.state_values.keys():
                    # p, r
                    p = self.count_base_prob(state, action, nxtState)
                    self.state_transition_prob[
                        self.state_indices[state], self.actions.index(action), self.state_indices[nxtState]] += p

    def takeAction(self, action):
        position = self.grid_world.nxtPosition(action)
        # update GridWorld
        return GridWorld(state=position)

    def choose_random_action(self):
        # choose_random_action
        action = np.random.choice(self.actions)
        return action


    def update(self):

        # Compute the action values $Q(s,a)$

        _action_values = np.repeat(self.rewards, 4).reshape((self.num_states, 4)) + self.discount * np.sum(
            self.state_transition_prob * self.state_values_vec, axis=2, keepdims=False
        )

        # Evaluate the deterministic policy $\pi(s)$
        self.policy = np.argmax(_action_values, axis=1)

        # Compute the values $V(s)$
        values = np.max(_action_values, axis=1, keepdims=False)

        # Compute the value difference $|\V_{k}-V_{k+1}|\$ for check the convergence
        diff = np.linalg.norm(self.state_values_vec[:] - values[:])

        # Update the current value estimate
        self.state_values_vec = values

        return diff, values

    def nxtPosition(self, state, action):
        if action == "up":
            nxtState = (state[0] - 1, state[1])
        elif action == "down":
            nxtState = (state[0] + 1, state[1])
        elif action == "left":
            nxtState = (state[0], state[1] - 1)
        else:
            nxtState = (state[0], state[1] + 1)
        if (nxtState[0] >= 0) and (nxtState[0] <= 2):
            if (nxtState[1] >= 0) and (nxtState[1] <= 3):
                if nxtState != (1, 1):
                    return nxtState
        return state

    # def giveReward(self, state):
    #     if state == WIN_STATE:
    #         return 1
    #     elif state == LOSE_STATE:
    #         return -1
    #     else:
    #         return 0

    def fit(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False):

        if logging:
            history = []

        # Value iteration loop
        for _iter in range(1, int(max_iteration + 1)):

            # Update the value estimate
            diff, values = self.update()
            if logging:
                history.append(diff)
            if verbose:
                print('Iteration: {0}\tValue difference: {1}'.format(_iter, diff))

            # Check the convergence
            if diff < tolerance:
                if verbose:
                    print('Converged at iteration {0}.'.format(_iter))
                break

        if logging:
            return diff, history, values, self.policy
        else:
            return diff

def draw_plot(histories, max_n=100):
    plt.figure(figsize=(16, 10))

    for name, history, scalar_value in histories:
    # val = plt.plot(history.epoch, history['val_' + key],
    #                '--', label=name.title() + ' Val')
        plt.plot(range(len(history)), history, label="# samples / Values :" + name + " / " + '{:.2f}'.format(scalar_value))
    #
    plt.xlabel('Epochs')
    plt.ylabel('Diff')
    plt.legend()
    #
    plt.xlim([0, max_n])
    plt.ylim([0, 4])
    # plt.title(name)
    plt.savefig("model_based_iteration.png")
    plt.show()

if __name__ == "__main__":

    # # of agent random moving
    observation_epoch = [40, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 2000,3000]#,50]#,100,150,200]
    row_list = list()
    row_list_p = list()
    for i in observation_epoch:
        ag = Agent()
        #col_list = list()
        col_list_p = list()
        col_list_p.append(i)
        ag.get_observation_by_random(i)
        diff, history, values, policy = ag.fit(max_iteration=10000, tolerance=0.01, verbose=True, logging = True)
        value_scalar = np.linalg.norm(values[:])
        row_list.append((str(i), history, value_scalar))
        col_list_p.extend(policy)

        #row_list.append(col_list)
        row_list_p.append(col_list_p)
    draw_plot(row_list,300)
    df_hist = pd.DataFrame(row_list)
    df_policy = pd.DataFrame(row_list_p)
    # save result csv
    df_policy.to_csv("modelbase_rl.csv")
    print(policy)