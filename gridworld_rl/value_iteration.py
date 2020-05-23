import numpy as np
from grid_world import GridWorld
from settings import *
from matplotlib import pyplot as plt

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
        for state in self.state_values.keys():
            self.rewards[self.state_indices[state]] = self.giveReward(state)
            for action in self.actions:
                if action == "up":
                    action_probs = zip(["up", "left", "right"], [0.8, 0.1, 0.1])
                if action == "down":
                    action_probs = zip(["down", "left", "right"], [0.8, 0.1, 0.1])
                if action == "left":
                    action_probs = zip(["left", "up", "down"], [0.8, 0.1, 0.1])
                if action == "right":
                    action_probs = zip(["right", "up", "down"], [0.8, 0.1, 0.1])
                for a, p in action_probs:
                    nxtState = self.nxtPosition(state, action)
                    self.state_transition_prob[self.state_indices[state], self.actions.index(a), self.state_indices[nxtState]] += p
        
        self.discount = 0.9
    
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

        return diff


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

    def giveReward(self, state):
        if state == WIN_STATE:
            return 1
        elif state == LOSE_STATE:
            return -1
        else:
            return 0
    
    def fit(self, max_iteration=1e3, tolerance=1e-3, verbose=False, logging=False):
        
        if logging:
            history=[]

        # Value iteration loop
        for _iter in range(1, int(max_iteration+1)):

            # Update the value estimate
            diff = self.update()
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
            return diff, history
        else:
            return diff



if __name__ == "__main__":
    ag = Agent()
    ag.fit(max_iteration=10000, tolerance=0.01, verbose=True)

