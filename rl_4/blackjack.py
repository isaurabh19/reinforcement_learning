# %%
import gym
from tqdm import trange
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

hit = 1
stick = 0
gamma = 1.0
num_episodes = 10000


class Policy:

    def get_action(self, state):
        if state[0] < 20:
            return hit
        return stick





def generate_episode(env, policy):
    episode = []
    state = env.reset()
    while True:
        action = policy.get_action(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def policy_evaluation(env, policy):
    V = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    for n in trange(num_episodes + 1):
        episode = generate_episode(env, policy)
        states, actions, rewards = zip(*episode)
        G = 0
        first_occurence = {state[0]: i for i, state in enumerate(states)}
        for i in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[i]
            if first_occurence[states[i][0]] == i:  # first-visit MC
                vs = V[states[i]][actions[i]]
                N[states[i]][actions[i]] += 1.0
                V[states[i]][actions[i]] += (G - vs) / N[states[i]][actions[i]]
    return V


def plot_v(V):
    def get_V(x, y, usable_ace):
        if (x, y, usable_ace) in V:
            return V[(x, y, usable_ace)]
        return 0

    def single_fig(filename, usable_ace, ax):
        z = np.array([get_V(x, y, usable_ace) for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
        ax.plot_surface(X, Y, Z=z)
        ax.view_init(elev=60)
        ax.set_xlabel('PLayer current sum')
        ax.set_ylabel('Dealer show card')
        ax.set_zlabel('Value')
        plt.savefig(filename)
        plt.show()

    X, Y = np.meshgrid(np.arange(1, 11), np.arange(11, 21))
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.set_title('Usable Ace')
    single_fig('usable_ace.png', True, ax1)
    ax2 = plt.axes(projection='3d')
    ax2.set_title('No Usable ace')
    single_fig('not_usable_ace.png', False, ax2)

blackjack = gym.make('Blackjack-v0')
stick_policy = Policy()
V = policy_evaluation(blackjack, stick_policy)

plot_v(V)
