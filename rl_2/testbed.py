import numpy as np
from matplotlib import pyplot as plt

trials = 2000
default_steps = 10000


def get_qstar():
    return np.random.normal(size=10)


def q4():
    testbed = []
    for _ in range(trials):
        qstar_trials = get_qstar()
        reward_registry_single_trials = np.array(
            [np.random.normal(qstar, 1.0, default_steps) for qstar in qstar_trials]).T
        testbed.append(reward_registry_single_trials)
    testbed = np.array(testbed)
    np.save('testbed.npy', testbed)
    print("Q4 done")


def q7():
    testbed = []
    qstar_bed = []
    for _ in range(trials):
        qstar_trials = np.zeros(10)
        qstar_registry = []
        reward_registry = []
        for step in range(default_steps):
            reward_registry.append(tuple(np.random.normal(qstar, 1) for qstar in qstar_trials))
            additions = np.random.normal(0, 0.01, 10)
            qstar_trials = np.add(qstar_trials, additions)
            qstar_registry.append(qstar_trials)
        qstar_bed.append(qstar_registry)
        testbed.append(reward_registry)
    qstar_bed = np.array(qstar_bed)
    testbed = np.array(testbed)
    np.save('nonstationary_testbed.npy', testbed)
    np.save('nonstationary_qstars.npy', qstar_bed)


def violin_plot(testbed):
    actions = np.arange(10)
    sample_rewards = [[] for i in range(10)]
    trial = testbed[0]
    for step in range(10000):
        action = actions[step % 10]
        reward = trial[step, action]
        sample_rewards[action].append(reward)

    plt.violinplot(sample_rewards, showmeans=True)
    plt.savefig('violin_plot.png')
    plt.show()


q4()
q7()
testbed = np.load('testbed.npy')
violin_plot(testbed)
