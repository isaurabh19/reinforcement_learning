import numpy as np
import time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


def get_qstar():
    return np.random.normal(size=10)


def run(qstars):
    # TODO make it generic so that q7 can be done
    # for step in range(10000):
    #     tuple(np.random.normal(qstar) for qstar in qstars)
    return [tuple(np.random.normal(qstar) for qstar in qstars) for step in range(10000)]


# start = time.time()
# qstar_trials = np.array([get_qstar() for i in range(500)])
#
# testbed = Parallel(n_jobs=8)(delayed(run)(qstar) for qstar in qstar_trials)
# testbed = np.array(testbed)
# np.save('final_test.npy', testbed)
testbed = np.load('final_test.npy')
trial = testbed[0]
true_means = np.mean(trial, axis=0)
true_variances = np.var(trial, axis=0)

actions = np.arange(10)
sample_rewards = [[] for i in range(10)]
for step in range(10000):
    action = actions[step % 10]
    reward = trial[step, action]
    sample_rewards[action].append(reward)

plt.violinplot(sample_rewards, showmeans=True)
plt.show()
# sample_rewards = np.array(sample_rewards)
# sample_mean = np.mean(sample_rewards, axis=1)
# import glob
# arrs = [ np.load(file) for file in glob.glob('*.npy')]
# final_test = np.concatenate(arrs, axis=0)
# np.save('final_test.npy', final_test)
