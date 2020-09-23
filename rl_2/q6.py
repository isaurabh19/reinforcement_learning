import numpy as np
import random
import time
from matplotlib import pyplot as plt

epsilon = 0.1
testbed = np.load('final_test.npy')


def sample_averaging(reward_registry):
    q_stars = np.zeros(10)
    n = np.zeros(10)
    rewards_obtained = []
    optimal_action_list = []
    for step in range(100):
        max_a = np.argmax(q_stars)
        action = random.choices([max_a, np.random.randint(0, 10)], [1 - epsilon, epsilon])[0]
        reward = reward_registry[step, action]
        n[action] += 1
        q_stars[action] += q_stars[action] + (1 / n[action]) * (reward - q_stars[action])
        rewards_obtained.append(reward)
        optimal_action = np.argmax(reward_registry[step])
        optimal_action_list.append(optimal_action == action)
    return rewards_obtained, optimal_action_list


start = time.time()
results = [sample_averaging(testbed[trial]) for trial in range(20)]
print(f'Time {time.time() - start}')
results = np.array(results)
rewards = results[:, 0, :]
optimal_actions = results[:, 1, :]
average_rewards = np.mean(rewards, axis=0)
std_error = (np.std(rewards, axis=0) / np.sqrt(rewards.shape[0])) * 1.96
upper_bound = np.max(testbed, axis=2)
average_upper_bound = np.mean(upper_bound, axis=0)
total_optimal_actions = np.sum(optimal_actions == 1, axis=0)
fractional_optimal_actions = np.divide(total_optimal_actions, optimal_actions.shape[0])

fig, ax = plt.subplots()
ax.plot(np.arange(rewards.shape[1]), average_rewards)
ax.fill_between(np.arange(rewards.shape[1]), average_rewards - std_error, average_rewards + std_error, alpha=0.2)
plt.show()
fig.savefig('q6.png')
