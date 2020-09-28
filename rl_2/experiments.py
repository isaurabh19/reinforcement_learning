import numpy as np
import random
import time
from matplotlib import pyplot as plt

testbed = np.load('testbed.npy')
modified_testbed = np.load('nonstationary_testbed.npy')
trials = 2000
default_steps = 1000
alpha = 0.1
c = 2
figsize = (25, 25)


# epsilon = 0.1


def constant_size_estimates(*args):
    current_q_estimate, _, reward = args
    return alpha * (reward - current_q_estimate)


def sample_averaging_estimates(*args):
    current_q_estimate, n, reward = args
    return (1 / n) * (reward - current_q_estimate)


def ucb(q_estimates, n_action, step):
    zeros_indx = np.where(n_action == 0)[0]
    if zeros_indx.size != 0:
        return zeros_indx[0]
    actions = q_estimates + c * np.sqrt(np.log(step) / n_action)
    return np.argmax(actions)


def bandit(reward_registry, epsilon=0.1, estimate_method=sample_averaging_estimates, initial_q=0.0,
           selection_method_epsilon=True, qstar_registry=np.zeros(1), epsilon_annealing=False):
    q_estimates = np.full(10, initial_q)
    n = np.zeros(10)
    rewards_obtained = []
    optimal_action_list = []
    for step in range(default_steps):
        if not selection_method_epsilon:
            action = ucb(q_estimates, n, step)
        else:
            max_a = np.argmax(q_estimates)
            action = random.choices([max_a, np.random.randint(0, 10)], [1 - epsilon, epsilon])[0]

        reward = reward_registry[step, action]
        n[action] += 1
        q_estimates[action] += estimate_method(q_estimates[action], n[action], reward)
        rewards_obtained.append(reward)
        optimal_action = np.argmax(np.mean(reward_registry, axis=0))
        if qstar_registry.any():
            optimal_action = np.argmax(qstar_registry[step])
        optimal_action_list.append(optimal_action == action)

        if epsilon_annealing and step % 1000 == 0:
            epsilon /= 2
    return rewards_obtained, optimal_action_list


def core(epsilon, testset=testbed, estimate_method=sample_averaging_estimates, initial_q=0.0,
         selection_method_epsilon=True, qstar_registry=np.zeros(1), epsilon_annealing=False):
    start = time.time()
    print(f'Started at {time.ctime()}')
    if qstar_registry.any():
        results = [bandit(testset[trial], epsilon, estimate_method, initial_q, selection_method_epsilon,
                          qstar_registry[trial]) for trial in range(trials)]
    else:
        results = [bandit(testset[trial], epsilon, estimate_method, initial_q, selection_method_epsilon, epsilon_annealing=epsilon_annealing)
                   for trial in range(trials)]
    print(f'Time {time.time() - start}')
    results = np.array(results)
    rewards = results[:, 0, :]
    optimal_actions = results[:, 1, :]
    average_rewards = np.mean(rewards, axis=0)
    std_error = (np.std(rewards, axis=0) / np.sqrt(trials)) * 1.96
    fractional_optimal_actions = np.mean(optimal_actions, axis=0) * 100
    return average_rewards, std_error, fractional_optimal_actions


def plot_upperbound(ax, testset=testbed):
    upper_bound = np.max(np.mean(testset, axis=1), axis=1)
    average_upper_bound = np.mean(upper_bound)
    ax.plot(np.arange(default_steps), [average_upper_bound] * default_steps, label='Avg Upper bound')
    ax.legend()


def plot(average_rewards, std_error, fractional_optimal_actions, ax1, ax2, label1):
    line1, = ax1.plot(np.arange(default_steps), average_rewards, label=label1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average rewards')
    ax1.fill_between(np.arange(default_steps), average_rewards - std_error, average_rewards + std_error, alpha=0.2)
    ax1.legend()
    line2, = ax2.plot(np.arange(default_steps), fractional_optimal_actions, label=label1)
    ax2.set_ylabel('Optimal Action %')
    ax2.set_xlabel('Steps')
    ax2.legend()
    return line1, line2


def q6():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    for epsilon in [0.1, 0.01, 0.0]:
        average_rewards, std_error, fractional_optimal_actions = core(epsilon)
        plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2, f'epsilon {epsilon}')
    plot_upperbound(ax)
    #     plt.legend()
    fig.savefig('q6_1k.png')
    plt.show()


def q7():
    qstar_bed = np.load('nonstationary_qstars.npy')
    epsilon = 0.1
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    method_names = ['sample_averaging_estimates', 'constant_size_estimates']
    for i, estimate_method in enumerate([sample_averaging_estimates, constant_size_estimates]):
        average_rewards, std_error, fractional_optimal_actions = core(epsilon, modified_testbed, estimate_method,
                                                                      qstar_registry=qstar_bed)
        plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2, method_names[i])
    #     plot_upperbound(ax, modified_testbed)
    #     plt.legend()
    upper_bound = np.max(qstar_bed, axis=2)
    average_upper_bound = np.mean(upper_bound, axis=0)
    ax.plot(np.arange(default_steps), average_upper_bound[:default_steps], label='Avg Upper bound')
    ax.legend()
    fig.savefig('q7_1k.png')
    plt.show()


def q8a():
    epsilons = [0.1, 0.0]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    initial_q = [5.0, 0.0]
    for epsilon in epsilons:
        for q in initial_q:
            label = f'Epsilon {epsilon} Initial Q {q}'
            average_rewards, std_error, fractional_optimal_actions = core(epsilon, initial_q=q)
            plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2, label)
    plot_upperbound(ax, testbed)
    fig.savefig('q8a_10k.png')
    plt.show()


def q8b():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    epsilon = 0.1
    average_rewards, std_error, fractional_optimal_actions = core(epsilon)
    plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2,
         f'Epsilon greedy {epsilon}')
    average_rewards, std_error, fractional_optimal_actions = core(epsilon, selection_method_epsilon=False)
    plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2,
         f'UCB c={c}')
    plot_upperbound(ax, testbed)
    fig.savefig('q8b_1k.png')
    plt.show()

def q9():
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize)
    epsilon = 0.3
    average_rewards, std_error, fractional_optimal_actions = core(epsilon, epsilon_annealing=True)
    plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2,
         f'Epsilon anealing greedy {epsilon}')
    epsilon = 0.1
    average_rewards, std_error, fractional_optimal_actions = core(epsilon, epsilon_annealing=False)
    plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2,
         f'Epsilon greedy {epsilon}')
    epsilon = 0.01
    average_rewards, std_error, fractional_optimal_actions = core(epsilon, epsilon_annealing=False)
    plot(average_rewards, std_error, fractional_optimal_actions, ax, ax2,
         f'Epsilon greedy {epsilon}')
    fig.savefig('q9.png')
    plt.show()


q6()
q7()
q8a()
q8b()
q9()
print('Done')
