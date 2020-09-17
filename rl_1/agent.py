from environment import Environment
from policy import ManualPolicy, RandomPolicy, WorsePolicy, BetterThanRandomPolicy
from matplotlib import pyplot as plt
import numpy as np


def qt2():
    reward = 0
    four_rooms = Environment()
    s = four_rooms.grid
    manual_policy = ManualPolicy()
    while reward != 1:
        action = manual_policy.get_action(s)
        s, reward = four_rooms.simulate(action)
    print("Game Won!!")


def qt3(policy):
    i = 0
    # policy = RandomPolicy()
    rewards_trails = []
    while i < 10:
        steps = 0
        reward = []
        reward_sum = 0
        four_rooms = Environment()
        state = four_rooms.grid
        while steps < 10000:
            action = policy.get_action(state, four_rooms.current)
            state, r = four_rooms.simulate(action)
            steps += 1
            reward_sum += r
            reward.append(reward_sum)
            if r == 1:
                print(f"game won! in step no {steps}")
            if steps % 1000 == 0:
                print(f'Steps finished {steps}/10000')
        rewards_trails.append(reward)
        print(f"Trial {i + 1}/10 done")
        i += 1

    rewards_trails = np.array(rewards_trails)
    # for count in range(10):
    #     plt.plot(np.arange(10000), rewards_trails[count], linestyle='dotted')
    # plt.plot(np.arange(10000), np.average(rewards_trails, axis=0), color='black')
    # plt.ylabel("Cumulative reward")
    # plt.xlabel("Steps")
    # plt.savefig('qt3.png')
    # plt.show()
    return rewards_trails


# def qt4b():
#     i = 0
#     rewards_trails = []
#     while i < 10:
#         policy = BetterThanRandomPolicy()
#         four_rooms = Environment()
#         state = four_rooms.grid
#         reward = []
#         reward_sum = 0
#         steps = 0
#         while steps < 10000:
#             action = policy.get_action(state, four_rooms.current)
#             print(state)
#             print(f'Action is {action}')
#             state, r = four_rooms.simulate(action)
#             if r == 1:
#                 print(f"Game Won!! in {steps}")
#             steps += 1
#             reward_sum += r
#             reward.append(reward_sum)
#         rewards_trails.append(reward)
#         i += 1
#
#     rewards_trails = np.array(rewards_trails)
#     for count in range(10):
#         plt.plot(np.arange(10000), rewards_trails[count], linestyle='dotted')
#     plt.plot(np.arange(10000), np.average(rewards_trails, axis=0), color='black')
#     plt.ylabel("Cumulative reward")
#     plt.xlabel("Steps")
#     plt.savefig('qt4.png')
#     plt.show()


# qt2()
def plot(rewards_trails, color):
    for count in range(10):
        plt.plot(np.arange(10000), rewards_trails[count], linestyle='dotted')
    line,  = plt.plot(np.arange(10000), np.average(rewards_trails, axis=0), linestyle='solid', color=color)
    plt.ylabel("Cumulative reward")
    plt.xlabel("Steps")
    return line


random_policy_rewards = qt3(RandomPolicy())
worse_policy_rewards = qt3(WorsePolicy())
better_policy_rewards = qt3(BetterThanRandomPolicy())
line1 = plot(random_policy_rewards, 'black')
line2 = plot(worse_policy_rewards, 'green')
line3 = plot(better_policy_rewards, 'blue')
plt.legend([line1, line2, line3], ['random', 'worse', 'better'])
plt.savefig('qt4.png')
plt.show()
