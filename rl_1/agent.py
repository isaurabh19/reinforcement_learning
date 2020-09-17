from environment import Environment
from policy import ManualPolicy, RandomPolicy, WorsePolicy, BetterThanRandomPolicy, LearningPolicy
from matplotlib import pyplot as plt
import numpy as np


def manual_agent():
    reward = 0
    four_rooms = Environment()
    s = four_rooms.grid
    manual_policy = ManualPolicy()
    while reward != 1:
        action = manual_policy.get_action(s)
        s, reward = four_rooms.simulate(action)
    print("Game Won!!")


def non_learning_agent(policy):
    i = 0
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
                four_rooms.reset_game()
            if steps % 1000 == 0:
                print(f'Steps finished {steps}/10000')
        rewards_trails.append(reward)
        print(f"Trial {i + 1}/10 done")
        i += 1

    rewards_trails = np.array(rewards_trails)
    return rewards_trails


def learning_agent():
    policy = LearningPolicy()
    i = 0
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
                # Agent learns this is where it gets the max reward and remembers this information so that it can
                # exploit
                policy.max_reward_loc = four_rooms.current
                four_rooms.reset_game()

            if steps % 1000 == 0:
                print(f'Steps finished {steps}/10000')
        rewards_trails.append(reward)
        print(f"Trial {i + 1}/10 done")
        i += 1

    rewards_trails = np.array(rewards_trails)
    return rewards_trails


def plot(rewards_trails, color):
    for count in range(10):
        plt.plot(np.arange(10000), rewards_trails[count], linestyle='dotted')
    line, = plt.plot(np.arange(10000), np.average(rewards_trails, axis=0), linestyle='solid', color=color)
    return line


# manual_agent()
random_policy_rewards = non_learning_agent(RandomPolicy())
# worse_policy_rewards = non_learning_agent(WorsePolicy())
# better_policy_rewards = non_learning_agent(BetterThanRandomPolicy())
line1 = plot(random_policy_rewards, 'red')
# line2 = plot(worse_policy_rewards, 'green')
# line3 = plot(better_policy_rewards, 'blue')
plt.ylabel("Cumulative reward")
plt.xlabel("Steps")
# plt.legend([line1, line2, line3], ['random', 'worse', 'better'])
# plt.savefig('qt4.png')
# plt.show()
learned_policy_rewards = learning_agent()
line4 = plot(learned_policy_rewards, 'black')
plt.legend([line1, line4], ['random', 'learned'])
plt.savefig('qt5.png')
plt.show()
