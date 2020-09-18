import random
import numpy as np


class Policy:
    action_dict = {'u': 'up', 'd': 'down', 'r': 'right', 'l': 'left'}
    state_action_map = {}

    def get_action(self, state, *args):
        pass

    def _dfs(self, state, goal, start):
        def add_neighbours(og_position):
            neighbours = [((og_position[0] + 1, og_position[1]), 'd'),
                          ((og_position[0] - 1, og_position[1]), 'u'),
                          ((og_position[0], og_position[1] + 1), 'r'),
                          ((og_position[0], og_position[1] - 1), 'l')]
            np.random.shuffle(neighbours)  # To get new paths at each trial
            for neighbour in neighbours:
                coordinate, action = neighbour
                if 0 <= coordinate[0] < 11 and 0 <= coordinate[1] < 11 and state[coordinate] != -1 \
                        and visited[coordinate] == 0:
                    stack.append(neighbour)
                    path[coordinate] = (og_position, action)

        stack = [(start, 'o')]
        path = {}
        visited = np.zeros((11, 11))
        goal_found = False
        while stack:
            top = stack.pop()
            pos, _ = top
            if pos == goal:
                goal_found = True
                break
            visited[pos] = 1
            add_neighbours(pos)
        if goal_found:
            node = goal
            while node != start:
                self.state_action_map[path[node][0]] = path[node][1]
                node = path[node][0]
            if start not in self.state_action_map:
                return random.choice(list(self.action_dict.values()))
            return self.state_action_map[start]


class ManualPolicy(Policy):

    def get_action(self, state, *args):
        print(state)
        action = input("Select an action as u,d,r,l for up, down, right, left respectively")
        return self.action_dict[action]


class RandomPolicy(Policy):

    def get_action(self, state, *args):
        # print(state)
        return random.choice(list(self.action_dict.values()))


class WorsePolicy(Policy):

    def get_action(self, state, *args):
        return self.action_dict['l']


class BetterThanRandomPolicy(Policy):

    def get_action(self, state, *args):
        start = args[0]
        reward_loc = args[1]
        # if start in self.state_action_map:  # A type of memoization to avoid redundant calculation
        #     return self.action_dict[self.state_action_map[start]]
        #
        # return self.action_dict[self._dfs(state, self.goal, start)]
        return self.__stochastic_policy(start, reward_loc)

    def __stochastic_policy(self, current, reward_loc=None):
        if reward_loc:
            weights = [0.25] * 4
            if reward_loc[0] > current[0]:
                weights[1] = 0.7
            else:
                weights[0] = 0.7

            if reward_loc[1] > current[1]:
                weights[2] = 0.7
            else:
                weights[3] = 0.7
            act = random.choices(list(self.action_dict.values()), weights)[0]
            return act
        else:
            return random.choice(list(self.action_dict.values()))


class LearningPolicy(Policy):
    max_reward_loc = []

    def get_action(self, state, *args):
        start = args[0]
        if start in self.state_action_map:  # A type of memoization to avoid redundant calculation
            return self.action_dict[self.state_action_map[start]]

        if self.max_reward_loc:
            return self.action_dict[self._dfs(state, self.max_reward_loc, start)]
        else:
            return random.choice(list(self.action_dict.values()))
