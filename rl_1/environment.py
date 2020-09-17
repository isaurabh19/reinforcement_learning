import numpy as np
import random


class Environment:
    def __init__(self):
        self.start = (10, 0)
        self.target = (0, 10)
        self.current = self.start
        self.__initialize_grid()
        self.action_dict = {'up': Up(self.grid), 'down': Down(self.grid),
                            'right': Right(self.grid), 'left': Left(self.grid)}

    def __initialize_grid(self):
        self.grid = np.zeros((11, 11))
        self.grid[:, 5] = -1
        self.grid[2][5] = 0
        self.grid[9][5] = 0
        self.grid[5, :5] = -1
        self.grid[6, 5:] = -1
        self.grid[5, 1] = 0
        self.grid[6, 8] = 0
        self.grid[self.start] = 5

    def simulate(self, action):
        new_current, reward = self.__move(action)
        self.grid[self.current] = 0
        self.grid[new_current] = 5
        self.current = new_current
        if reward == 1:
            self.current = self.start
        return self.grid, reward

    def __move(self, action):
        reward = 0
        action_obj = self.action_dict[action]
        new_current = action_obj.move(self.current)
        if self.target == new_current:
            reward = 1
        return new_current, reward


class Action:

    def __init__(self, grid):
        self.grid = grid

    def is_allowed(self, position):
        if 0 <= position[0] < self.grid.shape[0] and 0 <= position[1] < self.grid.shape[1] and self.grid[position] != -1:
            return True
        return False

    def move(self, current_position):
        pass

    def execute_move(self, possibilities, current_position):
        weights = [0.8, 0.1, 0.1]
        new_position = random.choices(possibilities, weights)[0]
        if self.is_allowed(new_position):
            return new_position
        return current_position


class Down(Action):

    def move(self, current_position):
        next_position = current_position[0] + 1, current_position[1]
        possibilities = [next_position, (current_position[0], current_position[1] - 1),
                         (current_position[0], current_position[1] + 1)]
        return self.execute_move(possibilities, current_position)


class Up(Action):

    def move(self, current_position):
        next_position = current_position[0] - 1, current_position[1]
        possibilities = [next_position, (current_position[0], current_position[1] - 1),
                         (current_position[0], current_position[1] + 1)]
        return self.execute_move(possibilities, current_position)


class Right(Action):

    def move(self, current_position):
        next_position = current_position[0], current_position[1] + 1
        possibilities = [next_position, (current_position[0] + 1, current_position[1]),
                         (current_position[0] - 1, current_position[1])]
        return self.execute_move(possibilities, current_position)


class Left(Action):

    def move(self, current_position):
        next_position = current_position[0], current_position[1] - 1
        possibilities = [next_position, (current_position[0] + 1, current_position[1]),
                         (current_position[0] - 1, current_position[1])]
        return self.execute_move(possibilities, current_position)
