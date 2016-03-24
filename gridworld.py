import numpy as np
import operator

class GridWorld(object):
  def __init__(self, dimensions, start_state, goal_state,
      goal_reward = 10, step_reward = -1):
    self.dimensions = dimensions
    self.states = set(np.ndindex(dimensions))
    self.current_state = start_state
    self.goal_state = goal_state
    self.goal_reward = goal_reward
    self.step_reward = step_reward

  actions_and_updates = [
    ('up', (-1, 0)),
    ('down', (1, 0)),
    ('left', (0, -1)),
    ('right', (0, 1))
  ]

  actions = [name for (name, _) in actions_and_updates]
  coordinate_updates = [direction for (_, direction) in actions_and_updates]

  def take_action(self, action):
    def keep_within_bounds(coordinate, dimension_size):
      return max(0, min(coordinate, dimension_size - 1))

    def reward(new_state):
      if new_state == self.goal_state:
        return self.goal_reward + self.step_reward
      else:
        return self.step_reward

    update = self.coordinate_updates[self.actions.index(action)]
    updated_coords = map(operator.add, self.current_state, update)
    bounded_coords = map(keep_within_bounds, updated_coords, self.dimensions)
    self.current_state = tuple(bounded_coords)

    return reward(self.current_state), self.current_state
