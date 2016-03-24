import numpy as np
import operator

class GridWorld(object):
  def __init__(self, dimensions, start, goal):
    self.dimensions = dimensions
    self.current = start
    self.goal = goal
    self.all_positions = set(np.ndindex(dimensions))

  directions_with_names = [
    ('up', (-1, 0)),
    ('down', (1, 0)),
    ('left', (0, -1)),
    ('right', (0, 1))
  ]

  directions = [direction for (_, direction) in directions_with_names]
  direction_names = [name for (name, _) in directions_with_names]

  def move(self, direction_name):
    def within_bounds(idx, size):
      return max(0, min(idx, size - 1))

    direction_idx = self.direction_names.index(direction_name)
    coordinate_updates = self.directions[direction_idx]
    updated_position = map(operator.add, self.current, coordinate_updates)
    bounded_position = map(within_bounds, updated_position, self.dimensions)
    self.current = tuple(bounded_position)
