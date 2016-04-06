import numpy as np


def print_grid(values):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(precision=1, suppress=True, linewidth=200)

  coords = values.keys()
  (xs, ys) = zip(*coords)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for coords in values:
    array[coords] = values[coords]
  print array

  np.set_printoptions(**original_printoptions)