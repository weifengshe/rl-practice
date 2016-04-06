import numpy as np


def print_state_value_estimates(agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(precision=1, suppress=True, linewidth=200)

  values = agent.state_value_estimates

  coords = values.keys()
  (xs, ys) = zip(*coords)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for coords in values:
    array[coords] = values[coords]
  print array

  np.set_printoptions(**original_printoptions)


def print_state_actions(agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(suppress=True, linewidth=200)

  actions = { state: agent.choose_action(state) for state in agent.state_value_estimates }

  coords = actions.keys()
  (xs, ys) = zip(*coords)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.empty(dimensions, dtype='string')
  for coords in actions:
    array[coords] = actions[coords]
  print array

  np.set_printoptions(**original_printoptions)
