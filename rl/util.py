import numpy as np


def print_state_value_estimates(environment, agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(precision=1, suppress=True, linewidth=200)

  (xs, ys) = zip(*environment.states)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for state in environment.states:
    array[state] = agent.state_value_estimate(state)
  print array

  np.set_printoptions(**original_printoptions)


def print_state_actions(environment, agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(suppress=True, linewidth=200)

  (xs, ys) = zip(*environment.states)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.empty(dimensions, dtype='string')
  for state in environment.states:
    array[state] = agent.choose_action(state)
  print array

  np.set_printoptions(**original_printoptions)
