from rl.environment import GridWorld, Cliff
from rl.prediction import Sarsa, TD
from rl.control import EpsilonPolicy, GreedyActionPolicy, GreedyAfterstatePolicy
from rl.simulation import Simulation
import numpy as np
import math


# environment = GridWorld(
#     dimensions = (10, 10),
#     start_state = (1, 0),
#     goal_state = (9, 9),
#     forbidden_states = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 3), (6, 3)],
#     goal_reward = 10,
#     step_reward = -1,
#     max_steps = 1000)
environment = Cliff(
  dimensions = (4, 12),
  start_state = (3, 0),
  goal_state = (3, 11),
  cliff_states = [(3, x) for x in xrange(1, 11)],
  cliff_reward = -100,
  step_reward = -1,
  max_steps = 100)

td = TD(
    states=environment.states,
    td_lambda=0.8,
    learning_rate=0.05)
epsilon_greedy = EpsilonPolicy(GreedyAfterstatePolicy(environment, td), lambda k: k**(-0.25))
simulation = Simulation(environment, epsilon_greedy, td)

# sarsa = Sarsa(
#     state_actions=environment.state_actions,
#     td_lambda=0,
#     learning_rate=0.05)
# greedy = GreedyActionPolicy(sarsa)
# epsilon_greedy = EpsilonPolicy(greedy, lambda k: 0.1)
# sarsa.target_policy = greedy
# simulation = Simulation(environment, epsilon_greedy, sarsa)


for step in xrange(1, 10000):
  episode = simulation.run_episode()
  if step % 10 == 0:
    print len(episode),
print

np.set_printoptions(precision=1, suppress=True, linewidth=200)
def print_values(values):
  coords = values.keys()
  (xs, ys) = zip(*coords)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for coords in values:
    array[coords] = values[coords]
  print array

print_values(td.values)
# print_values(sarsa.max_values)
# print sarsa.values((0,0))
