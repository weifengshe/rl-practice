from rl.environment import GridWorld
from rl.prediction import Sarsa, TD
from rl.control import EpsilonPolicy, GreedyActionPolicy, GreedyStatePolicy
from rl.simulation import Simulation
import numpy as np
import math


environment = GridWorld(
    dimensions = (3, 3),
    start_state = (1, 0),
    goal_state = (2, 2),
    forbidden_states = [(1, 1)],
    goal_reward = 10,
    step_reward = -1,
    max_steps = 1000)
sarsa = Sarsa(
    state_actions=environment.state_actions,
    td_lambda=0.8,
    learning_rate=0.05)
td = TD(
    states=environment.states,
    td_lambda=0.9,
    learning_rate=0.05)
# epsilon_greedy = EpsilonPolicy(GreedyStatePolicy(environment, td), lambda k: k**(-0.2))
# td.policy = epsilon_greedy
# simulation = Simulation(environment, epsilon_greedy, td)
epsilon_greedy = EpsilonPolicy(GreedyActionPolicy(sarsa), lambda k: k**(-0.25))
sarsa.policy = epsilon_greedy
simulation = Simulation(environment, epsilon_greedy, sarsa)


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

#print_values(td.values)
print_values(sarsa.max_values)
print sarsa.values((0,0))
