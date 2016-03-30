from rl.environment import GridWorld
from rl.prediction import TD
from rl.control import EpsilonPolicy, GreedyPolicy
from rl.simulation import Simulation
import numpy as np


environment = GridWorld(
    dimensions = (20, 20),
    start_state = (0, 0),
    goal_state = (19, 19),
    goal_reward = 10,
    step_reward = -1,
    max_steps = 100)
td = TD(
    states=environment.states,
    td_lambda=0.5,
    learning_rate=0.01)
epsilon_greedy = EpsilonPolicy(GreedyPolicy(environment.get_followups, td))

simulation = Simulation(environment, epsilon_greedy, td)

for step in xrange(1, 1000):
  episode = simulation.run_episode()
  if step % 100 == 0:
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
