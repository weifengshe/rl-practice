from gridworld import GridWorld
from td import TD
from td0 import TD0
from greedy import Epsilon, Greedy
from simulation import Simulation
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
greedy = Epsilon(Greedy(environment.get_followups, td))

simulation = Simulation(environment)

for step in xrange(1, 10000):
  episode = simulation.run_policy(greedy, td)
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