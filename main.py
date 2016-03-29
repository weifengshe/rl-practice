from gridworld import GridWorld
from td0 import TD0
from greedy import Greedy
from simulation import Simulation
import numpy as np


environment = GridWorld(
    dimensions = (20, 20),
    start_state = (0, 0),
    goal_state = (19, 19),
    goal_reward = 10,
    step_reward = -1,
    max_steps = 100)
td = TD0(
    states=environment.states,
    learning_rate=0.01)
greedy = Greedy(environment.get_followups, td, epsilon=1.0)

simulation = Simulation(environment)

for step in xrange(1, 1000):
  greedy.epsilon = 1.0/step
  episode = simulation.run_policy(greedy.choose_action, td.learn)
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