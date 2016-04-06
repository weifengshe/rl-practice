from rl.environments import GridWorld, Cliff
from rl.agents import SarsaAgent, TDAgent
from rl import Simulation
import numpy as np
import math

np.set_printoptions(precision=1, suppress=True, linewidth=200)
def print_state_values(values):
  coords = values.keys()
  (xs, ys) = zip(*coords)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for coords in values:
    array[coords] = values[coords]
  print array


environment = GridWorld()
# environment = Cliff()

agents = [
  SarsaAgent(environment),
  TDAgent(environment)]

for agent in agents:
  print type(agent)

  simulation = Simulation(environment, agent)

  for step in xrange(1, 10000):
    episode = simulation.run_episode()
    if step % 10 == 0:
      print len(episode),
  print

  print_state_values(agent.state_value_estimates)
  print
