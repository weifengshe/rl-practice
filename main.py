from rl.environments import GridWorld, Cliff
from rl.agents import SarsaAgent, TDAgent, QAgent
from rl import Simulation
from rl import util

# environment = GridWorld()
environment = Cliff()

agents = [
  TDAgent(environment),
  SarsaAgent(environment),
  QAgent(environment)]

for agent in agents:
  print type(agent).__name__
  simulation = Simulation(environment, agent)
  for step in xrange(1, 1000):
    episode = simulation.run_episode()

  util.print_state_value_estimates(environment, agent)
  util.print_state_actions(environment, agent)
  print
