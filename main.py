from rl.environments import Cliff
from rl.agents import TDAgent
from rl import Simulation
from rl import util

environment = Cliff()

agent = TDAgent(environment, exploration=True)

simulation = Simulation(environment, agent)
for step in xrange(1, 1000):
  episode = simulation.run_episode()

print "State value estimates"
util.print_state_value_estimates(environment, agent)

print
print "Sampled actions on each state"
agent.exploration = False
util.print_state_actions(environment, agent)
