from rl.environments import GridWorld
from rl import Simulation
from rl import util
import random

######
# Exercise: Value iteration with dynamic programming
#
# Look at the code in the ValueIteration class below
# and implement the __new_state_value function.
#
# The gridworld looks like this:
# ```
# +--------+
# |. . . +1|
# |. X . -1|
# |. . . . |
# +--------+
# ```
# where:
#
# - `+1` is an end state that gives reward +1 when reached,
# - `-1` is an end state that gives reward -1 when reached,
# - `|` and `-` are walls
#
# Agent can walk to any of the four main directions on
# any state. Walking to a wall retains the current state.
# All steps give reward -0.1.

def run_exercise():
  environment = GridWorld()
  value_iteration = ValueIteration(environment)

  for step in xrange(10000):
    value_iteration.iterate()

  print "Optimal state value estimates"
  util.print_state_value_estimates(environment, value_iteration)

  #### TODO extra:
  #### Uncomment the following three lines and implement
  #### ValueIteartion.choose_action function.
  # print
  # print "Estimated optimal policy"
  # util.print_state_actions(environment, value_iteration)


class ValueIteration(object):
  def __init__(self, environment):
    # Lookup-table of state values.
    # They are actually tuples of x,y values but you don't
    # need to care about that.
    self.state_values = dict.fromkeys(environment.states, 0)

    # Function from states to (action, reward, afterstate) tuples.
    self.get_followups = environment.get_followups

  def iterate(self):
    self.state_values = {
      state: self.__new_state_value(state)
      for state in self.state_values}

  def __new_state_value(self, state):
    #### TODO:
    # Change this function so that it calculates
    # the new value of the state using:
    #
    # - self.get_followups(state) to get the list of
    #   (action, reward, new_state) tuples available on this state
    # - self.state_values[state] to read the current value of state
    return 0

  def state_value_estimate(self, state):
    return self.state_values[state]

  def choose_action(self, state):
    #### TODO extra:
    # Implement this function to return the optimal action
    # based on the state values.
    return '-'


run_exercise()

