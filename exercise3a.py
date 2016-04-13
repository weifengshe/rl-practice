from rl.environments import GridWorld
from rl import Simulation
from rl import util
import random

######
# Exercise: Iterative policy evaluation with dynamic programming
#
# Look at the code in the IterativePolicyEvaluation class below
# and implement the __new_state_value function.
#
#
# The gridworld looks like this:
#
# ```
# +----+
# |E...|
# |....|
# |....|
# |...E|
# +----+
# ```
#
# where:
#
# - `E` is the end state,
# - `|` and `-` are walls
#
# Agent can walk to any of the four main directions on any state.
# Walking to a wall retains the current state. All steps cause reward -1.

def run_exercise():
  environment = GridWorld(
    dimensions=(4, 4),
    end_states=[(0, 0), (3, 3)],
    state_rewards={},
    step_reward=-1
  )

  policy = UniformRandomPolicy(environment.actions)
  evaluation = IterativePolicyEvaluation(environment, policy)

  for step in xrange(10000):
    evaluation.iterate()

  print "State value estimates"
  util.print_state_value_estimates(environment, evaluation)


class IterativePolicyEvaluation(object):
  def __init__(self, environment, policy):
    # Lookup-table of state values.
    # They are actually tuples of x,y values but you don't
    # need to care about that.
    self.state_values = dict.fromkeys(environment.states, 0)

    # Function from states to (action, reward, afterstate) tuples.
    self.get_followups = environment.get_followups

    # The policy probability of taking an action
    self.action_probability = policy.action_probability

  def iterate(self):
    self.state_values = {
      state: self.__new_state_value(state) for state in self.state_values}

  def __new_state_value(self, state):
    #### TODO:
    # Change this function so that it calculates
    # the new state value of the state using:
    #
    # - self.get_followups(state) to get the list of
    #   (action, reward, new_state) tuples available on this state
    # - self.action_probability(state, action) to get the
    #   policy's probability of taking the action on this state
    # - self.state_values[state] to read the current value of state
    return 0

  def state_value_estimate(self, state):
    return self.state_values[state]


class UniformRandomPolicy(object):
  def __init__(self, actions):
    self.probability = 1.0 / len(actions)

  def action_probability(self, state, action):
    return self.probability


run_exercise()

