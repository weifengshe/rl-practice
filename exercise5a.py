from rl.environments import Cliff
from rl import Simulation
from rl import util
import random

######
# Exercise: TD-learning for control
#
# Look at the code in the TDAgent class below and implement
# the missing parts. Can you teach the agent to find
# the optimal route?
#
#
# The cliff world looks like this:
#
# ```
# +-------------+
# |.............|
# |.............|
# |.............|
# |SxxxxxxxxxxxE|
# +-------------+
# ```
#
# where:
#
# - `S` is the start state,
# - `E` is the end state,
# - `x` is a canyon
# - `|` and `-` are walls
#
# Agent can walk to any of the four main directions on any state.
# Walking to a wall retains the current state. Walking to a canyon
# causes reward -100 and the agent to teleport to the start state.
# All other steps cause reward -1.

def run_exercise():
  environment = Cliff()

  agent = TDAgent(environment, exploration=True)

  simulation = Simulation(environment, agent)
  for step in xrange(1, 10000):
    episode = simulation.run_episode()

  print "State value estimates"
  util.print_state_value_estimates(environment, agent)

  print
  print "Sampled actions on each state"
  agent.exploration = False
  util.print_state_actions(environment, agent)


class TDAgent(object):
  def __init__(self, environment, exploration=True):
    # Lookup-table of state values.
    # They are actually tuples of x,y values but you don't
    # need to care about that.
    self.state_values = dict.fromkeys(environment.states, 0)

    # List of all available actions.
    self.actions = environment.actions

    # Function from states to (action, reward, afterstate) tuples.
    self.get_followups = environment.get_followups

    # List of states visited in this episode.
    self.episode_state_history = []

    # Constants to be tweaked according to your tastes.
    self.td_lambda = 0.9
    self.learning_rate = 0.01

    # When False, we will use Greedy instead of EpsilonGreedy.
    self.exploration = exploration

  def start_episode(self):
    self.episode_state_history = []

  def choose_action(self, state):
    # Keep these two lines unchanged.
    self.episode_state_history.append(state)
    epsilon = self.epsilon(len(self.episode_state_history))

    action_values = {
      action: reward + self.state_values[afterstate]
      for action, reward, afterstate in self.get_followups(state)}

    if random.random() > epsilon:
      return max(self.actions, key=lambda action: action_values[action])
    else:
      return random.choice(self.actions)

  def learn(self, state, action, reward, new_state):
    #### TODO:
    # Change this function to implement TD(0) or TD(lambda) using
    #
    # - reversed(self.episode_state_history) to iterate
    #   over past states in reverse order (if doing TD(lambda)).
    # - self.state_values[state] to get and update the values of states
    # - self.td_lambda for the lambda value
    # - self.learning_rate for alpha value
    #
    # You can assume that discount factor is 1.

    td_error = reward + self.state_values[new_state] - self.state_values[state]
    for index, state in enumerate(reversed(self.episode_state_history)):
      change = self.learning_rate * (self.td_lambda**index) * td_error
      self.state_values[state] += change

  def epsilon(self, k):
    if self.exploration:
      return 1.0 / k
      ### Alternative schedules to try:
      # return 0.1
      # return k**(-0.75)
    else:
      return 0

  def state_value_estimate(self, state):
    return self.state_values[state]


run_exercise()