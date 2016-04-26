from rl.environments import Cliff
from rl import Simulation
from rl import util
import random

######
# Exercise: Sarsa-learning for control
#
# Look at the code in the SarsaAgent class below and implement
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

  agent = SarsaAgent(environment, exploration=True)

  simulation = Simulation(environment, agent)
  for step in xrange(1, 1000):
    episode = simulation.run_episode()

  print "Estimated max action value of each state"
  util.print_state_value_estimates(environment, agent)

  print
  print "Sampled actions on each state"
  agent.exploration = False
  util.print_state_actions(environment, agent)


class SarsaAgent(object):
  def __init__(self, environment, exploration=True):
    # Lookup-table of state action values.
    self.state_action_values = {
      state: dict.fromkeys(actions, 0)
      for state, actions in environment.state_actions.viewitems()
    }

    # List of all available actions.
    self.actions = environment.actions

    # List of state-action pairs taken in this episode.
    self.episode_state_action_history = []

    # How many times learn() has been called
    self.learning_step = 0

    # Constants to be tweaked according to your tastes.
    self.sarsa_lambda = 0.9
    self.learning_rate = 0.1

    # When False, we will use Greedy instead of EpsilonGreedy.
    self.exploration = exploration

  def start_episode(self):
    self.episode_state_action_history = []

  def choose_action(self, state):
    epsilon = self.epsilon(self.learning_step + 1)

    #### TODO:
    # Change this function to implement an epsilon-greedy policy using
    #
    # - self.state_action_values[state][action] to get the
    #   state-action value estimates
    # - epsilon for the current epsilon value
    #
    # You can assume that discount factor is 1.

    action = random.choice(self.actions)
    self.episode_state_action_history.append((state, action))
    return action

  def learn(self, state, action, reward, new_state):
    self.learning_step += 1

    #### TODO:
    # Change this function to implement Sarsa(0) or Sarsa(lambda) using
    #
    # - reversed(self.episode_state_action_history) to iterate
    #   over past states in reverse order (if doing Sarsa(lambda)).
    # - self.state_action_values[state][action] to get and
    #   update the state-action value estimates
    # - self.sarsa_lambda for the lambda value
    # - self.learning_rate for alpha value
    #
    # You can assume that discount factor is 1.
    pass

  def epsilon(self, k):
    if self.exploration:
      return 1.0 / k
      ### Alternative schedules to try:
      # return 0.1
      # return k**(-0.75)
    else:
      return 0

  def state_value_estimate(self, state):
    return self.state_action_values[state][self.choose_action(state)]


run_exercise()
