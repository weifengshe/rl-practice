from rl.environments import GridWorld
from rl import Simulation
from rl import util
import random

######
# Exercise: TD-learning for evaluation
#
# Look at the code in the TemporalDifferencePredictor class below
# and implement the missing parts.
#
# The gridworld looks like this:
#
# ```
# +----+
# |E...|
# |....|
# |..S.|
# |...E|
# +----+
# ```
#
# where:
#
# - `S` is the start state,
# - `E` is the end state,
# - `|` and `-` are walls
#
# Agent can walk to any of the four main directions on any state.
# Walking to a wall retains the current state. All steps cause reward -1.

def run_exercise():
  environment = GridWorld(
    dimensions=(4, 4),
    end_states=[(0, 0), (3, 3)],
    start_state=(2, 2),
    nonstates=[],
    state_rewards={},
    step_reward=-1
  )

  random_policy = RandomPolicy(environment.actions)
  td = TemporalDifferencePredictor(environment.states)

  simulation = Simulation(environment, random_policy)
  simulation.register_learner(td)
  for step in xrange(1, 1000):
    episode = simulation.run_episode()

  print "State value estimates"
  util.print_state_value_estimates(environment, td)


class TemporalDifferencePredictor(object):
  def __init__(self, states, exploration=True):
    # Lookup-table of state values.
    # They are actually tuples of x,y values but you don't
    # need to care about that.
    self.state_values = dict.fromkeys(states, 0)

    # List of states visited in this episode.
    self.episode_state_history = []

    # Constants to be tweaked according to your tastes.
    self.td_lambda = 0.9
    self.learning_rate = 0.1

  def start_episode(self):
    self.episode_state_history = []

  def learn(self, state, action, reward, new_state):
    self.episode_state_history.append(state)

    #### TODO:
    # Change this function to implement TD(0) or TD(lambda) using
    #
    # - reversed(self.episode_state_history) to iterate
    #   over past states in reverse order (if doing TD(lambda)).
    # - self.state_values[state] to get and update the
    #   state value estimates
    # - self.td_lambda for the lambda value
    # - self.learning_rate for alpha value
    #
    # You can assume that discount factor is 1.

  def state_value_estimate(self, state):
    return self.state_values[state]


class RandomPolicy(object):
  def __init__(self, actions):
    self.actions = actions

  def choose_action(self, state):
    return random.choice(self.actions)

  def start_episode(self):
    pass

  def learn(self, state, action, reward, new_state):
    pass


run_exercise()

