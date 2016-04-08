import random

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
    self.learning_rate = 0.9

    # When False, we will use Greedy instead of EpsilonGreedy.
    self.exploration = exploration

  def start_episode(self):
    self.episode_state_history = []

  def choose_action(self, state):
    # Keep these two lines unchanged.
    self.episode_state_history.append(state)
    epsilon = self.epsilon(len(self.episode_state_history))

    #### TODO:
    # Change this function to implement an epsilon-greedy policy using
    #
    # - self.get_followups(state) to list the available choices
    # - self.state_values[state] to get the values of states
    # - epsilon for the current epsilon value
    #
    # You can assume that discount factor is 1.
    return random.choice(self.actions)

  def learn(self, state, action, reward, new_state):
    #### TODO:
    # Change this function to implement TD(0) or TD(lambda) using
    #
    # - reversed(self.episode_state_history[index]) to iterate
    #   over past states in reverse order (if doing TD(lambda)).
    # - self.state_values[state] to get and update the values of states
    # - self.td_lambda for the lambda value
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
    return self.state_values[state]
