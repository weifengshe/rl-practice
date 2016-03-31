from operator import itemgetter


class GreedyActionPolicy:
  def __init__(self, state_action_values):
    self.state_action_values = state_action_values

  def start_episode(self):
    pass

  def choose_action(self, state):
    action_values = self.state_action_values.action_values(state)
    return max(action_values.iteritems(), key=itemgetter(1))[0]

  def choices(self, state):
    return self.state_action_values.action_values(state).keys()
