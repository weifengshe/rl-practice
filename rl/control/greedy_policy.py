import random


class GreedyPolicy:
  def __init__(self, get_followups, state_values):
    self.get_followups = get_followups
    self.state_values = state_values

  def start_episode(self):
    pass

  def choose_action(self, state):
    def value(followup):
      (action, reward, next_state) = followup
      return reward + self.state_values[next_state]

    followups = self.get_followups(state)
    (action, _, _) = max(*followups, key=value)
    return action
