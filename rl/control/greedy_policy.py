import random


class GreedyPolicy:
  def __init__(self, environment, state_values):
    self.environment = environment
    self.state_values = state_values

  def start_episode(self):
    pass

  def choose_action(self, state):
    def value(followup):
      (action, reward, next_state) = followup
      return reward + self.state_values[next_state]

    (action, _, _) = max(*self.followups(state), key=value)
    return action

  def followups(self, state):
    return self.environment.get_followups(state)

  def choices(self, state):
    return [action for (action, _, _) in self.followups(state)]