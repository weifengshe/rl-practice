import random


class Greedy:
  def __init__(self, get_followups, state_values):
    self.get_followups = get_followups
    self.state_values = state_values

  def choose_action(self, state, epsilon):
    def value(followup):
      (action, reward, next_state) = followup
      return reward + self.state_values[next_state]

    followups = self.get_followups(state)
    if random.random() > epsilon:
      (action, _, _) = max(*followups, key=value)
    else:
      (action, _, _) = random.choice(followups)

    return action
