
class Greedy:
  def __init__(self, get_followups, state_values):
    self.get_followups = get_followups
    self.state_values = state_values

  def choose_action(self, state):
    def value(followup):
      (action, reward, next_state) = followup
      return reward + self.state_values[next_state]

    (action, _, _) = max(*self.get_followups(state), key=value)
    return action
