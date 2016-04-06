class TD(object):
  def __init__(self, states, td_lambda, learning_rate):
    self.values = dict.fromkeys(states, 0)
    self.learning_rate = learning_rate
    self.td_lambda = td_lambda
    self.start_episode()

  def start_episode(self):
    self.past_states = []

  def value(self, state):
    return self.values[state]

  def learn(self, state, action, reward, new_state):
    td_error = reward + self.values[new_state] - self.values[state]

    self.past_states.append(state)
    for index, past_state in enumerate(reversed(self.past_states)):
      self.values[past_state] += \
          self.learning_rate * (self.td_lambda**index) * td_error