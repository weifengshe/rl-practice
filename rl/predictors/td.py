class TD(object):
  def __init__(self, states, td_lambda, learning_rate, approximator=None):
    if approximator is None:
      self.approximator = _LookupTable(states, learning_rate)
    else:
      self.approximator = approximator
    self.td_lambda = td_lambda
    self.start_episode()

  def start_episode(self):
    self.past_states = []

  def value(self, state):
    return self.approximator[state]

  def learn(self, state, action, reward, new_state):
    td_error = reward + self.approximator[new_state] - self.approximator[state]

    self.past_states.append(state)
    for index, past_state in enumerate(reversed(self.past_states)):
      self.approximator.update(past_state, (self.td_lambda**index) * td_error)


class _LookupTable(object):
  def __init__(self, states, learning_rate):
    self.values = dict.fromkeys(states, 0)
    self.learning_rate = learning_rate

  def update(self, state, change):
    self.values[state] += self.learning_rate * change

  def __getitem__(self, state):
    return self.values[state]

  def keys(self):
    return self.values.keys()
