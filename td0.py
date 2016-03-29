class TD0(object):
  def __init__(self, states, learning_rate):
    self.values = dict.fromkeys(states, 0)
    self.learning_rate = learning_rate

  def __getitem__(self, state):
    return self.values[state]

  def learn(self, state, action, reward, new_state):
    self.values[state] = (1 - self.learning_rate) * self[state] + \
        self.learning_rate * (reward + self[new_state])
