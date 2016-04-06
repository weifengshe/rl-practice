class TD0(object):
  def __init__(self, states, learning_rate):
    self.values = dict.fromkeys(states, 0)
    self.learning_rate = learning_rate

  def start_episode(self):
    self.past_states = []

  def value(self, state):
    return self.values[state]

  def learn(self, state, action, reward, new_state):
    self.values[state] = (1 - self.learning_rate) * self.values[state] + \
        self.learning_rate * (reward + self.values[new_state])
