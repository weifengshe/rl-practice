class _LookupTable(object):
  def __init__(self, state_actions, learning_rate):
    self.__values = {
      state: dict.fromkeys(actions, 0)
      for state, actions in state_actions.viewitems()
    }
    self.learning_rate = learning_rate

  def update(self, state, action, change):
    self.__values[state][action] += self.learning_rate * change

  def values(self, state):
    return self.__values[state]

  def value(self, state, action):
    return self.__values[state][action]

  def keys(self):
    return self.__values.keys()


class Sarsa(object):
  def __init__(self, state_actions, td_lambda, learning_rate, approximator=None):
    if approximator is None:
      self.approximator = _LookupTable(state_actions, learning_rate)
    else:
      self.approximator = approximator
    self.td_lambda = td_lambda
    self.start_episode()

  def start_episode(self):
    self.past_state_actions = []

  def values(self, state):
    return self.approximator.values(state)

  def value(self, state, action):
    return self.approximator.value(state, action)

  def learn(self, state, action, reward, new_state):
    assert self.target_policy is not None
    new_action = self.target_policy.choose_action(new_state)
    td_error = reward + self.approximator.value(new_state, new_action) - self.approximator.value(state, action)

    self.past_state_actions.append((state, action))
    for index, (past_state, past_action) in enumerate(reversed(self.past_state_actions)):
      self.approximator.update(past_state, past_action, (self.td_lambda**index) * td_error)
