from types import TupleType

class Sarsa(object):
  def __init__(self, state_actions, td_lambda, learning_rate):
    self.__values = {
      state: dict.fromkeys(actions, 0)
      for state, actions in state_actions.viewitems()
    }
    self.learning_rate = learning_rate
    self.td_lambda = td_lambda
    self.start_episode()

  def start_episode(self):
    self.past_state_actions = []

  def action_values(self, state):
    return self.__values[state]

  def value(self, state, action):
    return self.__values[state][action]

  def __getitem__(self, state_action):
    (state, action) = state_action
    return self.value(state, action)

  def learn(self, state, action, reward, new_state):
    assert self.policy is not None
    new_action = self.policy.choose_action(new_state)
    td_error = reward + self[new_state, new_action] - self[state, action]

    self.past_state_actions.append((state, action))
    for index, (past_state, past_action) in enumerate(reversed(self.past_state_actions)):
      self.__values[past_state][past_action] += \
          self.learning_rate * (self.td_lambda**index) * td_error

  @property
  def max_values(self):
    return {
        state: max(self.__values[state].values())
        for state in self.__values
      }
