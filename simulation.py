class _NoLearning(object):
  def reset(self):
    pass

  def learn(self, state, action, reward, new_state):
    pass
_no_learning = _NoLearning()


class Simulation(object):
  def __init__(self, environment):
    self.environment = environment

  def run_policy(self, policy, learner=_no_learning):
    self.environment.reset()
    policy.reset()
    learner.reset()

    state = self.environment.current_state
    history = []

    while not self.environment.terminated:
      action = policy.choose_action(state)
      reward, new_state = self.environment.take_action(action)
      learner.learn(state, action, reward, new_state)
      history.append((state, action, reward))
      state = new_state

    history.append((new_state, None, None))
    return history
