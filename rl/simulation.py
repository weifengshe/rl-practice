class _NoLearning(object):
  def start_episode(self):
    pass

  def learn(self, state, action, reward, new_state):
    pass
_no_learning = _NoLearning()


class Simulation(object):
  def __init__(self, environment, policy, learner=_no_learning):
    self.environment = environment
    self.policy = policy
    self.learner = learner

  def run_episode(self):
    self.environment.start_episode()
    self.policy.start_episode()
    self.learner.start_episode()

    history = []
    while not self.environment.terminated:
      history.append(self.run_step())

    history.append((self.environment.current_state, None, None))
    return history

  def run_step(self):
    assert not self.environment.terminated
    state = self.environment.current_state
    action = self.policy.choose_action(state)
    reward, new_state = self.environment.take_action(action)
    self.learner.learn(state, action, reward, new_state)
    return (state, action, reward)
