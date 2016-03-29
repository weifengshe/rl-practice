class Simulation(object):
  def __init__(self, environment):
    self.environment = environment

  def run_policy(self, policy, feedback=None):
    self.environment.reset()
    state = self.environment.current_state

    history = []
    while not self.environment.terminated:
      action = policy(state)
      reward, new_state = self.environment.take_action(action)
      if feedback is not None:
        feedback(state, action, reward, new_state)
      history.append((state, action, reward))
      state = new_state

    history.append((new_state, None, None))
    return history
