class Simulation(object):
  def __init__(self, environment):
    self.environment = environment

  def run_policy(self, policy):
    self.environment.reset()
    history = []
    state = self.environment.current_state

    while not self.environment.terminated:
      action = policy(state)
      reward, new_state = self.environment.take_action(action)
      history.append((state, action, reward))
      state = new_state

    history.append((new_state, None, None))
    return history
