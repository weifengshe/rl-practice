class Simulation(object):
  def __init__(self, environment, agent):
    self.environment = environment
    self.agent = agent
    self.learners = [agent]

  def register_learner(self, learner):
    self.learners.append(learner)

  def run_episode(self):
    self.environment.start_episode()
    for learner in self.learners:
      learner.start_episode()

    while not self.environment.terminated:
      yield self.run_step()

    yield self.environment.current_state

  def run_step(self):
    assert not self.environment.terminated
    state = self.environment.current_state
    action = self.agent.choose_action(state)
    reward, new_state = self.environment.take_action(action)
    for learner in self.learners:
      learner.learn(state, action, reward, new_state)
    return (state, action, reward)
