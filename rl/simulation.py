class Simulation(object):
  def __init__(self, environment, agent):
    self.environment = environment
    self.agent = agent
    self.learners = [agent]

  def register_learner(self, learner):
    self.learners.append(learner)

  # Run the entire episode, discarding the step results
  def run_episode(self):
    for _ in self.episode_steps():
      pass

  # Generator that yields step results
  def episode_steps(self):
    self.environment.start_episode()
    for learner in self.learners:
      learner.start_episode()

    while not self.environment.terminated:
      yield self.run_step()

    yield (self.environment.current_state, None, None)

  def run_step(self):
    assert not self.environment.terminated
    state = self.environment.current_state
    action = self.agent.choose_action(state)
    reward, new_state = self.environment.take_action(action)
    is_end = self.environment.terminated
    for learner in self.learners:
      learner.learn(state, action, reward, new_state, is_end)
    return (state, action, reward)
