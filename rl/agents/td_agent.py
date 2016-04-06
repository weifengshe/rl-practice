from rl.predictors import TD
from rl.policies import EpsilonPolicy, GreedyAfterstatePolicy

class TDAgent(object):
  def __init__(self, environment):
    self.predictor = TD(
        states=environment.states,
        td_lambda=0.8,
        learning_rate=0.05)
    self.policy = EpsilonPolicy(
        actions=environment.actions,
        inner_policy=GreedyAfterstatePolicy(environment, self.predictor),
        epsilon=lambda k: k**(-0.25))

  def start_episode(self):
    self.predictor.start_episode()
    self.policy.start_episode()

  def choose_action(self, state):
    return self.policy.choose_action(state)

  def learn(self, state, action, reward, new_state):
    return self.predictor.learn(state, action, reward, new_state)

  def state_value_estimate(self, state):
    return self.predictor.value(state)