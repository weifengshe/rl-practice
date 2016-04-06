from rl.predictors import Sarsa
from rl.policies import EpsilonPolicy, GreedyActionPolicy

class SarsaAgent(object):
  def __init__(self, environment):
    self.predictor = Sarsa(
        state_actions=environment.state_actions,
        td_lambda=0.8,
        learning_rate=0.05)
    self.policy = EpsilonPolicy(
        GreedyActionPolicy(self.predictor),
        lambda k: k**(-0.25))
    self.predictor.target_policy = self.policy

  def start_episode(self):
    self.predictor.start_episode()
    self.policy.start_episode()

  def choose_action(self, state):
    return self.policy.choose_action(state)

  def learn(self, state, action, reward, new_state):
    return self.predictor.learn(state, action, reward, new_state)

  @property
  def state_value_estimates(self):
    return self.predictor.max_values
