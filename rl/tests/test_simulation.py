import unittest
from ..simulation import Simulation
from rl.environments.gridworld import GridWorld

class TestSimulation(unittest.TestCase):
  def setUp(self):
    self.environment = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      goal_state = (2, 2),
      goal_reward = 10,
      step_reward = -1)

    self.stupid_policy = PolicyStub({
        (1, 1): 'left',
        (1, 0): 'left'
      })

    self.smart_policy = PolicyStub({
        (1, 1): 'right',
        (1, 2): 'down'
      })


  def test_policy_run_for_max_steps(self):
    simulation = Simulation(self.environment, self.stupid_policy)
    history = simulation.run_episode()

    self.assertEqual(len(history), 101)
    self.assertEqual(history[0], ((1, 1), 'left', -1))
    self.assertEqual(history[1], ((1, 0), 'left', -1))
    self.assertEqual(history[-2], ((1, 0), 'left', -1))
    self.assertEqual(history[-1], ((1, 0), None, None))

  def test_policy_run_finding_goal(self):
    simulation = Simulation(self.environment, self.smart_policy)
    history = simulation.run_episode()

    self.assertEqual(history, [
        ((1, 1), 'right', -1),
        ((1, 2), 'down', 9),
        ((2, 2), None, None)])

  def test_policy_run_gives_feedback(self):
    learner = LearnerStub()
    simulation = Simulation(self.environment, self.smart_policy, learner)
    simulation.run_episode()

    self.assertEqual(learner.accumulator, [
        ((1, 1), 'right', -1, (1, 2)),
        ((1, 2), 'down', 9, (2, 2))])


class PolicyStub(object):
  def __init__(self, action_lookup):
    self.action_lookup = action_lookup

  def choose_action(self, state):
    return self.action_lookup[state]

  def start_episode(self):
    pass

class LearnerStub(object):
  def __init__(self):
    self.accumulator = []

  def learn(self, state, action, reward, new_state):
    self.accumulator.append((state, action, reward, new_state))

  def start_episode(self):
    pass
