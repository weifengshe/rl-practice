import unittest
from simulation import Simulation
from gridworld import GridWorld


class TestSimulation(unittest.TestCase):
  def setUp(self):
    self.environment = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      goal_state = (2, 2),
      goal_reward = 10,
      step_reward = -1)
    self.simulation = Simulation(self.environment)

  def test_policy_run_for_max_steps(self):
    history = self.simulation.run_policy(lambda s: 'left')

    self.assertEqual(len(history), 101)
    self.assertEqual(history[0], ((1, 1), 'left', -1))
    self.assertEqual(history[1], ((1, 0), 'left', -1))
    self.assertEqual(history[-2], ((1, 0), 'left', -1))
    self.assertEqual(history[-1], ((1, 0), None, None))

  def test_policy_run_finding_goal(self):
    def smart_policy(state):
      actions = {
        (1, 1): 'right',
        (1, 2): 'down'
      }
      return actions[state]

    history = self.simulation.run_policy(smart_policy)

    self.assertEqual(history, [
        ((1, 1), 'right', -1),
        ((1, 2), 'down', 9),
        ((2, 2), None, None)])

  def test_policy_run_gives_feedback(self):
    def smart_policy(state):
      actions = {
        (1, 1): 'right',
        (1, 2): 'down'
      }
      return actions[state]

    feedback = []
    def accumulate_feedback(state, action, reward, new_state):
      feedback.append((state, action, reward, new_state))

    self.simulation.run_policy(smart_policy, accumulate_feedback)

    self.assertEqual(feedback, [
        ((1, 1), 'right', -1, (1, 2)),
        ((1, 2), 'down', 9, (2, 2))])
