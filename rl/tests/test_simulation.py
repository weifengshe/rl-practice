import unittest
from ..simulation import Simulation
from rl.environments.gridworld import GridWorld

class TestSimulation(unittest.TestCase):
  def setUp(self):
    self.environment = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      end_states = [(2, 2)],
      nonstates = [],
      state_rewards = {(2, 2): 10},
      step_reward = -1)

    self.stupid_agent = AgentStub({
        (1, 1): 'left',
        (1, 0): 'left'
      })

    self.smart_agent = AgentStub({
        (1, 1): 'right',
        (1, 2): 'down'
      })


  def test_policy_run_for_max_steps(self):
    simulation = Simulation(self.environment, self.stupid_agent)
    history = simulation.run_episode()

    self.assertEqual(len(history), 101)
    self.assertEqual(history[0], ((1, 1), 'left', -1))
    self.assertEqual(history[1], ((1, 0), 'left', -1))
    self.assertEqual(history[-2], ((1, 0), 'left', -1))
    self.assertEqual(history[-1], ((1, 0), None, None))

  def test_policy_run_finding_goal(self):
    simulation = Simulation(self.environment, self.smart_agent)
    history = simulation.run_episode()

    self.assertEqual(history, [
        ((1, 1), 'right', -1),
        ((1, 2), 'down', 9),
        ((2, 2), None, None)])

  def test_policy_run_gives_feedback(self):
    simulation = Simulation(self.environment, self.smart_agent)
    simulation.run_episode()

    self.assertEqual(self.smart_agent.experience, [
        ((1, 1), 'right', -1, (1, 2)),
        ((1, 2), 'down', 9, (2, 2))])


class AgentStub(object):
  def __init__(self, action_lookup):
    self.action_lookup = action_lookup
    self.experience = []

  def choose_action(self, state):
    return self.action_lookup[state]

  def start_episode(self):
    pass

  def learn(self, state, action, reward, new_state):
    self.experience.append((state, action, reward, new_state))
