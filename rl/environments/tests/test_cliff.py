import unittest
from ..cliff import Cliff


class TestCliff(unittest.TestCase):
  def setUp(self):
    self.world = Cliff(
      dimensions = (4, 4),
      start_state = (3, 0),
      goal_state = (3, 3),
      cliff_states = [(3, 1), (3, 2)],
      cliff_reward = -100,
      step_reward = -1,
      max_steps = 100)

  def test_lists_all_actions(self):
    self.assertEqual(sorted(self.world.actions),
      sorted(['left', 'right', 'up', 'down']))

  def test_lists_all_possible_states(self):
    self.assertEqual(len(self.world.states), 16)

  def test_lists_all_possible_state_action_pairs(self):
    self.assertEqual(len(self.world.state_actions), 16)
    actions = self.world.state_actions[0, 0]
    self.assertEqual(sorted(actions),
      sorted(['left', 'right', 'up', 'down']))

  def test_lists_all_followups(self):
    self.assertEqual(sorted(self.world.get_followups((2, 2))), [
      ('down', -100, (3, 0)),
      ('left', -1, (2, 1)),
      ('right', -1, (2, 3)),
      ('up', -1, (1, 2))])

  def test_move(self):
    world = Cliff(
      dimensions = (4, 4),
      start_state = (1, 1),
      goal_state = (3, 3),
      cliff_states = [(3, 1), (3, 2)])

    self.assertEqual(world.current_state, (1, 1))

    _, new_state = world.take_action('up')
    self.assertEqual(new_state, (0, 1))

    _, new_state = world.take_action('left')
    self.assertEqual(new_state, (0, 0))

    _, new_state = world.take_action('down')
    self.assertEqual(new_state, (1, 0))

    _, new_state = world.take_action('right')
    self.assertEqual(new_state, (1, 1))

  def test_moves_on_boundaries(self):
    actions_on_boundaries = [
      ('up', (0, 0)),
      ('left', (0, 0)),
      ('right', (2, 2)),
      ('down', (2, 2))
    ]

    for action, state in actions_on_boundaries:
      self.world = Cliff((3, 3), state, (1, 1), [])
      _, new_state = self.world.take_action(action)
      self.assertEqual(state, new_state)

  def test_every_step_is_penalized(self):
    reward, _ = self.world.take_action('up')
    self.assertEqual(reward, -1)
    reward, _ = self.world.take_action('left')
    self.assertEqual(reward, -1)

  def test_step_to_cliff_is_penalized(self):
    reward, _ = self.world.take_action('right')
    self.assertEqual(reward, -100)

  def test_step_to_cliff_teleports_to_start_state(self):
    self.world.take_action('up')
    self.world.take_action('right')
    _, state = self.world.take_action('down')
    self.assertEqual(state, (3, 0))

  def test_process_terminates_at_goal_state(self):
    self.assertFalse(self.world.terminated)
    self.world.take_action('up')
    self.world.take_action('right')
    self.world.take_action('right')
    self.world.take_action('right')
    self.assertFalse(self.world.terminated)
    self.world.take_action('down')
    self.assertTrue(self.world.terminated)

    with self.assertRaises(AssertionError):
      self.world.take_action('down')

  def test_process_terminates_after_max_steps(self):
    self.assertEqual(self.world.max_steps, 100)

    for step in xrange(100):
      self.assertEqual(self.world.step, step)
      self.assertFalse(self.world.terminated)
      self.world.take_action('up')

    self.assertEqual(self.world.step, 100)
    self.assertTrue(self.world.terminated)

  def test_resetting_returns_to_start_state(self):
    reward, _ = self.world.take_action('up')
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('down')
    self.assertTrue(self.world.terminated)

    self.world.start_episode()
    self.assertFalse(self.world.terminated)
    self.assertEqual(self.world.current_state, (3, 0))
    self.assertEqual(self.world.step, 0)
