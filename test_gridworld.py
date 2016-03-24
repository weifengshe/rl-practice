import unittest
from gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
  def setUp(self):
    self.world = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      goal_state = (2, 2),
      goal_reward = 10,
      step_reward = -1)

  def test_lists_all_actions(self):
    self.assertEqual(sorted(self.world.actions),
      sorted(['left', 'right', 'up', 'down']))

  def test_lists_all_possible_states(self):
    self.assertEqual(len(self.world.states), 9)

  def test_move(self):
    self.assertEqual(self.world.current_state, (1, 1))
    
    _, new_state = self.world.take_action('up')
    self.assertEqual(new_state, (0, 1))

    _, new_state = self.world.take_action('left')
    self.assertEqual(new_state, (0, 0))

    _, new_state = self.world.take_action('down')
    self.assertEqual(new_state, (1, 0))

    _, new_state = self.world.take_action('right')
    self.assertEqual(new_state, (1, 1))

  def test_moves_on_boundaries(self):
    actions_on_boundaries = [
      ('up', (0, 0)),
      ('left', (0, 0)),
      ('right', (2, 2)),
      ('down', (2, 2))
    ]

    for action, state in actions_on_boundaries:
      self.world = GridWorld((3, 3), state, (1, 1))
      _, new_state = self.world.take_action(action)
      self.assertEqual(state, new_state)

  def test_every_step_is_penalized(self):
    reward, _ = self.world.take_action('up')
    self.assertEqual(reward, -1)

    reward, _ = self.world.take_action('up')
    self.assertEqual(reward, -1)

  def test_reaching_goal_is_rewarded(self):
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('down')
    self.assertEqual(reward, 10 - 1)

  def test_process_terminates_at_goal_state(self):
    self.assertFalse(self.world.terminated)
    reward, _ = self.world.take_action('right')
    self.assertFalse(self.world.terminated)
    reward, _ = self.world.take_action('down')
    self.assertTrue(self.world.terminated)

    with self.assertRaises(AssertionError):
      self.world.take_action('down')

  def test_resetting_returns_to_start_state(self):
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('down')
    self.assertTrue(self.world.terminated)

    self.world.reset()
    self.assertFalse(self.world.terminated)
    self.assertEqual(self.world.current_state, (1, 1))
