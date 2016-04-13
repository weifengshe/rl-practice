import unittest
from ..gridworld import GridWorld


class TestGridWorld(unittest.TestCase):
  def setUp(self):
    self.world = GridWorld(
      dimensions = (3, 3),
      start_state = (1, 1),
      nonstates = [],
      end_states = [(2, 2)],
      state_rewards = {(0, 1): 10, (0, 0): 15},
      step_reward = -1,
      max_steps = 100)

  def test_lists_all_actions(self):
    self.assertEqual(sorted(self.world.actions),
      sorted(['left', 'right', 'up', 'down']))

  def test_lists_all_possible_states(self):
    self.assertEqual(len(self.world.states), 9)

  def test_lists_all_possible_state_action_pairs(self):
    self.assertEqual(len(self.world.state_actions), 9)
    actions = self.world.state_actions[0, 0]
    self.assertEqual(sorted(actions),
      sorted(['left', 'right', 'up', 'down']))

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
      self.world = GridWorld(
          dimensions=(3, 3),
          start_state=state,
          nonstates=[],
          end_states=[(1, 1)])
      _, new_state = self.world.take_action(action)
      self.assertEqual(state, new_state)

  def test_cannot_move_to_nonstates(self):
    world = GridWorld(
      dimensions = (3, 3),
      start_state = (0, 0),
      end_states = [(2, 2)],
      nonstates = [(1, 0), (1, 1)])

    _, new_state = world.take_action('down')
    self.assertEqual(new_state, (0, 0))

    world.take_action('right')
    _, new_state = world.take_action('down')
    self.assertEqual(new_state, (0, 1))

    world.take_action('right')
    world.take_action('down')
    _, new_state = world.take_action('down')
    self.assertEqual(new_state, (2, 2))

  def test_nonstates_are_not_actually_states(self):
    world = GridWorld(
      dimensions = (3, 3),
      start_state = (0, 0),
      end_states = [(2, 2)],
      nonstates = [(1, 0), (1, 1)])

    self.assertEqual(len(world.states), 7)
    self.assertFalse((1, 0) in world.states)
    self.assertFalse((1, 1) in world.states)

  def test_every_step_is_penalized(self):
    reward, _ = self.world.take_action('down')
    self.assertEqual(reward, -1)

    reward, _ = self.world.take_action('down')
    self.assertEqual(reward, -1)

  def test_reaching_reward_state(self):
    reward, _ = self.world.take_action('up')
    self.assertEqual(reward, 10 - 1)
    reward, _ = self.world.take_action('left')
    self.assertEqual(reward, 15 - 1)

  def test_process_terminates_at_goal_state(self):
    self.assertFalse(self.world.terminated)
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

  def test_lists_all_followups(self):
    self.assertEqual(sorted(self.world.get_followups((0, 1))), [
      ('down', -1, (1, 1)),
      ('left', 14, (0, 0)),
      ('right', -1, (0, 2)),
      ('up', 9, (0, 1))])

  def test_end_state_has_no_followups(self):
    self.assertEqual(sorted(self.world.get_followups((2, 2))), [])

  def test_resetting_returns_to_start_state(self):
    reward, _ = self.world.take_action('right')
    reward, _ = self.world.take_action('down')
    self.assertTrue(self.world.terminated)

    self.world.start_episode()
    self.assertFalse(self.world.terminated)
    self.assertEqual(self.world.current_state, (1, 1))
    self.assertEqual(self.world.step, 0)
