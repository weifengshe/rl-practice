from ale_python_interface import ALEInterface
import os

rom_directory = os.path.dirname(os.path.realpath(__file__)) + "/roms"

class Breakout(object):
  def __init__(self):
    self.ale = ALEInterface()
    self.ale.setInt('random_seed', 123)
    self.ale.setBool("display_screen", False)
    self.ale.setBool("sound", False)
    self.ale.loadROM("%s/breakout.bin" % rom_directory)

  def start_episode(self):
    self.ale.reset_game()

  def take_action(self, action):
    assert not self.terminated
    reward = self.ale.act(action)
    return (reward, self.current_state)

  @property
  def current_state(self):
    return self.ale.getScreenRGB()

  @property
  def actions(self):
    return self.ale.getLegalActionSet()

  @property
  def terminated(self):
    return self.ale.game_over()
