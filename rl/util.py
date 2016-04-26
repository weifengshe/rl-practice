import os
import fnmatch
import webbrowser
import petname
import numpy as np
import imageio


def generate_name():
  return petname.Generate(3, '-')

def save_animation(episode, filename):
  directory = os.path.dirname(filename)
  if not os.path.exists(directory):
      os.makedirs(directory)

  image_generator = (screenshot_pair[1] for (screenshot_pair, actions, rewards) in episode)

  imageio.mimwrite(filename, image_generator, fps=50)
  print "Saved video to %s" % filename

def open_videos_in_web_browser(directory):
  html_file_name = write_html_index_for_videos(directory)
  webbrowser.open("file://" + os.path.abspath(html_file_name))

def write_html_index_for_videos(directory):
  files = [file for file in os.listdir(directory) if fnmatch.fnmatch(file, '*.gif')]
  links = '\n'.join('<h2>%s</h2><div><img src="%s"></div>' % (file, file) for file in files)
  html = "<html><head><title>%s</title><body>%s</body></html>" % (directory, links)

  html_file_name = "%s/index.html" % directory
  with open(html_file_name, "w") as html_file:
    html_file.write(html)

  return html_file_name

def print_state_value_estimates(environment, agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(precision=1, suppress=True, linewidth=200)

  (xs, ys) = zip(*environment.states)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.zeros(dimensions)
  for state in environment.states:
    array[state] = agent.state_value_estimate(state)
  print array

  np.set_printoptions(**original_printoptions)


def print_state_actions(environment, agent):
  original_printoptions = np.get_printoptions()
  np.set_printoptions(suppress=True, linewidth=200)

  (xs, ys) = zip(*environment.states)
  dimensions = (max(*xs) + 1, max(*ys) + 1)
  array = np.full(dimensions, '-', dtype='string')
  for state in environment.states:
    array[state] = agent.choose_action(state)
  print array

  np.set_printoptions(**original_printoptions)
