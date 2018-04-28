import numpy as np

from tensorboardX import SummaryWriter

class Logger(object):
  def __init__(self, log_freq_episode=None, log_freq_iters=None):
    self.log_freq_episode = log_freq_episode
    self.log_freq_iters = log_freq_iters

  def log_at_iter(self, n, log_str):
    if n > 0 and n % self.log_freq_iters == 0:
      print(log_str)

  def log_at_episode(self, e, log_str):
    if e > 0 and e % self.log_freq_episode == 0:
      print(log_str)


class TensorboardXLogger(Logger):
  def __init__(self, log_dir, log_freq_episode=None, log_freq_iters=None): 
    super(TensorboardXLogger, self).__init__(
        log_freq_episode=log_freq_episode,
        log_freq_iters=log_freq_iters)
    self.log_dir = log_dir
    self.summary_writer = SummaryWriter(log_dir=log_dir)
