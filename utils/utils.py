import numpy as np
import torch

def smooth1d(y, smooth):
  """
  Smooth data using hanning filter
  :param y: input data to be smoothed
  :type y: numpy.array or list
  :param smooth: number of window length for smoothing
  :type smooth: int
  :return: smoothed data
  :rtype: numpy.array
  """
  if isinstance(y, list):
      y = np.array(y)
  if smooth >= len(y):
      smooth = 1
  y_pad = np.pad(y, (smooth//2, smooth-1-smooth//2), mode='edge')
  y_smooth = np.convolve(y_pad, np.ones((smooth,)) / smooth, mode='valid')
  return y_smooth
