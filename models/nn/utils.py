import numpy as np


def init_random_params(scale, layer_sizes, rng=np.random.RandomState(0)):
  """Initializes random params for neural network."""
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for (m, n) in zip(layer_sizes[:-1], layer_sizes[1:])]
