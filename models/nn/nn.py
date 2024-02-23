import abc

import jax


class NeuralNetwork(metaclass=abc.ABCMeta):

  """Base class for neural network."""

  def __init__(self):
    self.step_size = 0.001
    self.params = None

  @abc.abstractmethod
  def predict(self, inputs, params=None):
    """Predicts neural network output."""

  @abc.abstractmethod
  def loss(self, batch, params=None):
    """Computes loss."""

  @jax.jit
  def grad(self, batch, params=None):
    """Computes gradient."""
    return jax.grad(self.loss)(batch, params)

  @jax.jit
  def update(self, batch, params=None):
    """Updates neural network with batch."""
    grads = self.grad(batch, params)
    return [(w - self.step_size * dw, b - self.step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]
