from functools import partial

import jax
import jax.numpy as jnp

from nn import nn, utils

_DEFAULT_PARAM_SCALE = 0.1


class MultiLayerPerceptron(nn.NeuralNetwork):

  """Cartpole auto-differentiated dynamics model."""

  @classmethod
  def from_prob(cls, dim_state, dim_control):
    """Creates MLP from prob."""
    layer_sizes = [dim_state + dim_control, 1024, 512, dim_state]
    params = utils.init_random_params(_DEFAULT_PARAM_SCALE, layer_sizes)
    return cls(dim_state, dim_control, params)

  def __init__(self, dim_state, dim_control, params, **kwargs):
    """Cartpole dynamics.

        Args:
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """
    self.dim_state = dim_state
    self.dim_control = dim_control
    self.params = params
    super().__init__()

  @partial(jax.jit, static_argnums=0)
  def predict(self, inputs, params=None):
    """Predicts neural network output."""
    params = self.params if params is None else params
    activations = inputs
    for w, b, in params:
      outputs = jnp.dot(activations, w) + b
      activations = jnp.tanh(outputs)
    return activations

  @partial(jax.jit, static_argnums=0)
  def loss(self, batch, params=None):
    """Computes loss."""
    inputs, targets = batch
    preds = self.predict(inputs, params)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))
