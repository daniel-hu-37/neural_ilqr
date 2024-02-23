from jax.experimental import ode
import jax.numpy as jnp

from nn import nn, utils

_DEFAULT_PARAM_SCALE = 0.1


class NeuralODE(nn.NeuralNetwork):

  """Cartpole auto-differentiated dynamics model."""

  @classmethod
  def from_prob(cls, dim_state, dim_control, dt):
    """Creates MLP from prob."""
    layer_sizes = [dim_state + dim_control, 1024, 512, dim_state]
    params = utils.init_random_params(_DEFAULT_PARAM_SCALE, layer_sizes)
    return cls(dim_state, dim_control, params, dt)

  def __init__(self, dim_state, dim_control, params, dt, **kwargs):
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
    self.dt = dt
    super().__init__()

  def predict(self, inputs, params=None):
    """Predicts neural network output."""
    params = self.params if params is None else params

    def forward(inputs):
      """Computes dynamics."""
      activations = inputs
      for w, b, in params:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)
      return activations

    return ode.odeint(forward, inputs, self.dt)

  def loss(self, batch, params=None):
    """Computes loss."""
    inputs, targets = batch
    preds = self.predict(inputs, params)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))
