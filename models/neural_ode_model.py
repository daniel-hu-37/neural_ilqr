import jax
import jax.numpy as jnp

from ilqr.dynamics import AutoDiffDynamics, apply_constraint
from models.nn import neural_ode


class NeuralODEModel(AutoDiffDynamics):

  """Cartpole auto-differentiated dynamics model."""

  def __init__(self, dt, dim_state, dim_control, **kwargs):
    """Cartpole dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N].
            max_bounds: Maximum bounds for action [N].
            mc: Cart mass [kg].
            mp: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """
    self.model = neural_ode.NeuralODE(dim_state, dim_control, dt, **kwargs)

    @jax.jit
    def f(x, u):
      # Constrain action space.
      if constrain:
        u = apply_constraint(u, min_bounds, max_bounds, np=jnp)

      x_bar = jnp.hstack(x, u)
      x_bar = self.model.predict(x_bar)

      return x_bar

    super(NeuralODEModel, self).__init__(f,
                                         dim_state=3,
                                         dim_control=1,
                                         **kwargs)
