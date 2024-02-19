from jax.experimental.ode import odeint
import jax.numpy as jnp
from ilqr.dynamics import AutoDiffDynamics, apply_constraint


class NeuralODE(AutoDiffDynamics):

  """Cartpole auto-differentiated dynamics model."""

  def __init__(self,
               dt,
               constrain=True,
               min_bounds=-1.0,
               max_bounds=1.0,
               **kwargs):
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
    dim_state = 3
    dim_control = 1
    w1 = jnp.zeros((dim_state + dim_control, 1024))
    b1 = jnp.zeros(1024)
    w2 = jnp.zeros((1024, 512))
    b2 = jnp.zeros(512)
    w3 = jnp.zeros((512, dim_state))
    b3 = jnp.zeros(dim_state)

    def ode_func(x):
      dx_dt = jnp.tanh(jnp.dot(x, w1) + b1)
      dx_dt = jnp.tanh(jnp.dot(dx_dt, w2) + b2)
      dx_dt = jnp.dot(dx_dt, w3) + b3
      return dx_dt

    def f(x, u, i):
      # Constrain action space.
      if constrain:
        u = apply_constraint(u, min_bounds, max_bounds, np=jnp)

      x_bar = jnp.hstack(x, u)
      x_bar = odeint(ode_func, x_bar, dt)

      return x_bar

    super(NeuralODE, self).__init__(f, dim_state=3, dim_control=1, **kwargs)
