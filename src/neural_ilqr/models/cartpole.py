# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Cartpole example."""
from functools import partial

import jax
import jax.numpy as jnp

from ilqr.dynamics import AutoDiffDynamics


class CartpoleDynamics:

  """Cartpole auto-differentiated dynamics model."""

  def __init__(self, mc=1.0, mp=0.1, l=1.0, g=9.80665, **kwargs):
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
    self._params = {'mc': mc, 'mp': mp, 'l': l, 'g': g}
    self.dim_state = 4
    self.dim_control = 1
    self.f_x = jax.jit(jax.jacobian(self.f, 0))
    self.f_u = jax.jit(jax.jacobian(self.f, 1))

  @partial(jax.jit, static_argnums=0)
  def f(self, x, u, **params):
    x_ = x[0]
    x_dot = x[1]
    sin_theta = jnp.sin(x[2])
    cos_theta = jnp.cos(x[2])
    theta_dot = x[3]
    F = u[0]

    # Define dynamics model as per Razvan V. Florian's
    # "Correct equations for the dynamics of the cart-pole system".
    # Friction is neglected.
    params = self._params if not params else params

    # Eq. (23)
    temp = (F + params['mp'] * params['l'] * theta_dot**2 * sin_theta) / (
        params['mc'] + params['mp'])
    numerator = params['g'] * sin_theta - cos_theta * temp
    denominator = params['l'] * (4.0 / 3.0 - params['mp'] * cos_theta**2 /
                                 (params['mc'] + params['mp']))
    theta_dot_dot = numerator / denominator

    # Eq. (24)
    x_dot_dot = temp - params['mp'] * params[
        'l'] * theta_dot_dot * cos_theta / (params['mc'] + params['mp'])

    # Deaugment state for dynamics.
    theta = jnp.arctan2(sin_theta, cos_theta)
    dt = 0.05

    return jnp.stack([
        x_ + x_dot * dt,
        x_dot + x_dot_dot * dt,
        theta + theta_dot * dt,
        theta_dot + theta_dot_dot * dt,
    ]).T
