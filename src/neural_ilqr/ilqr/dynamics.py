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
"""Dynamics model."""

import abc

import jax
import numpy as np


class Dynamics(metaclass=abc.ABCMeta):

  """Dynamics Model."""

  @property
  @abc.abstractmethod
  def dim_state(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def dim_control(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def has_hessians(self):
    """Whether the second order derivatives are available."""
    raise NotImplementedError

  @abc.abstractmethod
  def f(self, x, u):
    """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            Next state [state_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def f_x(self, x, u):
    """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/dx [state_size, state_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def f_u(self, x, u):
    """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/du [state_size, action_size].
        """
    raise NotImplementedError


class AutoDiffDynamics(Dynamics):

  """Auto-differentiated Dynamics Model."""

  def __init__(self, f, dim_state, dim_control, hessians=False, **kwargs):
    """Constructs an AutoDiffDynamics model.

        Args:
            f: Vector Theano tensor expression.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
    self._f = f
    self._dim_state = dim_state
    self._dim_control = dim_control
    self._has_hessians = hessians

    super(AutoDiffDynamics, self).__init__()

  @property
  def dim_state(self):
    return self._dim_state

  @property
  def dim_control(self):
    return self._dim_control

  @property
  def has_hessians(self):
    """Whether the second order derivatives are available."""
    return self._has_hessians

  def f(self, x, u):
    """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            Next state [state_size].
        """
    return self._f(x, u)

  @jax.jit
  def f_x(self, x, u):
    """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/dx [state_size, state_size].
        """
    return jax.jacobian(self._f, 0)(x, u)

  @jax.jit
  def f_u(self, x, u):
    """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/du [state_size, action_size].
        """
    return jax.jacobian(self._f, 1)(x, u)


@jax.jit
def apply_constraint(u, min_bounds, max_bounds, np=np):
  """Constrains a control vector between given bounds through a squashing
    function.

    Args:
        u: Control vector [action_size].
        min_bounds: Minimum control bounds [action_size].
        max_bounds: Maximum control bounds [action_size].

    Returns:
        Constrained control vector [action_size].
    """
  diff = (max_bounds - min_bounds) / 2.0
  mean = (max_bounds + min_bounds) / 2.0
  return diff * np.tanh(u) + mean
