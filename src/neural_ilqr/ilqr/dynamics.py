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

  @classmethod
  @abc.abstractmethod
  def f(cls, x, u, params):
    """Dynamics model.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            Next state [state_size].
        """
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def f_x(cls, x, u, params):
    """Partial derivative of dynamics model with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/dx [state_size, state_size].
        """
    raise NotImplementedError

  @classmethod
  @abc.abstractmethod
  def f_u(cls, x, u, params):
    """Partial derivative of dynamics model with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size].

        Returns:
            df/du [state_size, action_size].
        """
    raise NotImplementedError


class AutoDiffDynamics(metaclass=abc.ABCMeta):

  """Auto-differentiated Dynamics Model."""

  def __init_subclass__(cls, f, **kwargs) -> None:
    super().__init_subclass__(**kwargs)
    cls.f = f

  def __init__(self, dim_state, dim_control, **kwargs):
    """Constructs an AutoDiffDynamics model.

        Args:
            f: Vector Theano tensor expression.
            hessians: Evaluate the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
    self.dim_state = dim_state
    self.dim_control = dim_control
