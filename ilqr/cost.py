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
"""Instantaneous Cost Function."""

import abc

from jax import jacobian
import jax.numpy as jnp
import numpy as np


class Cost(metaclass=abc.ABCMeta):

  """Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

  @property
  @abc.abstractmethod
  def dim_state(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def dim_control(self):
    raise NotImplementedError

  @abc.abstractmethod
  def l(self, x, u, i, terminal=False):
    """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
    raise NotImplementedError

  @abc.abstractmethod
  def l_x(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def l_u(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def l_xx(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def l_ux(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
    raise NotImplementedError

  @abc.abstractmethod
  def l_uu(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
    raise NotImplementedError


class AutoDiffCost(Cost):

  """Auto-differentiated Instantaneous Cost.

    NOTE: The terminal cost needs to at most be a function of x and i, whereas
          the non-terminal cost can be a function of x, u and i.
    """

  def __init__(self, dim_state, dim_control, l, l_terminal=None, **kwargs):
    """Constructs an AutoDiffCost.

        Args:
            l: Vector Theano tensor expression for instantaneous cost.
                This needs to be a function of x and u and must return a scalar.
            l_terminal: Vector Theano tensor expression for terminal cost.
                This needs to be a function of x only and must retunr a scalar.
            **kwargs: Additional keyword-arguments to pass to
                `theano.function()`.
        """
    self._dim_state = dim_state
    self._dim_control = dim_control
    self._l = l
    self._l_terminal = l_terminal if l_terminal is not None else l

    super(AutoDiffCost, self).__init__()

  @property
  def dim_state(self):
    return self._dim_state

  @property
  def dim_control(self):
    return self._dim_control

  def l(self, x, u, i, terminal=False):
    """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
    if terminal:
      return self._l_terminal(x, u, i)

    return self._l(x, u, i)

  def l_x(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
    if terminal:
      return jacobian(self._l_terminal, 0)(x, u, i)

    return jacobian(self._l, 0)(x, u, i)

  def l_u(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
    if terminal:
      # Not a function of u, so the derivative is zero.
      return jnp.zeros(self.dim_control)

    return jacobian(self._l, 1)(x, u, i)

  def l_xx(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
    if terminal:
      return jacobian(self.l_x, 0)(x, u, i, terminal=True)

    return jacobian(self.l_x, 0)(x, u, i)

  def l_ux(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
    if terminal:
      # Not a function of u, so the derivative is zero.
      return jnp.zeros((self.dim_control, self.dim_state))

    return jacobian(self.l_u, 0)(x, u, i)

  def l_uu(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
    if terminal:
      # Not a function of u, so the derivative is zero.
      return jnp.zeros((self.dim_control, self.dim_control))

    return jacobian(self.l_u, 1)(x, u, i)


class QRCost(Cost):

  """Quadratic Regulator Instantaneous Cost."""

  def __init__(self, Q, R, Q_terminal=None, x_goal=None, u_goal=None):
    """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
            x_goal: Goal state [state_size].
            u_goal: Goal control [action_size].
        """
    self.Q = np.array(Q)
    self.R = np.array(R)

    if Q_terminal is None:
      self.Q_terminal = self.Q
    else:
      self.Q_terminal = np.array(Q_terminal)

    if x_goal is None:
      self.x_goal = np.zeros(Q.shape[0])
    else:
      self.x_goal = np.array(x_goal)

    if u_goal is None:
      self.u_goal = np.zeros(R.shape[0])
    else:
      self.u_goal = np.array(u_goal)

    assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
    assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
    assert self.R.shape[0] == self.R.shape[1], "R must be square"
    assert self.Q.shape[0] == self.x_goal.shape[0], "Q & x_goal mismatch"
    assert self.R.shape[0] == self.u_goal.shape[0], "R & u_goal mismatch"

    # Precompute some common constants.
    self._Q_plus_Q_T = self.Q + self.Q.T
    self._R_plus_R_T = self.R + self.R.T
    self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

    super(QRCost, self).__init__()

  def l(self, x, u, i, terminal=False):
    """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
    Q = self.Q_terminal if terminal else self.Q
    R = self.R
    x_diff = x - self.x_goal
    squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

    if terminal:
      return squared_x_cost

    u_diff = u - self.u_goal
    return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

  def l_x(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
    Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
    x_diff = x - self.x_goal
    return x_diff.T.dot(Q_plus_Q_T)

  def l_u(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
    if terminal:
      return np.zeros_like(self.u_goal)

    u_diff = u - self.u_goal
    return u_diff.T.dot(self._R_plus_R_T)

  def l_xx(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
    return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

  def l_ux(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
    return np.zeros((self.R.shape[0], self.Q.shape[0]))

  def l_uu(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
    if terminal:
      return np.zeros_like(self.R)

    return self._R_plus_R_T


class PathQRCost(Cost):

  """Quadratic Regulator Instantaneous Cost for trajectory following."""

  def __init__(self, Q, R, x_path, u_path=None, Q_terminal=None):
    """Constructs a QRCost.

        Args:
            Q: Quadratic state cost matrix [state_size, state_size].
            R: Quadratic control cost matrix [action_size, action_size].
            x_path: Goal state path [N+1, state_size].
            u_path: Goal control path [N, action_size].
            Q_terminal: Terminal quadratic state cost matrix
                [state_size, state_size].
        """
    self.Q = np.array(Q)
    self.R = np.array(R)
    self.x_path = np.array(x_path)

    state_size = self.Q.shape[0]
    action_size = self.R.shape[0]
    path_length = self.x_path.shape[0]

    if Q_terminal is None:
      self.Q_terminal = self.Q
    else:
      self.Q_terminal = np.array(Q_terminal)

    if u_path is None:
      self.u_path = np.zeros(path_length - 1, action_size)
    else:
      self.u_path = np.array(u_path)

    assert self.Q.shape == self.Q_terminal.shape, "Q & Q_terminal mismatch"
    assert self.Q.shape[0] == self.Q.shape[1], "Q must be square"
    assert self.R.shape[0] == self.R.shape[1], "R must be square"
    assert state_size == self.x_path.shape[1], "Q & x_path mismatch"
    assert action_size == self.u_path.shape[1], "R & u_path mismatch"
    assert path_length == self.u_path.shape[0] + 1, \
            "x_path must be 1 longer than u_path"

    # Precompute some common constants.
    self._Q_plus_Q_T = self.Q + self.Q.T
    self._R_plus_R_T = self.R + self.R.T
    self._Q_plus_Q_T_terminal = self.Q_terminal + self.Q_terminal.T

    super(PathQRCost, self).__init__()

  def l(self, x, u, i, terminal=False):
    """Instantaneous cost function.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            Instantaneous cost (scalar).
        """
    Q = self.Q_terminal if terminal else self.Q
    R = self.R
    x_diff = x - self.x_path[i]
    squared_x_cost = x_diff.T.dot(Q).dot(x_diff)

    if terminal:
      return squared_x_cost

    u_diff = u - self.u_path[i]
    return squared_x_cost + u_diff.T.dot(R).dot(u_diff)

  def l_x(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/dx [state_size].
        """
    Q_plus_Q_T = self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T
    x_diff = x - self.x_path[i]
    return x_diff.T.dot(Q_plus_Q_T)

  def l_u(self, x, u, i, terminal=False):
    """Partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            dl/du [action_size].
        """
    if terminal:
      return np.zeros_like(self.u_path)

    u_diff = u - self.u_path[i]
    return u_diff.T.dot(self._R_plus_R_T)

  def l_xx(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dx^2 [state_size, state_size].
        """
    return self._Q_plus_Q_T_terminal if terminal else self._Q_plus_Q_T

  def l_ux(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u and x.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/dudx [action_size, state_size].
        """
    return np.zeros((self.R.shape[0], self.Q.shape[0]))

  def l_uu(self, x, u, i, terminal=False):
    """Second partial derivative of cost function with respect to u.

        Args:
            x: Current state [state_size].
            u: Current control [action_size]. None if terminal.
            i: Current time step.
            terminal: Compute terminal cost. Default: False.

        Returns:
            d^2l/du^2 [action_size, action_size].
        """
    if terminal:
      return np.zeros_like(self.R)

    return self._R_plus_R_T
