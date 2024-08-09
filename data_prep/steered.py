# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Steered Molecular Dynamics.

Biasing a simulation towards a value of a collective variable using a time dependent Harmonic Bias.
This method implements such a bias.

The Hamiltonian is amended with a term
:math:`\\mathcal{H} = \\mathcal{H}_0 + \\mathcal{H}_\\mathrm{HB}(\\xi(t))` where
:math:`\\mathcal{H}_\\mathrm{HB}(\\xi) = \\boldsymbol{K}/2 (\\xi_0(t) - \\xi)^2`
biases the simulations around the collective variable :math:`\\xi_0(t)`.
"""

from typing import NamedTuple

import jax
import jax.nn as nn
from jax import JaxArray
from jax import numpy as np
from pysages.methods.bias import Bias
from pysages.methods.core import generalize
from pysages.colvars.core import TwoPointCV, AxisCV
from pysages.colvars.coordinates import weighted_barycenter, barycenter


JaxArray = jax.Array

class SteeredState(NamedTuple):
    """
    Description of a state biased by a harmonic potential for a CV.

    xi: JaxArray
        Collective variable value of the last simulation step.

    bias: JaxArray
        Array with harmonic biasing forces for each particle in the simulation.

    centers: JaxArray
        Moving centers of the harmonic bias applied.

    forces: JaxArray
        Array with harmonic forces for each collective variable in the simulation.

    work: JaxArray
        Array with the current work applied in the simulation.
    """

    xi: JaxArray
    bias: JaxArray
    centers: JaxArray
    forces: JaxArray
    work: JaxArray

    def __repr__(self):
        return repr("PySAGES" + type(self).__name__)


class Steered(Bias):
    """
    Steered method class.
    """

    __special_args__ = Bias.__special_args__.union({"kspring", "steer_velocity"})

    def __init__(self, cvs, kspring, center, steer_velocity, **kwargs):
        """
        Arguments
        ---------
        cvs: Union[List, Tuple]
            A list or tuple of collective variables, length `N`.
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        center:
            An array of length `N` representing the initial state of the harmonic biasing potential.
        steer_velocity:
            An array of length `N` representing the constant steer_velocity. Units are cvs/time.
        """
        super().__init__(cvs, center, **kwargs)
        self.cv_dimension = len(cvs)
        self.kspring = kspring
        self.steer_velocity = steer_velocity

    def __getstate__(self):
        state, kwargs = super().__getstate__()
        state["kspring"] = self._kspring
        state["steer_velocity"] = self._steer_velocity
        return state, kwargs

    @property
    def kspring(self):
        """
        Retrieve the spring constant.
        """
        return self._kspring

    @kspring.setter
    def kspring(self, kspring):
        """
        Set new spring constant.

        Arguments
        ---------
        kspring:
            A scalar, array length `N` or symmetric `N x N` matrix. Restraining spring constant.
        """
        # Ensure array
        kspring = np.asarray(kspring)
        shape = kspring.shape
        N = self.cv_dimension

        if len(shape) > 2:
            raise RuntimeError(f"Wrong kspring shape {shape} (expected scalar, 1D or 2D)")
        if len(shape) == 2:
            if shape != (N, N):
                raise RuntimeError(f"2D kspring with wrong shape, expected ({N}, {N}), got {shape}")
            if not np.allclose(kspring, kspring.T):
                raise RuntimeError("Spring matrix is not symmetric")

            self._kspring = kspring
        else:  # len(shape) == 0 or len(shape) == 1
            kspring_size = kspring.size
            if kspring_size not in (N, 1):
                raise RuntimeError(f"Wrong kspring size, expected 1 or {N}, got {kspring_size}.")

            self._kspring = np.identity(N) * kspring
        return self._kspring

    @property
    def steer_velocity(self):
        """
        Retrieve current steer_velocity of the collective variable.
        """
        return self._steer_velocity

    @steer_velocity.setter
    def steer_velocity(self, steer_velocity):
        """
        Set the steer_velocity of the collective variable.
        """
        steer_velocity = np.asarray(steer_velocity)
        if steer_velocity.shape == ():
            steer_velocity = steer_velocity.reshape(1)
        if len(steer_velocity.shape) != 1 or steer_velocity.shape[0] != self.cv_dimension:
            raise RuntimeError(
                f"Invalid steer_velocity expected {self.cv_dimension} got {steer_velocity.shape}."
            )
        self._steer_velocity = steer_velocity

    def build(self, snapshot, helpers, *args, **kwargs):
        return _steered(self, snapshot, helpers)


def _steered(method, snapshot, helpers):
    cv = method.cv
    center = method.center
    steer_velocity = method.steer_velocity
    dt = snapshot.dt
    kspring = method.kspring
    natoms = np.size(snapshot.positions, 0)

    def initialize():
        xi, _ = cv(helpers.query(snapshot))
        bias = np.zeros((natoms, helpers.dimensionality()))
        centers = center
        work = np.asarray(0.0)
        forces = np.zeros(len(xi))
        return SteeredState(xi, bias, centers, forces, work)

    def update(state, data):
        xi, Jxi = cv(data)
        forces = kspring @ (xi - state.centers).flatten()
        work = state.work + forces @ steer_velocity.flatten() * dt
        centers = state.centers + dt * steer_velocity
        bias = -Jxi.T @ forces.flatten()
        bias = bias.reshape(state.bias.shape)

        return SteeredState(xi, bias, centers, forces, work)

    return snapshot, initialize, generalize(update, helpers)


class SteeredLogger:
    """
    Logs the state of the collective variable and other parameters in Steered.

    Parameters
    ----------
    steered_file:
        Name of the output steered log file.

    log_period:
        Time steps between logging of collective variables and Steered parameters.
    """

    def __init__(self, steered_file, log_period):
        """
        SteeredLogger constructor.
        """
        self.steered_file = steered_file
        self.log_period = log_period
        self.counter = 0

    def save_work(self, xi, centers, forces, work):
        """
        Append the cv, centers, and work to log file.
        """
        with open(self.steered_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\t")
            f.write("\t".join(map(str, centers.flatten())) + "\t")
            f.write("\t".join(map(str, forces.flatten())) + "\t")
            f.write(str(work) + "\n")

    def __call__(self, snapshot, state, timestep):
        """
        Implements the logging itself. Interface as expected for Callbacks.
        """
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            self.save_work(state.xi, state.centers, state.forces, state.work)

        self.counter += 1

class DistanceToInterface(TwoPointCV):
    def __init__(self, indices, axis, sigma, scope, bins=100, coeff=1):
        super().__init__(indices)
        self.axis = axis
        self.sigma = sigma
        self.scope = scope
        self.bins = bins
        self.coeff = coeff
    @property
    def function(self):
        return lambda r1, r2: distance_to_interface(
            r1, r2, axis=self.axis,
            sigma=self.sigma, scope=self.scope,
            bins=self.bins, coeff=self.coeff
        )
def distance_to_interface(p1, p2, axis, sigma, scope, bins, coeff):
    mobile_axis = barycenter(p1)[axis]
    positions_axis = p2.flatten()[axis::3]
    centers = np.linspace(scope[0], scope[1], bins)
    centers = np.expand_dims(centers, 1)
    positions_axis = np.expand_dims(positions_axis, 0)
    diff = positions_axis - centers
    mass = np.exp(-0.5 * (diff / sigma) ** 2)
    mass = np.sum(mass, axis=1)
    mass_diff = np.abs(mass[1:] - mass[:-1])
    centers = np.squeeze(centers)
    centers_mean = (centers[1:] + centers[:-1]) / 2
    probability = nn.softmax(mass_diff * coeff)
    interface = np.sum(probability * centers_mean)
    return mobile_axis - interface



class Orientational_Order_Parameter_QTensor(AxisCV):
    """
    Calculate the orientational order parameter S of the liquid crystals.
    This version calculates the S using diagonalization of the Q tensor.

    ***THIS COLLECTIVE VARIABLE DOES NOT CONSIDER PBE***

    The orientational order parameter S is a collective variable that is used to
    measure the degree of molecular alignment or order within the material, 
    usually liquid crystal. It quantifies the strength of the alignment.
    An order parameter S=0 corresponds to a randomly aligned system, whereas an
    order parameter S=1 corresponds to a perfect alignment along a specific
    direction.

    Parameters
    ----------
    positions: DeviceArray of the shape ([2N, ])
        Points in space that form the set of vectors.
    formation: numpy.ndarray of the shape ([N, 2])
        Specifies which pairs of the atoms will form a vector
    Returns
    -------
    float
        The average degree of which all the vectors differs from their normal
        vector.
    """

    def __init__(self, indices, axis, formation):
        super().__init__(indices, axis)
        self.axis = axis
        self.formation = formation


    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.vector_normal_discrepancy` for details.
        """
        # TODO: implement the function
        return lambda r1: orientational_order_parameter_qtensor(r1,
                                                                self.formation)
def orientational_order_parameter_qtensor(positions, formation):
    """
    Calculate the orientational order parameter S of the liquid crystals.

    The orientational order parameter S is a collective variable that is used to
    measure the degree of molecular alignment or order within the material, 
    usually liquid crystal. It quantifies the strength of the alignment.
    An order parameter S=0 corresponds to a randomly aligned system, whereas an
    order parameter S=1 corresponds to a perfect alignment along a specific
    direction.

    ***THIS COLLECTIVE VARIABLE ASSUMES ALL POSITIONS ARE NOT WRAPPED BY PBC***

    Parameters
    ----------
    positions: DeviceArray of the shape ([2N, ])
        Points in space that form the set of vectors.
    formation: numpy.ndarray of the shape ([N, 2])
        Specifies which pairs of the atoms will form a vector
    Returns
    -------
    float
        The orientational order parameter S, in the range of [-0.5, 1].

    """
    N = formation.shape[0]
    # jax.debug.print("cvcos: pos={x}", x=positions[:3])
    # Calculate the orientations
    formation = formation.flatten() # ([2N, ])
    vectors = positions[formation] # ([2N, 3])
    vectors = vectors.reshape([N, 2, 3])
    vector_directions = vectors[:, 0, :] - vectors[:, 1, :] # [N, 3]

    # Normalize the orientations
    vector_directions = vector_directions / np.linalg.norm(vector_directions,
                                                           axis=1,
                                                           keepdims=True)

    # Calculate the outer product
    reshaped_directions = vector_directions[:, :, np.newaxis] # [N, 3, 1]
    outer_products = reshaped_directions @ reshaped_directions.transpose(0, 2, 1)

    # Calculate the Q-tensor
    q_tensor = np.mean(outer_products, axis=0) * 1.5 - 0.5 * np.eye(3)

    # Calculate the order parameter by diagonalize the Q tensor
    eigw, eigv = np.linalg.eigh(q_tensor)
    eigenvalue = eigw[np.abs(eigw).argmax()]
    # jax.debug.print("cv: {x}", x=np.real(eigenvalue))
    return np.real(eigenvalue)


class Orientational_Order_Parameter_Axis(AxisCV):
    """
    Calculate the orientational order parameter S of the liquid crystals.
    This version calculates the S w.r.t. particular axis.

    ***THIS COLLECTIVE VARIABLE DOES NOT CONSIDER PBE***

    The orientational order parameter S is a collective variable that is used to
    measure the degree of molecular alignment or order within the material, 
    usually liquid crystal. It quantifies the strength of the alignment.
    An order parameter S=0 corresponds to a randomly aligned system, whereas an
    order parameter S=1 corresponds to a perfect alignment along a specific
    direction.

    Parameters
    ----------
    positions: DeviceArray of the shape ([2N, ])
        Points in space that form the set of vectors.
    formation: numpy.ndarray of the shape ([N, 2])
        Specifies which pairs of the atoms will form a vector
    Returns
    -------
    float
        The average degree of which all the vectors differs from their normal
        vector.
    """

    def __init__(self, indices, axis, formation):
        super().__init__(indices, axis)
        self.axis = axis
        self.formation = formation


    @property
    def function(self):
        """
        Returns
        -------
        Callable
            See `pysages.colvars.vector_normal_discrepancy` for details.
        """
        return lambda r1: orientational_order_parameter(r1, self.axis, self.formation)


def orientational_order_parameter(positions, axis, formation):
    """
    Calculate the orientational order parameter S of the liquid crystals.

    The orientational order parameter S is a collective variable that is used to
    measure the degree of molecular alignment or order within the material, 
    usually liquid crystal. It quantifies the strength of the alignment.
    An order parameter S=0 corresponds to a randomly aligned system, whereas an
    order parameter S=1 corresponds to a perfect alignment along a specific
    direction.

    ***THIS COLLECTIVE VARIABLE ASSUMES ALL POSITIONS ARE NOT WRAPPED BY PBE***

    Parameters
    ----------
    positions: DeviceArray of the shape ([2N, ])
        Points in space that form the set of vectors.
    axis: int
        The axis as the director
    formation: DeviceArray of the shape ([N, 2])
        Specifies which pairs of the atoms will form a vector
    Returns
    -------
    float
        The orientational order parameter S, in the range of [-0.5, 1].
    """
    N = formation.shape[0]
    # Calculate the orientations
    formation = formation.flatten() # ([2N, ])
    vectors = positions[formation] # ([2N, 3])
    vectors = vectors.reshape([N, 2, 3])
    vector_directions = vectors[:, 0, :] - vectors[:, 1, :] # [N, 3]
    
    # Normalize the orientations
    vector_directions = vector_directions / np.linalg.norm(vector_directions, 
                                                           axis=1,
                                                           keepdims=True)
    
    # Calculate the S value
    axis_cosine = vector_directions[:, axis] # [N, ]
    s_value = (3 * np.power(axis_cosine, 2) - 1) / 2
    output = np.mean(s_value)
    return output


class CVLogger:
    def __init__(self, cv_file, log_period):
        self.cv_file = cv_file
        self.log_period = log_period
        self.counter = 0


    def save_cv(self, xi):
        with open(self.cv_file, "a+", encoding="utf8") as f:
            f.write(str(self.counter) + "\t")
            f.write("\t".join(map(str, xi.flatten())) + "\n")


    def __call__(self, snapshot, state, timestep):
        if self.counter >= self.log_period and self.counter % self.log_period == 0:
            self.save_cv(state.xi)
        self.counter += 1
