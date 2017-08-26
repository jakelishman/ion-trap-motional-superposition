from __future__ import division
import numpy as np
import scipy.optimize
from math import *

# Functions on linear operators

def adj(arr):
    """Adjoint of a matrix."""
    return np.conj(arr.T)
def inner_product(final, operator, start):
    """
    Calculates the quantity <final|operator|start>, assuming that both final and
    start are given as kets, represented by 1D numpy.arrays of complex.
    """
    return adj(final).dot(operator).dot(start)


# Building blocks of linear operators for sideband transitions

def ladder_transition_indices(dim, ladder):
    """
    Returns a tuple which, when used as an array index, will return a single
    slice of the sigma-plus * ladder diagonal, followed immediately by the
    sigma-minus * adj(ladder) diagonal.
    """
    if ladder >= 0:
        xs = np.arange(ladder, dim - ladder, dtype = np.intp)
    else:
        xs = np.concatenate((
            np.arange((dim // 2) + ladder, dtype = np.intp),
            np.arange((dim // 2) - ladder, dim, dtype = np.intp) ))
    return (xs, np.roll(xs, xs.shape[0] // 2))

def generate_carrier_updater(matrix):
    """
    Create a function which will update the given matrix representing a carrier
    transition, when given a new angle.
    """
    dim = matrix.shape[0]
    trans_indx = ladder_transition_indices(dim, 0)
    def update_carrier(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the carrier transition."""
        np.fill_diagonal(matrix, cos(angle) + 0.0j)
        matrix[trans_indx] = -1.0j * sin(angle)
    return update_carrier

def generate_d_carrier_updater(matrix):
    """
    Create a function which will update the given matrix representing the
    derivative of a carrier transition, when given a new angle.
    """
    dim = matrix.shape[0]
    trans_indx = ladder_transition_indices(dim, 0)
    def update_d_carrier(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the derivative of the carrier transition."""
        np.fill_diagonal(matrix, -sin(angle))
        matrix[trans_indx] = -1.0j * cos(angle)
    return update_d_carrier

def generate_red_updater(matrix):
    """
    Create a function which will update the given matrix representing a 1st red
    transition, when given a new angle.
    """
    dim = matrix.shape[0]
    ns = dim // 2
    phase_indx = np.diag_indices(dim)
    trans_indx = ladder_transition_indices(dim, -1)
    root = [ sqrt(n) for n in xrange(ns + 1) ]
    def update_red(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the 1st red transition."""
        matrix[phase_indx] = [ cos(root[n + z] * angle)
                               for z in (1, 0) for n in xrange(ns) ]
        matrix[trans_indx] = [ -1.0j * sin(root[n] * angle)
                               for _ in (0, 1) for n in xrange(1, ns) ]
    return update_red

def generate_d_red_updater(matrix):
    """
    Create a function which will update the given matrix representing the
    derivative of a 1st red transition, when given a new angle.
    """
    dim = matrix.shape[0]
    ns = dim // 2
    phase_indx = np.diag_indices(dim)
    trans_indx = ladder_transition_indices(dim, -1)
    root = [ sqrt(n) for n in xrange(ns + 1) ]
    def update_d_red(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the derivative of the 1st red transition."""
        matrix[phase_indx] = [ -root[n + z] * sin(root[n + z] * angle)
                               for z in (1, 0) for n in xrange(ns) ]
        matrix[trans_indx] = [ -1.0j * root[n] * cos(root[n] * angle)
                               for _ in (0, 1) for n in xrange(1, ns) ]
    return update_d_red

def generate_blue_updater(matrix):
    """
    Create a function which will update the given matrix representing a 1st blue
    transition, when given a new angle.
    """
    dim = matrix.shape[0]
    ns = dim // 2
    phase_indx = np.diag_indices(dim)
    trans_indx = ladder_transition_indices(dim, 1)
    root = [ sqrt(n) for n in xrange(ns + 1) ]
    def update_blue(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the 1st blue transition."""
        matrix[phase_indx] = [ cos(root[n + z] * angle)
                               for z in (0, 1) for n in xrange(ns) ]
        matrix[trans_indx] = [ -1.0j * sin(root[n] * angle)
                               for _ in (0, 1) for n in xrange(1, ns) ]
    return update_blue

def generate_d_blue_updater(matrix):
    """
    Create a function which will update the given matrix representing the
    derivative of a 1st blue transition, when given a new angle.
    """
    dim = matrix.shape[0]
    ns = dim // 2
    phase_indx = np.diag_indices(dim)
    trans_indx = ladder_transition_indices(dim, 1)
    root = [ sqrt(n) for n in xrange(ns + 1) ]
    def update_d_blue(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the derivative of the 1st blue transition."""
        matrix[phase_indx] = [ -root[n + z] * sin(root[n + z] * angle)
                               for z in (0, 1) for n in xrange(ns) ]
        matrix[trans_indx] = [ -1.0j * root[n] * cos(root[n] * angle)
                               for _ in (0, 1) for n in xrange(1, ns) ]
    return update_d_blue

def make_ground_state(populated, ns):
    """
    Return a 1D np.array of complex of length 2 * ns, which represents a ket of
    a normalised ground state, with the `populated` motional levels populated in
    equal amounts at the same phase.  For example,
        make_ground_state([0, 2, 3], 5)
    will produce
        array([0, 0, 0, 0, 0, 1/sqrt(3), 0, 1/sqrt(3), 1/sqrt(3), 0]).
              |---excited---|----------------ground----------------|
    """
    assert len(populated) > 0,\
        "There must be at least one populated motional state."
    out = np.zeros(2 * ns, dtype = np.complex128)
    for i in populated:
        out[ns + i] = 1.0 / sqrt(len(populated))
    return out

def motional_states_needed(colours):
    """
    Count up the number of motional states we need to consider for full accuracy
    for a given set of colours.  The red and the blue sidebands can both force
    us to consider a new motional mode, but the carrier just transitions between
    existing ones.

    This function can overestimate for some pulse sequences, but those are ones
    which have a ['r', 'r'] or similar, which ought to be described by a single
    'r' of different length.  Also, a pulse sequence starting with 'r' and in
    the |g0> state will consider one too many states, but this is also a silly
    pulse sequence, because it doesn't do anything.
    """
    return reduce(lambda acc, c: acc + {'c':0, 'r':1, 'b':1}[c], colours, 1)


# Computational classes

class ColourOperator(object):
    def __init__(this, colour, ns):
        this.op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)
        this.d_op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)
        this._updater = {
            'c': generate_carrier_updater,
            'r': generate_red_updater,
            'b': generate_blue_updater }[colour](this.op)
        this._d_updater = {
            'c': generate_d_carrier_updater,
            'r': generate_d_red_updater,
            'b': generate_d_blue_updater }[colour](this.d_op)
        this._colour = colour
        this._angle = None

    @property
    def angle(this):
        return this._angle
    @angle.setter
    def angle(this, new_angle):
        this._angle = new_angle
        this._updater(pi * this._angle)
        this._d_updater(pi * this._angle)

class PulseSequence(object):
    """
    Arguments:
    colours: 1D list of 'c' | 'r' | 'b' --
        The sequence of coloured pulses to apply, with 'c' being the carrier,
        'r' being the first red sideband and 'b' being the first blue sideband.
        If the array looks like
            ['r', 'b', 'c'],
        then the carrier will be applied first, then the blue, then the red,
        similar to how we'd write them in operator notation.

    target (optional, None): 1D list of unsigned int --
        The motional levels that should be populated in the ground state after
        the pulse sequence is completed.  These will be equally populated, and
        in the same phase as each other.  For example, `[0, 2, 3]` corresponds
        to a target state `(|g0> + |g2> + |g3>)/sqrt(3)`.

    start (optional, [0]): 1D list of unsigned int --
        The motional levels that should be populated in the ground state at the
        beginning of the pulse sequence.  These will be equally populated, and
        in the same phase as each other.  For example, `[0, 2, 3]` corresponds
        to a start state `(|g0> + |g2> + |g3>)/sqrt(3)`.

    use_cache (optional, True): boolean --
        Whether to cache all the calculation results for all angles.  If `True`,
        calling `this.U` or other calculation functions will store the results,
        so they won't need to be recalculated if you use a different set of
        angles, then come back.  If this is `False`, then only the most recent
        set of angles will be stored---this is preferable for optimisation runs,
        particularly with superpositions of large n.
    """
    def __init__(this, colours, target = None, start = [0]):
        this.colours = colours
        this._len = len(this.colours)
        this._ns = max(motional_states_needed(colours),
                       max(target) + 1, max(start) + 1)
        this._target = make_ground_state(target, this._ns)\
                       if target is not None else None
        this._start = make_ground_state(start, this._ns)
        this._lin_ops =\
            np.array([ ColourOperator(c, this._ns) for c in this.colours ])
        this._angles = None

        # Output storage.
        this._u = np.empty((2 * this._ns, 2 * this._ns), dtype = np.complex128)
        this._d_u = np.empty((this._len, 2 * this._ns, 2 * this._ns),
                             dtype = np.complex128)
        this._dist = float("inf")
        this._d_dist = np.zeros(this._len, dtype = np.float64)

        # Pre-allocate calculation scratch space and fixed variables.
        this._id = np.identity(2 * this._ns, dtype = np.complex128)
        this._partials_ltr = np.empty_like(this._d_u)
        this._partials_rtl = np.empty_like(this._d_u)
        this._partials_ltr[0] = this._id
        this._partials_rtl[0] = this._id
        this._temp = np.empty_like(this._u)

    def _update_propagator_and_derivatives(this):
        """
        Efficient method of calculating the complete propagator, and all the
        derivatives associated with it.

        Arguments:
        pulses: 1D list of (colour * angle) --
            colour: 'c' | 'r' | 'b' --
                The colour of the pulse to be applied, 'c' is the carrier, 'r' is
                the first red sideband and 'b' is the first blue sideband.

            angle: double --
                The angle of specified pulse divided by pi, e.g. `angle = 0.5`
                corresponds to the pulse being applied for an angle of `pi / 2`.

            A list of the pulses to apply, at the given colour and angle.

        ns: unsigned --
            The number of motional states to consider when building the matrices.
            Note this is not the maximum motional state - the max will be |ns - 1>.

        Returns:
        propagator: 2D complex numpy.array --
            The full propagator for the chain of operators, identical to calling
            `multi_propagator(colours)(angles)`.

        derivatives: 1D list of (2D complex numpy.array) --
            A list of the derivatives of the propagator at the specified angles,
            with respect to each of the given angles in turn.
        """
        for i in xrange(this._len - 1):
            np.dot(this._partials_ltr[i], this._lin_ops[i].op,
                   out = this._partials_ltr[i + 1])
            np.dot(this._lin_ops[-(i + 1)].op, this._partials_rtl[i],
                   out = this._partials_rtl[i + 1])

        np.dot(this._partials_ltr[-1], this._lin_ops[-1].op, out = this._u)
        for i in xrange(this._len):
            np.dot(this._partials_ltr[i], this._lin_ops[i].d_op,
                   out = this._temp)
            np.dot(this._temp, this._partials_rtl[-(i + 1)], out = this._d_u[i])

    def _update_distance_and_derivatives(this):
        tus = inner_product(this._target, this._u, this._start)
        tus_conj = np.conj(tus)

        this._dist = 1.0 - (tus.real * tus.real + tus.imag * tus.imag)
        for i in xrange(this._len):
            prod = inner_product(this._target, this._d_u[i], this._start)
            this._d_dist[i] = -2.0 * (tus_conj * prod).real

    def _calculate_all(this, angles):
        assert len(angles) == this._len,\
            "There are {} colours in the sequence, but I got {} angles."\
            .format(this._len, len(angles))
        if np.array_equal(this._angles, angles):
            return
        this._angles = angles
        for i in xrange(len(this._angles)):
            this._lin_ops[i].angle = this._angles[i]
        this._update_propagator_and_derivatives()
        this._update_distance_and_derivatives()

    def U(this, angles):
        """
        Get the propagator of the pulse sequence stored in the class with the
        specified angles.
        """
        this._calculate_all(angles)
        return this._u

    def d_U(this, angles):
        """
        Get the derivatives of the propagator of the pulse sequence stored in
        the class with the specified angles.
        """
        this._calculate_all(angles)
        return this._d_u

    def distance(this, angles):
        """
        Get the distance of the pulse sequence stored in the class with the
        specified angles.
        """
        assert this._target is not None,\
            "You must set the target state to calculate the distance."
        this._calculate_all(angles)
        return this._dist

    def d_distance(this, angles):
        """
        Get the derivatives of the distance of the pulse sequence stored in the
        class with the specified angles.
        """
        assert this._target is not None,\
            "You must set the target state to calculate the distance."
        this._calculate_all(angles)
        return np.copy(this._d_dist)

    def optimise(this, initial_angles):
        assert this._target is not None,\
            "You must set the target state to optimise a pulse sequence."
        return scipy.optimize.minimize(this.distance, initial_angles,
                                       jac = this.d_distance,
                                       method = 'BFGS')
