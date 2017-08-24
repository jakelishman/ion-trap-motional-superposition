import numpy as np
import scipy.optimize
from math import *
from functional import *

# Details about elements of linear operator matrices.

def is_excited(row, col, ns):
    """True iff the position in the matrix is in the |e><e| block."""
    return col < ns and row < ns
def is_ground(row, col, ns):
    """True iff the position in the matrix is in the |g><g| block."""
    return col >= ns and row >= ns
def is_sigma_plus(row, col, ns):
    """True iff the position in the matrix is in the |e><g| block."""
    return col >= ns and row < ns
def is_sigma_minus(row, col, ns):
    """True iff the position in the matrix is in the |g><e| block."""
    return row >= ns and col < ns
def is_internal_transition(row, col, ns):
    """True iff the position in the matrix is in the |g><e| + |e><g| block."""
    return is_sigma_plus(row, col, ns) or is_sigma_minus(row, col, ns)
def start_n(_, col, ns):
    """If the matrix element is |m><n|, return the value `n`."""
    return col % ns
def end_n(row, _, ns):
    """If the matrix element is |m><n|, return the value `m`."""
    return row % ns
def is_ladder(ladder):
    """True iff the element's Fock part satisfies |n + ladder><n|, i.e.
    is_ladder(0) is true if it's an |n><n| element, is_ladder(1) is true if it's
    a creation state and is_ladder(-1) is true if it's an annihilation."""
    return lambda row, col, ns:\
        end_n(row, col, ns) - start_n(row, col, ns) is ladder


# Functions on linear operators

def adj(arr):
    """Adjoint of a matrix."""
    return np.conj(arr.T)
def inner_product(final, operator, start):
    """
    Calculates the quantity <final|operator|start>, assuming that both final and
    start are given as kets, represented by 1D numpy.arrays of complex.
    """
    return np.linalg.multi_dot([adj(final), operator, start])


# Building blocks of linear operators for sideband transitions

def c_el(angle, row, col, ns):
    """
    Return the element of the propagator matrix for the carrier signal in the
    LD regime when applied for an angle `angle` in the interaction picture.
    `ns` is the number of Fock states present in the matrix.
    """
    if is_ladder(0)(row, col, ns):
        if is_internal_transition(row, col, ns):
            return -1.0j * sin(angle)
        else: # |ej><ej| or |gj><gj|
            return cos(angle)
    else:
        return 0.0

def d_c_el(angle, row, col, ns):
    """
    Return the element of the derivative of the propagator matrix for the
    carrier signal w.r.t the angle in the LD regime when applied for an angle
    `angle` in the interaction picture.  `ns` is the number of Fock states
    present in the matrix.
    """
    # c_el is only sin or cos, and d(sin(x))/d(x) = sin(x + pi/2), and similar
    # for cos.
    return c_el(angle + pi / 2, row, col, ns)

def r_el(angle, row, col, ns):
    """
    Return the element of the propagator matrix for the red sideband signal in
    the LD regime when applied for an angle `angle` in the interaction picture.
    `ns` is the number of Fock states present in the matrix.
    """
    if is_excited(row, col, ns) and is_ladder(0)(row, col, ns):
        return cos(sqrt(start_n(row, col, ns) + 1) * angle)
    elif is_ground(row, col, ns) and is_ladder(0)(row, col, ns):
        return cos(sqrt(start_n(row, col, ns)) * angle)
    elif is_sigma_plus(row, col, ns) and is_ladder(-1)(row, col, ns):
        return -1.0j * sin(sqrt(start_n(row, col, ns)) * angle)
    elif is_sigma_minus(row, col, ns) and is_ladder(1)(row, col, ns):
        return -1.0j * sin(sqrt(start_n(row, col, ns) + 1) * angle)
    else:
        return 0.0

def d_r_el(angle, row, col, ns):
    """
    Return the element of the derivative of the propagator matrix for the
    red sideband signal w.r.t the angle in the LD regime when applied for an
    angle `angle` in the interaction picture.  `ns` is the number of Fock states
    present in the matrix.
    """
    if is_excited(row, col, ns) and is_ladder(0)(row, col, ns):
        factor = sqrt(start_n(row, col, ns) + 1)
        return -factor * sin(factor * angle)
    elif is_ground(row, col, ns) and is_ladder(0)(row, col, ns):
        factor = sqrt(start_n(row, col, ns))
        return -factor * sin(factor * angle)
    elif is_sigma_plus(row, col, ns) and is_ladder(-1)(row, col, ns):
        factor = sqrt(start_n(row, col, ns))
        return factor * -1.0j * cos(factor * angle)
    elif is_sigma_minus(row, col, ns) and is_ladder(1)(row, col, ns):
        factor = sqrt(start_n(row, col, ns) + 1)
        return factor * -1.0j * cos(factor * angle)
    else:
        return 0.0

def b_el(angle, row, col, ns):
    """
    Return the element of the propagator matrix for the blue sideband signal in
    the LD regime when applied for an angle `angle` in the interaction picture.
    `ns` is the number of Fock states present in the matrix.
    """
    if is_excited(row, col, ns) and is_ladder(0)(row, col, ns):
        return cos(sqrt(start_n(row, col, ns)) * angle)
    elif is_ground(row, col, ns) and is_ladder(0)(row, col, ns):
        return cos(sqrt(start_n(row, col, ns) + 1) * angle)
    elif is_sigma_plus(row, col, ns) and is_ladder(1)(row, col, ns):
        return -1.0j * sin(sqrt(start_n(row, col, ns) + 1) * angle)
    elif is_sigma_minus(row, col, ns) and is_ladder(-1)(row, col, ns):
        return -1.0j * sin(sqrt(start_n(row, col, ns)) * angle)
    else:
        return 0.0

def d_b_el(angle, row, col, ns):
    """
    Return the element of the derivative of the propagator matrix for the
    blue sideband signal w.r.t the angle in the LD regime when applied for an
    angle `angle` in the interaction picture.  `ns` is the number of Fock states
    present in the matrix.
    """
    if is_excited(row, col, ns) and is_ladder(0)(row, col, ns):
        factor = sqrt(start_n(row, col, ns))
        return -factor * sin(factor * angle)
    elif is_ground(row, col, ns) and is_ladder(0)(row, col, ns):
        factor = sqrt(start_n(row, col, ns) + 1)
        return -factor * sin(factor * angle)
    elif is_sigma_plus(row, col, ns) and is_ladder(1)(row, col, ns):
        factor = sqrt(start_n(row, col, ns) + 1)
        return factor * -1.0j * cos(factor * angle)
    elif is_sigma_minus(row, col, ns) and is_ladder(-1)(row, col, ns):
        factor = sqrt(start_n(row, col, ns))
        return factor * -1.0j * cos(factor * angle)
    else:
        return 0.0

def update_matrix(matrix, gen, angle, ns):
    """
    Internal function.
    Build a matrix at the specified angle using the given generator function.
    """
    for row in xrange(2 * ns):
        for col in xrange(2 * ns):
            matrix[row][col] = gen(pi * angle, row, col, ns)

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
        this._gen = {'c':c_el, 'r':r_el, 'b':b_el}[colour]
        this._d_gen = {'c':d_c_el, 'r':d_r_el, 'b':d_b_el}[colour]
        this._ns = ns
        this._angle = None
        this.op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)
        this.d_op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)

    @property
    def angle(this):
        return this._angle
    @angle.setter
    def angle(this, new_angle):
        this._angle = new_angle
        update_matrix(this.op, this._gen, this._angle, this._ns)
        update_matrix(this.d_op, this._d_gen, this._angle, this._ns)

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
