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
    return np.dot(adj(final), np.dot(operator, start))


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

def build_matrix(gen, angle, ns):
    """
    Internal function.
    Build a matrix at the specified angle using the given generator function.
    """
    return np.array([
        [ gen(pi * angle, row, col, ns) for col in xrange(2 * ns) ]
        for row in xrange(2 * ns) ])

def prop_generator(colour, ns = 10):
    """
    Return a function generating a matrix for the propagator of the specified
    colour.  This must be one of 'c', 'r' or 'b', for carrier, red and blue
    respectively.
    """
    return lambda angle:\
        build_matrix({'c':c_el, 'r':r_el, 'b':b_el}[colour], angle, ns)

def deriv_generator(colour, ns = 10):
    """
    Return a function generating a matrix of the derivative of a propagator of
    the specified colour.  The colour must be one of 'c', 'r' or 'b', for
    carrier, red or blue respectively.
    """
    return lambda angle:\
        build_matrix({'c':d_c_el, 'r':d_r_el, 'b':d_b_el}[colour], angle, ns)

def multi_propagator(colours, ns = 10):
    """
    Return a function which takes a list of angles, and returns the matrix-form
    propagator which corresponds to a series of propagators at the given angles.
    The input list to this function is
        [ colour_1, colour_2, ..., colour_n ],
    and the input list to the output funciton is
        [ angle_1, angle_2, ..., angle_n ]
    which corresponds to the operator
        U_1(pi * angle_1) . U_2(pi * angle_2) ... U_n(pi * angle_n),
    i.e. the last element of the list is the operator applied first.

    The colours must each be one of 'c', 'r' or 'b', which correspond to the
    carrier, red and blue sidebands respectively.
    """
    return reduce(np.dot, map(lambda tup: prop_generator(*tup, ns = ns), lst))

def propagator_and_derivatives(pulses, ns = 10):
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
    iden = np.identity(2 * ns, dtype = np.complex128)
    props  = map(lambda (c, a): prop_generator(c, ns = ns)(a), pulses)
    """List of single propagator matrices."""
    derivs = map(lambda (c, a): deriv_generator(c, ns = ns)(a), pulses)
    """List of the derivatives of the single propagator matrices."""
    part_f = scan(np.dot, props)
    part_b = scan_back(np.dot, props)
    propagator = part_f[-1]
    part_f = part_f[:-1]
    part_f.insert(0, iden) # [id, prop_0, ..., prop_0 . ... . prop_{n-1}]
    part_b = part_b[1:]
    part_b.append(iden) # [prop_1 . ... . prop_n, ..., prop_n, id]

    derivatives = np.array(map3(lambda *args: reduce(np.dot, args),
                                part_f, derivs, part_b))
    return propagator, derivatives

def distance_and_derivatives(start, target, propagator, propagator_derivatives):
    """
    More efficient method of calculating the infidelity of
        <target| propagator |start>
    and its derivatives with respect to each of the pulses' angles.
    """
    tus = inner_product(target, propagator, start)
    # tus stands for <target|U|start>
    tus_conj = tus.real - tus.imag * 1.0j
    distance = 1.0 - (tus * tus_conj).real
    def single_derivative(d_op):
        """
        Get the derivative with respect to one of the parameters.  This is
            (1 - |<t|U'|s>|^2)'
            = - (<s|adj(U)'|t><t|U|s> + <s|adj(U)|t><t|U'|s>)
            = - 2 * Re(adj(<t|U|s>) * <t|U'|s>)
        because adj(U)' = adj(U') for unitary operators.  This means we only
        have to do one additional matrix multiplication, which beats the naive
        implementation by a couple of multiplications and doesn't require the
        creation of any new matrices.  We also reuse `tus`.
        """
        return -2.0 * (tus_conj * inner_product(target, d_op, start)).real
    derivatives = np.array(map(single_derivative, propagator_derivatives))
    return distance, derivatives

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
    """
    def __init__(this, colours, target = None, start = [0]):
        this._cache = {}
        this._colours = colours
        this._ns = max(motional_states_needed(colours),
                       max(target) + 1, max(start) + 1)
        this._target = make_ground_state(target, this._ns)\
                       if target is not None else None
        this._start = make_ground_state(start, this._ns)

    @property
    def start(this):
        return this._start
    @start.setter
    def start(this, new_start):
        if not np.array_equal(this._start, new_start):
            this._cache = {}
            this._start = new_start

    @property
    def target(this):
        return this._target
    @target.setter
    def target(this, new_target):
        if not np.array_equal(this._target, new_target):
            this._cache = {}
            this._target = new_target

    def _calculate_all(this, angles):
        prop, d_prop =\
                propagator_and_derivatives(zip(this._colours, angles), this._ns)
        if this.target is not None:
            dist, d_dist = distance_and_derivatives(this.start, this.target,
                                                    prop, d_prop)
            this._cache[angles] = {
                "propagator" : prop,
                "propagator derivatives" : d_prop,
                "distance" : dist,
                "distance derivatives" : d_dist }
        else:
            this._cache[angles] = {
                "propagator" : prop,
                "propagator derivatives" : d_prop }

    def _lookup_or_calculate(this, angles, key):
        assert len(angles) == len(this._colours),\
            "The sequence is {} colours long, but there were {} angles."\
            .format(len(this._colours), len(angles))
        angles_ = tuple(angles)
        if angles_ not in this._cache:
            this._calculate_all(angles_)
        return this._cache[angles_][key]

    def U(this, angles):
        """
        Get the propagator of the pulse sequence stored in the class with the
        specified angles.
        """
        return this._lookup_or_calculate(angles, "propagator")

    def d_U(this, angles):
        """
        Get the derivatives of the propagator of the pulse sequence stored in
        the class with the specified angles.
        """
        return this._lookup_or_calculate(angles, "propagator derivatives")

    def distance(this, angles):
        """
        Get the distance of the pulse sequence stored in the class with the
        specified angles.
        """
        assert this.target is not None,\
            "You must set the target state to calculate the distance."
        return this._lookup_or_calculate(angles, "distance")

    def d_distance(this, angles):
        """
        Get the derivatives of the distance of the pulse sequence stored in the
        class with the specified angles.
        """
        assert this.target is not None,\
            "You must set the target state to calculate the distance."
        return this._lookup_or_calculate(angles, "distance derivatives")

    def optimise(this, initial_angles):
        assert this.target is not None,\
            "You must set the target state to optimise a pulse sequence."
        return scipy.optimize.minimize(this.distance, initial_angles,
                                       jac = this.d_distance,
                                       method = 'BFGS')
