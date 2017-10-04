from math import cos, sin, sqrt, pi
from cmath import exp as cexp
from functools import reduce
import ion_superpositions.state_specifier as state
import numpy as np

def adj(arr):
    """Adjoint of a matrix."""
    return np.conj(arr.T)
def inner_product(final, operator, start):
    """
    Calculates the quantity <final|operator|start>, assuming that both final and
    start are given as kets, represented by 1D numpy.arrays of complex.
    """
    return adj(final).dot(operator).dot(start)

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
    root = [ sqrt(n) for n in range(ns + 1) ]
    def update_red(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the 1st red transition."""
        matrix[phase_indx] = [ cos(root[n + z] * angle)
                               for z in (1, 0) for n in range(ns) ]
        matrix[trans_indx] = [ z * sin(root[n] * angle)
                               for z in (1, -1) for n in range(1, ns) ]
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
    root = [ sqrt(n) for n in range(ns + 1) ]
    def update_d_red(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the derivative of the 1st red transition."""
        matrix[phase_indx] = [ -root[n + z] * sin(root[n + z] * angle)
                               for z in (1, 0) for n in range(ns) ]
        matrix[trans_indx] = [ z * root[n] * cos(root[n] * angle)
                               for z in (1, -1) for n in range(1, ns) ]
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
    root = [ sqrt(n) for n in range(ns + 1) ]
    def update_blue(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the 1st blue transition."""
        matrix[phase_indx] = [ cos(root[n + z] * angle)
                               for z in (0, 1) for n in range(ns) ]
        matrix[trans_indx] = [ z * sin(root[n] * angle)
                               for z in (1, -1) for n in range(1, ns) ]
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
    root = [ sqrt(n) for n in range(ns + 1) ]
    def update_d_blue(angle):
        """Update the 3 relevant diagonals in the closed-over matrix with a new
        angle, representing the derivative of the 1st blue transition."""
        matrix[phase_indx] = [ -root[n + z] * sin(root[n + z] * angle)
                               for z in (0, 1) for n in range(ns) ]
        matrix[trans_indx] = [ z * root[n] * cos(root[n] * angle)
                               for z in (1, -1) for n in range(1, ns) ]
    return update_d_blue

def build_state_vector(populated, ns):
    """
    Return a 1D np.array of complex of length 2 * ns, which represents a ket of
    a normalised state, with the `populated` motional levels populated in equal
    amounts.  For example,
        build_state_vector([0, (2, 'e', 0.5), 3], 5)
    will produce
        array([0, 0, i/sqrt(3), 0, 0, 1/sqrt(3), 0, 0, 1/sqrt(3), 0]).
              |-------excited-------|------------ground------------|

    Arguments:
    populated: 1D list of state_specifier --
        The states to be populated in equal amounts in the output state.
        
    ns: unsigned int --
        The number of motional levels which are being considered.  The output
        vector will have length `2 * ns`.
    """
    assert len(populated) > 0,\
        "There must be at least one populated motional state."
    out = np.zeros(2 * ns, dtype = np.complex128)
    n = 1.0 / sqrt(len(populated))
    for ss in populated:
        out[state.idx(ss, ns)] = cexp(1.0j * pi * state.phase(ss)) * n
    return out

def motional_states_needed(colours):
    """
    Count up the number of motional states we need to consider for full accuracy
    for a given set of colours.  The red and the blue sidebands can both force
    us to consider a new motional mode, but the carrier just transitions between
    existing ones.

    self function can overestimate for some pulse sequences, but those are ones
    which have a ['r', 'r'] or similar, which ought to be described by a single
    'r' of different length.  Also, a pulse sequence starting with 'r' and in
    the |g0> state will consider one too many states, but self is also a silly
    pulse sequence, because it doesn't do anything.
    """
    return reduce(lambda acc, c: acc + {'c':0, 'r':1, 'b':1}[c], colours, 1)

class ColourOperator(object):
    def __init__(self, colour, ns):
        self.op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)
        self.d_op = np.zeros((2 * ns, 2 * ns), dtype = np.complex128)
        self.__updater = {
            'c': generate_carrier_updater,
            'r': generate_red_updater,
            'b': generate_blue_updater }[colour](self.op)
        self.__d_updater = {
            'c': generate_d_carrier_updater,
            'r': generate_d_red_updater,
            'b': generate_d_blue_updater }[colour](self.d_op)
        self.colour = colour
        self.__angle = None

    @property
    def angle(self):
        return self.__angle
    @angle.setter
    def angle(self, new_angle):
        if self.__angle != new_angle:
            self.__angle = new_angle
            self.__updater(pi * self.__angle)
            self.__d_updater(pi * self.__angle)
        return

    def U(self, angle):
        """ColourOperator().U(angle)

        Set the angle in the ColourOperator, and then return a copy of the
        operator matrix at that angle."""
        self.angle = angle
        return np.copy(self.op)

    def d_U(self, angle):
        """ColourOperator().d_U(angle)

        Set the angle, then return a copy of the derivative of the operator
        matrix at that angle."""
        self.angle = angle
        return np.copy(self.d_op)
