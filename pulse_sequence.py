"""
Supplementary types:
state_specifier: (motional * ?internal * ?phase) --
    motional: unsigned int --
        The motional level of the state, e.g. 0, 2, 3 etc.
    internal (optional, 'g'): 'g' | 'e' --
        The internal part of the state, either ground or excited.  Defaults to
        ground.
    phase (optional, 0.0): double --
        The relative phase of self state as an angle, divided by pi.  The
        resulting phase will be `e^(i * pi * phase)`, for example:
            0.0 => 1.0 + 0.0i,
            0.5 => 0.0 + 1.0i,
            1.0 => -1.0 + 0.0i,
            3.0 => -1.0 + 0.0i.

    Access self type using functions from state_specifier.py.
"""

from cmath import exp as cexp
from itertools import chain
from functools import reduce
from random import SystemRandom
import math
import state_specifier as state
import pulse_matrices as pm
import numpy as np
import scipy.optimize

def random_array(shape, lower=0.0, upper=1.0, **kwargs):
    """random_array(shape, lower=0.0, upper=1.0, **kwargs) -> array

    Return a np.array of `shape` with the elements filled with uniform random
    values between `lower` and `upper`."""
    rand = SystemRandom()
    length = reduce(lambda acc, x: acc * x, shape, 1)\
             if isinstance(shape, tuple) else shape
    rands = [rand.uniform(lower, upper) for _ in range(length)]
    return np.array(rands, **kwargs).reshape(shape)

def _format_complex(z, n_digits):
    z = round(z.real, n_digits) + round(z.imag, n_digits) * 1.0j
    if z.imag == 0:
        return str(z.real)
    elif z.real == 0:
        return "{}i".format(z.imag)
    return "({0} {2} {1}i)".format(z.real, abs(z.imag),
                                   {-1: '-', 1: '+'}[np.sign(z.imag)])

class PulseSequence(object):
    """
    Arguments:
    colours: 1D list of ('c' | 'r' | 'b') --
        The sequence of coloured pulses to apply, with 'c' being the carrier,
        'r' being the first red sideband and 'b' being the first blue sideband.
        If the array looks like
            ['r', 'b', 'c'],
        then the carrier will be applied first, then the blue, then the red,
        similar to how we'd write them in operator notation.

    target (optional, None): 1D list of state_specifier --
        The states which should be populated with equal probabilities after the
        pulse sequences is completed.  The relative phases are allowed to vary
        if `fixed_phase` is False.  For example, `[0, (2, 'e', 0.5), 3]` and
        `fixed_phase == False` corresponds to a target state
        `(|g0> + e^{ia}|e2> + e^{ib}|g3>)/sqrt(3)` with variable `a` and `b`
        (i.e. the choice of phase of the |e2> state is ignored).

        If not set, all functions which require a target will throw
        `AssertionError`.

    fixed_phase (optional, False): Boolean --
        Whether the phases of the target should be fixed during fidelity,
        distance and optimisation calculations.

    start (optional, [0]): 1D list of state_specifier --
        The motional levels that should be populated in the ground state at the
        beginning of the pulse sequence.  These will be equally populated, for
        example, `[0, (2, 'e', 0.5), 3]` corresponds to a start state of
        `(|g0> + i|e2> + |g3>)/sqrt(3)`.
    """
    def __init__(self, colours, target=None, fixed_phase=False, start=None):
        assert colours,\
            "You must have at least one colour in the sequence!"
        start = start if start is not None else [(0, 'g', 0.0)]
        self.colours = colours
        self._len = len(self.colours)
        self._n_phases = len(target) - 1 if target is not None else None
        self._ns = max(pm.motional_states_needed(colours),
                       max(list(map(state.motional, start))) + 1,
                       max(list(map(state.motional, target))) + 1\
                       if target is not None else 0)
        self.fixed_phase = fixed_phase or self._n_phases is 0
        self._orig_target = target
        self._target = pm.build_state_vector(target, self._ns)\
                       if target is not None else None
        self._start = pm.build_state_vector(start, self._ns)
        self._lin_ops =\
            np.array([pm.ColourOperator(c, self._ns) for c in self.colours])
        # The angles are for every colour in the sequence, read in left-to-right
        # order.  The angles are all divided by pi.
        self._angles = None
        # The stored phases are for every element of the target except the
        # first.  The first is always taken to be 0, and the others are rotated
        # so that self is true (if a first phase is supplied).  The phases are
        # stored as angles, divided by pi (e.g. 0 => 1, 0.5 => i, 1 => -1 etc).
        self._phases = None

        # Output storage.
        self._u = np.empty((2 * self._ns, 2 * self._ns), dtype=np.complex128)
        self._d_u = np.empty((self._len, 2 * self._ns, 2 * self._ns),
                             dtype=np.complex128)
        if target is not None:
            # self shouldn't be a np.array because it needs to be able to hold
            # state_specifier tuples which are variable length.
            self._new_target = [0 for _ in self._orig_target]
            self._dist = float("inf")
            self._d_dist_angles = np.empty(self._len, dtype=np.float64)
            self._d_dist_phases = np.empty(len(target) - 1, dtype=np.float64)

        # Pre-allocate calculation scratch space and fixed variables.
        self._id = np.identity(2 * self._ns, dtype=np.complex128)
        self._partials_ltr = np.empty_like(self._d_u)
        self._partials_rtl = np.empty_like(self._d_u)
        self._partials_ltr[0] = self._id
        self._partials_rtl[0] = self._id
        self._temp = np.empty_like(self._u)
        self._tus = 0.0j

    def _update_target_phases(self):
        self._new_target[0] = state.set_phase(self._orig_target[0], 0)
        for i, el in enumerate(self._orig_target[1:]):
            self._new_target[i + 1] = state.set_phase(el, self._phases[i])
        self._target = pm.build_state_vector(self._new_target, self._ns)

    def _update_propagator_and_derivatives(self):
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
            Note self is not the maximum motional state - the max will be |ns - 1>.

        Returns:
        propagator: 2D complex numpy.array --
            The full propagator for the chain of operators, identical to calling
            `multi_propagator(colours)(angles)`.

        derivatives: 1D list of (2D complex numpy.array) --
            A list of the derivatives of the propagator at the specified angles,
            with respect to each of the given angles in turn.
        """
        for i in range(self._len - 1):
            np.dot(self._partials_ltr[i], self._lin_ops[i].op,
                   out=self._partials_ltr[i + 1])
            np.dot(self._lin_ops[-(i + 1)].op, self._partials_rtl[i],
                   out=self._partials_rtl[i + 1])

        np.dot(self._partials_ltr[-1], self._lin_ops[-1].op, out=self._u)
        for i in range(self._len):
            np.dot(self._partials_ltr[i], self._lin_ops[i].d_op,
                   out=self._temp)
            np.dot(self._temp, self._partials_rtl[-(i + 1)], out=self._d_u[i])

    def _update_distance(self):
        self._tus = pm.inner_product(self._target, self._u, self._start)
        self._dist = 1.0 - (self._tus * np.conj(self._tus)).real

    def _update_distance_angle_derivatives(self):
        for i in range(self._len):
            prod = pm.inner_product(self._target, self._d_u[i], self._start)
            self._d_dist_angles[i] = -2.0 * (np.conj(self._tus) * prod).real

    def _update_distance_phase_derivatives(self):
        pref = 2.0 / math.sqrt(len(self._orig_target))
        u_start = np.dot(self._u, self._start)
        for i, pre_phase in enumerate(self._phases):
            phase = cexp(1.0j * math.pi * (0.5 - pre_phase))
            # we can calculate the inner product <g n_j|U|start> by
            # precalculating U|start>, then indexing to the relevant element.
            idx = state.idx(self._orig_target[i + 1], self._ns)
            self._d_dist_phases[i] =\
                pref * (phase * u_start[idx] * np.conj(self._tus)).real

    def _update_angles_if_required(self, angles):
        if angles is None:
            return False
        assert len(angles) == self._len,\
            "There are {} colours in the sequence, but I got {} angles."\
            .format(self._len, len(angles))
        if np.array_equal(self._angles, angles):
            return False
        self._angles = angles
        for i in range(len(self._angles)):
            self._lin_ops[i].angle = self._angles[i]
        return True

    def _update_phases_if_required(self, phases):
        if self.fixed_phase\
           or phases is None\
           or np.array_equal(self._phases, phases):
            return False
        assert 0 <= len(self._orig_target) - len(phases) <= 1,\
            "There are {} elements of the target, but I got {} phases."\
            .format(len(self._orig_target), len(phases))
        if len(phases) == len(self._orig_target) - 1:
            self._phases = phases
        else:
            self._phases = np.vectorize(lambda x: x - phases[0])(phases[1:])
        self._update_target_phases()
        return True

    def _calculate_propagator(self, angles):
        if self._update_angles_if_required(angles):
            self._update_propagator_and_derivatives()

    def _calculate_all(self, angles, phases=None):
        assert self._target is None or self.fixed_phase or phases is not None,\
            "If you're not in fixed phase mode, you need to specify the phases."
        updated_angles = self._update_angles_if_required(angles)
        updated_phases = self._update_phases_if_required(phases)
        if not (updated_angles or updated_phases):
            return
        if updated_angles:
            self._update_propagator_and_derivatives()
        if self._target is not None:
            self._update_distance()
            if updated_angles:
                self._update_distance_angle_derivatives()
            if updated_phases:
                self._update_distance_phase_derivatives()

    def U(self, angles):
        """
        Get the propagator of the pulse sequence stored in the class with the
        specified angles.
        """
        self._calculate_propagator(angles)
        return np.copy(self._u)

    def d_U(self, angles):
        """
        Get the derivatives of the propagator of the pulse sequence stored in
        the class with the specified angles.
        """
        self._calculate_propagator(angles)
        return np.copy(self._d_u)

    def distance(self, angles, phases=None):
        """
        Get the distance of the pulse sequence stored in the class with the
        specified angles.
        """
        assert self._target is not None,\
            "You must set the target state to calculate the distance."
        self._calculate_all(angles, phases)
        return self._dist

    def d_distance(self, angles, phases=None):
        """
        Get the derivatives of the distance of the pulse sequence stored in the
        class with the specified angles (and phases of the target state, if
        applicable).

        Outputs (angles * phases), with each being a 1D np.array.  The pulse
        angles are in left-to-right order (i.e. for "rcb", the order is
        [r, c, b]), and the phases are in the order that the elements of the
        target were given, excluding the first element of the target, which is
        assumed to maintain a phase of 1.
        """
        assert self._target is not None,\
            "You must set the target state to calculate the distance."
        assert self.fixed_phase or phases is not None,\
            "If you're not in fixed phase mode, you need to specify the phases."
        self._calculate_all(angles, phases)
        if self.fixed_phase:
            return np.copy(self._d_dist_angles)
        return np.copy(self._d_dist_angles), np.copy(self._d_dist_phases)

    def optimise(self, initial_angles=None, initial_phases=None, **kwargs):
        """optimise(initial_angles=None, initial_phases=None, **kwargs)"""
        assert self._target is not None,\
            "You must set the target state to optimise a pulse sequence."
        angles = random_array(self._len, dtype=np.float64)\
                 if initial_angles is None else initial_angles
        if not self.fixed_phase:
            phases = random_array(self._n_phases, dtype=np.float64)\
                     if initial_phases is None else initial_phases
            assert len(phases) == self._n_phases
            def _split(f):
                return lambda xs: f(xs[:-self._n_phases], xs[-self._n_phases:])
            target_f = _split(self.distance)
            jacobian = lambda xs: np.concatenate(_split(self.d_distance)(xs))
            inits = np.concatenate((angles, phases))
        else:
            target_f = self.distance
            jacobian = self.d_distance
            inits = angles
        return scipy.optimize.minimize(target_f, inits, jac=jacobian,
                                       method='BFGS', **kwargs)

    def split_result(self, opt_res):
        """split_result(opt_res) -> angles, phases"""
        if self.fixed_phase:
            return opt_res.x, np.zeros(1, dtype=np.float64)
        return opt_res.x[:-self._n_phases],\
               np.insert(opt_res.x[-self._n_phases:], 0, 0.0)

    def _print_trace(self, trace_out, n_digits=5):
        _transpose = lambda arr: map(list, np.transpose(list(map(list, arr))))
        _reorder_ind = lambda arr: map(reversed, _transpose(map(reversed, arr)))
        _colour_str = lambda cop: "{}({})".format(cop.colour,
                                                  round(cop.angle, n_digits))
        def _normalise_string_lengths(lst):
            maxl = reduce(lambda acc, s: max(acc, len(s)), lst, 0)
            return [s + " " * (maxl - len(s)) for s in lst]
        def _split_ground_excited(arr):
            split = lambda lst: [lst[:len(lst) // 2], lst[len(lst) // 2:]]
            return list(chain.from_iterable(map(split, arr)))
        def _group_pulses(arr):
            pair = lambda line: ["  ".join(line[2 * i : 2 * i + 2])\
                                 for i in range(len(line) // 2)]
            return [pair(list(line)) for line in arr]

        str_cols = [[_format_complex(x, 5) for x in lst] for lst in trace_out]
        str_cols = _split_ground_excited(str_cols)
        for i, col in enumerate(str_cols):
            str_cols[i] = col + ["|e>" if i % 2 is 0 else "|g>"]
        str_cols = map(_normalise_string_lengths, str_cols)
        str_cols = _reorder_ind(str_cols)
        colour_strings = list(map(_colour_str, self._lin_ops)) + ["start"]
        str_cols = _transpose([colour_strings] + _group_pulses(str_cols))
        motionals = ["|{}>".format(i) for i in range(self._ns - 1, -1, -1)]
        str_cols = [["", ""] + motionals] + list(str_cols)
        for line in _transpose(map(_normalise_string_lengths, str_cols)):
            print("  |  ".join(line))

    def trace(self, angles=None, fmt=True):
        """
        Prettily print the evolved state after each pulse of the colour
        sequence.  If `format == False`, then the states (including the start
        state) will be returned as a list, and nothing will be printed.

        If the angles of the pulses are not specified then the last used set of
        angles will be traced instead.  self is useful for tracing the immediate
        output of an optimise call.
        """
        self._update_angles_if_required(angles)
        out = np.empty((self._len + 1, 2 * self._ns), dtype=np.complex128)
        out[0] = self._start
        for i in range(self._len):
            out[i + 1] = np.dot(self._lin_ops[self._len - i - 1].op, out[i])
        if not fmt:
            return out
        else:
            self._print_trace(out)
