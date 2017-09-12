"""
Supplementary types:
state_specifier: (motional * ?internal * ?phase) --
    motional: unsigned int --
        The motional level of the state, e.g. 0, 2, 3 etc.
    internal (optional, 'g'): 'g' | 'e' --
        The internal part of the state, either ground or excited.  Defaults to
        ground.
    phase (optional, 0.0): double --
        The relative phase of this state as an angle, divided by pi.  The
        resulting phase will be `e^(i * pi * phase)`, for example:
            0.0 => 1.0 + 0.0i,
            0.5 => 0.0 + 1.0i,
            1.0 => -1.0 + 0.0i,
            3.0 => -1.0 + 0.0i.

    Access this type using functions from state_specifier.py.
"""

from __future__ import division, print_function
import numpy as np
import scipy.optimize
from math import *
from cmath import exp as cexp
import state_specifier as state
from pulse_matrices import *
from random import SystemRandom
import itertools

def random_array(shape, lower = 0.0, upper = 1.0, **kwargs):
    sr = SystemRandom()
    length = reduce(lambda acc, x: acc * x, shape, 1)\
             if isinstance(shape, tuple) else shape
    rands = [ sr.uniform(lower, upper) for _ in xrange(length) ]
    return np.array(rands, **kwargs).reshape(shape)

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
        this.colour = colour
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
    def __init__(this, colours, target = None, fixed_phase = False,
                 start = [(0, 'g', 0.0)]):
        assert len(colours) > 0,\
            "You must have at least one colour in the sequence!"
        this.colours = colours
        this._len = len(this.colours)
        this._n_phases = len(target) - 1 if target is not None else None
        this._ns = max(motional_states_needed(colours),
                       max(map(state.motional, start)) + 1,
                       max(map(state.motional, target)) + 1\
                       if target is not None else 0)
        this.fixed_phase = fixed_phase or this._n_phases is 0
        this._orig_target = target
        this._target = build_state_vector(target, this._ns)\
                       if target is not None else None
        this._start = build_state_vector(start, this._ns)
        this._lin_ops =\
            np.array([ ColourOperator(c, this._ns) for c in this.colours ])
        # The angles are for every colour in the sequence, read in left-to-right
        # order.  The angles are all divided by pi.
        this._angles = None
        # The stored phases are for every element of the target except the
        # first.  The first is always taken to be 0, and the others are rotated
        # so that this is true (if a first phase is supplied).  The phases are
        # stored as angles, divided by pi (e.g. 0 => 1, 0.5 => i, 1 => -1 etc).
        this._phases = None

        # Output storage.
        this._u = np.empty((2 * this._ns, 2 * this._ns), dtype = np.complex128)
        this._d_u = np.empty((this._len, 2 * this._ns, 2 * this._ns),
                             dtype = np.complex128)
        if target is not None:
            # This shouldn't be a np.array because it needs to be able to hold
            # state_specifier tuples which are variable length.
            this._new_target = [ 0 for _ in this._orig_target ]
            this._dist = float("inf")
            this._d_dist_angles = np.empty(this._len, dtype = np.float64)
            this._d_dist_phases = np.empty(len(target) - 1, dtype = np.float64)

        # Pre-allocate calculation scratch space and fixed variables.
        this._id = np.identity(2 * this._ns, dtype = np.complex128)
        this._partials_ltr = np.empty_like(this._d_u)
        this._partials_rtl = np.empty_like(this._d_u)
        this._partials_ltr[0] = this._id
        this._partials_rtl[0] = this._id
        this._temp = np.empty_like(this._u)
        this._tus = 0.0j

    def _update_target_phases(this):
        this._new_target[0] = state.set_phase(this._orig_target[0], 0)
        for i, s in enumerate(this._orig_target[1:]):
            this._new_target[i + 1] = state.set_phase(s, this._phases[i])
        this._target = build_state_vector(this._new_target, this._ns)

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

    def _update_distance(this):
        this._tus = inner_product(this._target, this._u, this._start)
        this._dist = 1.0 - (this._tus * np.conj(this._tus)).real

    def _update_distance_angle_derivatives(this):
        for i in xrange(this._len):
            prod = inner_product(this._target, this._d_u[i], this._start)
            this._d_dist_angles[i] = -2.0 * (np.conj(this._tus) * prod).real

    def _update_distance_phase_derivatives(this):
        pref = 2.0 / sqrt(len(this._orig_target))
        u_start = np.dot(this._u, this._start)
        for i, p in enumerate(this._phases):
            phase = cexp(1.0j * pi * (0.5 - p))
            # we can calculate the inner product <g n_j|U|start> by
            # precalculating U|start>, then indexing to the relevant element.
            idx = state.idx(this._orig_target[i + 1], this._ns)
            this._d_dist_phases[i] =\
                pref * (phase * u_start[idx] * np.conj(this._tus)).real

    def _update_angles_if_required(this, angles):
        if angles is None:
            return False
        assert len(angles) == this._len,\
            "There are {} colours in the sequence, but I got {} angles."\
            .format(this._len, len(angles))
        if np.array_equal(this._angles, angles):
            return False
        this._angles = angles
        for i in xrange(len(this._angles)):
            this._lin_ops[i].angle = this._angles[i]
        return True

    def _update_phases_if_required(this, phases):
        if this.fixed_phase\
           or phases is None\
           or np.array_equal(this._phases, phases):
            return False
        assert 0 <= len(this._orig_target) - len(phases) <= 1,\
            "There are {} elements of the target, but I got {} phases."\
            .format(len(this._orig_target), len(phases))
        if len(phases) == len(this._orig_target) - 1:
            this._phases = phases
        else:
            this._phases = np.vectorize(lambda x: x - phases[0])(phases[1:])
        this._update_target_phases()
        return True

    def _calculate_propagator(this, angles):
        if this._update_angles_if_required(angles):
            this._update_propagator_and_derivatives()

    def _calculate_all(this, angles, phases = None):
        assert this._target is None or this.fixed_phase or phases is not None,\
            "If you're not in fixed phase mode, you need to specify the phases."
        updated_angles = this._update_angles_if_required(angles)
        updated_phases = this._update_phases_if_required(phases)
        if not (updated_angles or updated_phases):
            return
        if updated_angles:
            this._update_propagator_and_derivatives()
        if this._target is not None:
            this._update_distance()
            if updated_angles:
                this._update_distance_angle_derivatives()
            if updated_phases:
                this._update_distance_phase_derivatives()

    def U(this, angles):
        """
        Get the propagator of the pulse sequence stored in the class with the
        specified angles.
        """
        this._calculate_propagator(angles)
        return np.copy(this._u)

    def d_U(this, angles):
        """
        Get the derivatives of the propagator of the pulse sequence stored in
        the class with the specified angles.
        """
        this._calculate_propagator(angles)
        return np.copy(this._d_u)

    def distance(this, angles, phases = None):
        """
        Get the distance of the pulse sequence stored in the class with the
        specified angles.
        """
        assert this._target is not None,\
            "You must set the target state to calculate the distance."
        this._calculate_all(angles, phases)
        return this._dist

    def d_distance(this, angles, phases = None):
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
        assert this._target is not None,\
            "You must set the target state to calculate the distance."
        assert this.fixed_phase or phases is not None,\
            "If you're not in fixed phase mode, you need to specify the phases."
        this._calculate_all(angles, phases)
        if this.fixed_phase:
            return np.copy(this._d_dist_angles)
        else:
            return np.copy(this._d_dist_angles), np.copy(this._d_dist_phases)

    def optimise(this, initial_angles = None, initial_phases = None, **kwargs):
        assert this._target is not None,\
            "You must set the target state to optimise a pulse sequence."
        angles = random_array(this._len, dtype = np.float64)\
                 if initial_angles is None else initial_angles
        if not this.fixed_phase:
            phases = random_array(this._n_phases, dtype = np.float64)\
                     if initial_phases is None else initial_phases
            assert len(phases) == this._n_phases
            def split(f):
                return lambda xs: f(xs[:-this._n_phases], xs[-this._n_phases:])
            target_f = split(this.distance)
            jacobian = lambda xs: np.concatenate(split(this.d_distance)(xs))
            inits = np.concatenate((angles, phases))
        else:
            target_f = this.distance
            jacobian = this.d_distance
            inits = angles
        return scipy.optimize.minimize(target_f, inits, jac = jacobian,
                                       method = 'BFGS', **kwargs)

    def split_result(this, opt_res):
        if this.fixed_phase:
            return opt_res.x, np.zeros(1, dtype = np.float64)
        else:
            return opt_res.x[:-this._n_phases],\
                   np.insert(opt_res.x[-this._n_phases:], 0, 0.0)

    def _print_trace(this, trace_out):
        n_digits = 5
        map_level_2 = lambda f, lst: map(lambda inner: map(f, inner), lst)
        transpose   = lambda arr:    list(map(list, np.transpose(arr)))
        reverse     = lambda lst:    list(reversed(lst))
        reorder_ind = lambda arr:    map(reverse, transpose(map(reverse, arr)))
        colour_str  = lambda cop:    "{}({})".format(cop.colour,
                                                     round(cop.angle, n_digits))
        def format_complex(z):
            z = round(z.real, n_digits) + round(z.imag, n_digits) * 1.0j
            if z.imag == 0:
                return str(z.real)
            elif z.real == 0:
                return "{}i".format(z.imag)
            return "({0} {2} {1}i)".format(z.real, abs(z.imag),
                                           {-1: '-', 1: '+'}[np.sign(z.imag)])
        def normalise_string_lengths(lst):
            maxl = reduce(lambda acc, s: max(acc, len(s)), lst, 0)
            return map(lambda s: s + " " * (maxl - len(s)), lst)
        def split_ground_excited(arr):
            split = lambda lst: [ lst[:len(lst) // 2], lst[len(lst) // 2:] ]
            return list(itertools.chain.from_iterable(map(split, arr)))
        def group_pulses(arr):
            pair = lambda line: [ "  ".join(line[2 * i : 2 * i + 2])\
                                  for i in xrange(len(line) // 2) ]
            return map(pair, arr)

        str_cols = split_ground_excited(map_level_2(format_complex, trace_out))
        for i, col in enumerate(str_cols):
            str_cols[i] = col + [ "|e>" if i % 2 is 0 else "|g>" ]
        str_cols = map(normalise_string_lengths, str_cols)
        individual = reorder_ind(str_cols)
        colour_strings = map(colour_str, this._lin_ops) + [ "start" ]
        with_colours = transpose([ colour_strings ] + group_pulses(individual))
        motionals = [ "|{}>".format(i) for i in xrange(this._ns - 1, -1, -1) ]
        with_motionals = [ [ "", "" ] + motionals ] + with_colours
        for line in transpose(map(normalise_string_lengths, with_motionals)):
            print("  |  ".join(line))

    def trace(this, angles = None, format = True):
        """
        Prettily print the evolved state after each pulse of the colour
        sequence.  If `format == False`, then the states (including the start
        state) will be returned as a list, and nothing will be printed.

        If the angles of the pulses are not specified then the last used set of
        angles will be traced instead.  This is useful for tracing the immediate
        output of an optimise call.
        """
        this._update_angles_if_required(angles)
        out = np.empty((this._len + 1, 2 * this._ns), dtype = np.complex128)
        out[0] = this._start
        for i in xrange(this._len):
            out[i + 1] = np.dot(this._lin_ops[this._len - i - 1].op, out[i])
        if not format:
            return out
        else:
            this._print_trace(out)
