from pulse_matrices import ColourOperator, build_state_vector
from math import *
from numpy import angle as arg
from operator import xor
import itertools as it
import numpy as np
import state_specifier as ss

class Tree(object):
    pass

class Leaf(Tree):
    def __init__(this, pulses, state, max_n):
        this.pulses = pulses
        this.state = state
        this.max_n = max_n

class Branch(Tree):
    def __init__(this, pulses, state, max_n, gtree, etree):
        this.pulses = pulses
        this.state = state
        this.max_n = max_n
        this.gtree = gtree
        this.etree = etree

def bound_angle(angle):
    """Get the angle (divided by pi) bounded on (-1, 1]."""
    out = fmod(angle, 2.0)
    return out if -1.0 < out <= 1.0 else out - np.sign(out) * 2.0

def feq(a, b):
    """Compare two floats for near equality."""
    return abs(a - b) < 1e-8

def other_coupled_element(colour, element):
    """other_coupled_element(colour, element) -> other_element

    Given one of the states coupled by a colour operator, get the state which is
    coupled to it.

    Arguments:
    - colour: 'c' | 'r' | 'b' -- The colour of the pulse applied.
    - element: state_specifier -- One element of the coupled pair.

    Returns:
    - other_element: state_specifier -- The other coupled element.

    Throws:
    - ValueError: If the given `element` does not interact with the pulse."""
    _int, _mot = ss.internal(element), ss.motional(element)
    if _int is {'c': None, 'r': 'g', 'b': 'e'}[colour] and _mot is 0:
        raise ValueError("The {} pulse doesn't interact with |{}0>."\
            .format(*{'r': ("red", 'g'), 'b': ("blue", 'e')}[colour]))
    _o_int = 'g' if _int is 'e' else 'e'
    if colour is 'c':
        return (_mot, _o_int)
    elif (colour is 'r' and _int is 'g') or (colour is 'b' and _int is 'e'):
        return (_mot - 1, _o_int)
    else:
        return (_mot + 1, _o_int)

def phase_neg(colour, into, out_of, into_is_ground):
    """phase_neg(colour, into, out_of, into_is_ground) -> neg

    Return a Boolean saying whether there is a relative negative sign due to the
    phase difference between the two states.

    Arguments:
    - colour: 'c' | 'r' | 'b' -- The colour of the pulse applied.
    - into: complex float --
        The coefficient of the element the pulse is moving the population into.
    - out_of: complex float --
        The coefficient of the element the pulse is moving the population out
        of.
    - into_is_ground: boolean --
        True if the pulse is moving the population into the ground state, and
        False if the pulse is moving the population into the excited state.

    Returns:
    - neg: boolean --
        True if there is a negation associated with the phase difference of the
        two coefficients (e.g. sin(delta) = -1 for the carrier).

    Throws:
    - ValueError --
        If complete population movement cannot be achieved due to the phase
        difference.

    Notes:
        If one of the coefficients is 0, then this function always returns
        False, even though both True and False are valid return values."""
    if feq(abs(into), 0.0) or feq(abs(out_of), 0.0):
        return False
    delta = bound_angle((arg(into) - arg(out_of)) / pi)
    allowed_deltas = [-0.5, 0.5] if colour is 'c' else [-1.0, 0.0, 1.0]
    if not any([feq(delta, x) for x in allowed_deltas]):
        raise ValueError(
            "The {} pulse needs a phase difference as one of {}, "
            "but I got {} and {} which is a difference of {}."\
                .format({'c': 'carrier', 'r': 'red', 'b': 'blue'}[colour],
                        allowed_deltas,
                        arg(into) / pi, arg(out_of) / pi, delta))
    if colour is 'c':
        return feq(delta, -0.5)
    else:
        return into_is_ground != feq(delta, 0.0)

def single_pulse(colour, target, state_vector, adjoint = True):
    """single_pulse(colour, target, state_vector, adjoint = True)
    -> angle

    Choose an angle for a pulse of `colour` which moves all the population of
    the state coupled to `target` into `target`.  In other words, find a `phi`
    such that
        colour(phi) . (a |target> + b |coupled to target>)
            = e^(i theta) sqrt(|a|^2 + |b|^2) |target>
    for some `theta`.  `a` and `b` are the coefficients of the two states, which
    the function will read from `state_vector`.  These must be of the correct
    phase difference.

    Arguments:
    - colour: 'c' | 'r' | 'b' -- The colour of the pulse applied.
    - target: state_specifier -- The element to move the population into.
    - state_vector: 1D numpy.array of complex float --
        The vector form of the ket of the current state of the system.
    - adjoint (optional, True): boolean --
        Whether the pulse is an adjoint pulse or not.  For finding the sequence
        of pulses, this will usually be True.

    Returns:
    - angle: real float --
        The angle `phi` from the summary, divided by pi.

    Throws:
    - ValueError --
        - If the `target` element isn't coupled by the colour specified.
        - If the coefficients of the two coupled elements don't allow complete
          population transfer."""
    ns = state_vector.shape[0] // 2
    other = other_coupled_element(colour, target)
    into = state_vector[ss.idx(target, ns)]
    out_of = state_vector[ss.idx(other, ns)]
    k = 1.0 if colour is 'c'\
        else 1.0 / sqrt(max(list(map(ss.motional, (target, other)))))
    if feq(abs(out_of), 0.0):
        return k
    neg = adjoint != phase_neg(colour, into, out_of, ss.internal(target) is 'g')
    neg = -1.0 if neg else 1.0
    pop = sqrt(abs(into) ** 2 + abs(out_of) ** 2)
    return 2.0 * k * atan((neg * abs(into) + pop) / abs(out_of)) / pi

def is_populated(element, state_vector):
    """is_populated(element, state_vector) -> bool

    Determine if the `element` (`state_specifier`) has non-zero probability."""
    ns = state_vector.shape[0] // 2
    return not feq(abs(state_vector[ss.idx(element, ns)]) ** 2, 0.0)

def both_populated(motional, state_vector):
    """both_populated(motional, state_vector) -> bool

    Determine if both states of phonon number `motional` have non-zero
    probability of being occupied."""
    return is_populated((motional, 'g'), state_vector)\
           and is_populated((motional, 'e'), state_vector)

def chequerboard_phases(target_spec, pi_phases = None):
    """chequerboard_phases(target_spec) -> new_target_spec

    Apply the phase chequerboard to the targets specified so that a pulse
    sequence to evolve to it will exist."""
    if pi_phases is None:
        pi_phases = [ 0.0 for _ in range(len(target_spec) - 1) ]
    else:
        pi_phases = list(pi_phases)
    assert len(pi_phases) == len(target_spec) - 1,\
        "There should be a pi phase for all but one of the targets."
    pi_phases.insert(0, 0.0)
    for i, t in enumerate(target_spec):
        p = 0.0 if ss.motional(t) % 2 == {'g':0, 'e':1}[ss.internal(t)]\
            else 0.5
        target_spec[i] = ss.set_phase(t, bound_angle(p + pi_phases[i]))
    return target_spec

def build_tree(target_spec):
    max_n = max(list(map(ss.motional, target_spec)))
    current_state = build_state_vector(target_spec, max_n + 1)
    ops = {'c': ColourOperator('c', max_n + 1),
           'r': ColourOperator('r', max_n + 1),
           'b': ColourOperator('b', max_n + 1)}
    def _tree(pulses, state, max_n):
        def nexts(colour, target):
            angle = single_pulse(colour, target, state, adjoint = True)
            new_pulses = pulses + [(colour, angle)]
            new_state = np.dot(ops[colour].U(-angle), state)
            new_max_n = max_n if colour is 'c' else max_n - 1
            return (new_pulses, new_state, new_max_n)
        if max_n == 0 and not is_populated((0, 'e'), state):
            return Leaf(pulses, state, max_n)
        elif max_n == 0:
            return _tree(*nexts('c', (0, 'g')))
        elif both_populated(max_n, state):
            return Branch(pulses, state, max_n,
                          _tree(*nexts('c', (max_n, 'g'))),
                          _tree(*nexts('c', (max_n, 'e'))))
        elif is_populated((max_n, 'e'), state):
            return _tree(*nexts('b', (max_n - 1, 'g')))
        else: # is_populated((max_n, 'g'), current_state)
            return _tree(*nexts('r', (max_n - 1, 'e')))
    return _tree([], current_state, max_n)

def extract_pulses(tree):
    assert isinstance(tree, Tree), "The tree must be a Tree!"
    if isinstance(tree, Leaf):
        return [ tree.pulses ]
    else: # isinstance(tree, Branch):
        return extract_pulses(tree.gtree) + extract_pulses(tree.etree)

def find_all_pulses(target_spec):
    def mapping(pi_phases):
        target = chequerboard_phases(target_spec, pi_phases)
        return list(target), list(extract_pulses(build_tree(target)))
    all_pi_phases = it.product([0.0, 1.0], repeat = len(target_spec) - 1)
    return list(map(mapping, all_pi_phases))

def find_pulses(target_spec):
    """find_pulses(target_spec) -> pulses

    Find a sequence of pulses which evolve the state |g0> into the state
    specified by `target_spec`.

    Arguments:
    - target_spec: array_like of state_specifier --
        The states which should be equally populated after the pulse sequence.

    Returns:
    - pulses: list of (colour * angle) --
        - colour: 'c' | 'r' | 'b' -- The colour of the pulse to apply.
        - angle: real float -- The angle to apply the pulse for, divided by pi.

    Notes:
        This can technically throw `ValueError` and `AssertionError`, but both
        of these imply logic errors in the underlying code and cannot be
        recovered from."""
    target_spec = chequerboard_phases(target_spec)
    max_n = max(list(map(ss.motional, target_spec)))
    current_state = build_state_vector(target_spec, max_n + 1)
    carrier, red, blue = tuple([ColourOperator(x, max_n+1) for x in "crb"])
    pulses = []
    for _ in range(2 * max_n + 2, 0, -1):
        if max_n == 0 and not is_populated((0, 'e'), current_state):
            return pulses
        elif max_n == 0:
            op, target = carrier, (0, 'g')
        elif both_populated(max_n, current_state):
            op, target = carrier, (max_n, 'g')
        elif is_populated((max_n, 'e'), current_state):
            op, target = blue, (max_n - 1, 'g')
        else: # is_populated((max_n, 'g'), current_state)
            op, target = red, (max_n - 1, 'e')
        angle = single_pulse(op.colour, target, current_state, adjoint = True)
        pulses.append((op.colour, angle))
        current_state = np.dot(op.U(-angle), current_state)
        max_n = max_n if op is carrier else max_n - 1
    assert False, "Logic error: failed to find the sequence."
