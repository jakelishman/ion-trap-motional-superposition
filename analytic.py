from pulse_matrices import ColourOperator, build_state_vector
from math import *
from numpy import angle as arg
from operator import xor
import numpy as np
import state_specifier as state

def bound_angle(angle):
    """Get the angle (divided by pi) bounded on (-1, 1]."""
    out = fmod(angle, 2.0)
    return out if -1.0 < out <= 1.0 else out - np.sign(out) * 2.0

def _feq(a, b):
    """Compare two floats for near equality."""
    return abs(a - b) < 1e-8

def _validate_target(internal, motional):
    """Check the target of an angle calculator is valid."""
    if not (internal is 'g' or internal is 'e'):
        raise TypeError("The internal state must be 'g' or 'e', but I got: {}"\
                        .format(internal))
    if not (isinstance(motional, int) and motional >= 0):
        raise TypeError(
            "The motional state must be an unsigned integer, but I got: {}"\
            .format(motional))
    return

def _find_angle(a, b, neg):
    """_find_angle(a, b, neg) -> angle

    Find the smallest positive angle `x` which satisfies
        x = 2 arctan( (-1 if neg else 1) * (|a| +- sqrt(|a|^2 + |b|^2)) / |b| ).
    This is the base equation which needs a solution for all of the sideband
    pulses.

    Arguments:
    a: complex float -- One of the complex coefficients of a state pair.
    b: complex float -- One of the complex coefficients of a state pair.
    neg: Boolean -- Whether the arctan argument should be negated.

    Returns:
    angle: real float -- The smallest positive solution for `x`, divided by pi.
    """
    if _feq(abs(b), 0.0):
        return 1.0
    root = sqrt(abs(a) ** 2 + abs(b) ** 2)
    return (2.0 / pi) * atan(((-1.0 if neg else 1.0) * abs(a) + root) / abs(b))

def carrier_pulse(internal, motional, excited, ground, adjoint = False):
    """carrier_pulse(internal, motional, excited, ground, adjoint)
    -> (angle, coeff)

    Choose an angle for a carrier pulse such that the populations in the ground
    and excited states will be moved completely into one or the other.  In other
    words: Find an `x` and an `a` such that
        c(x) . (ex |en> + gr |gn>) = e^(i a) sqrt(|ex|^2 + |gr|^2) |?n>,
    where `?` is `internal`, and `n` is `motional`.  There will only be
    solutions if `excited` and `ground` are pi/2 out of phase with each other
    (e.g. one is real, the other purely imaginary).

    Arguments:
    internal: 'g' | 'e' --
        Whether we should end up in the ground or excited state at the end.

    motional: unsigned int --
        The motional level we should end up in.  The populations referred to be
        `excited` and `ground` should both be in this level, since they're the
        ones the carrier effects.  This parameter is ignored, since it's not
        relevant to the carrier.

    excited: complex float --
        The starting complex coefficient of the |en> state.  Must be pi/2 out of
        phase with `ground`.

    ground: complex float --
        The starting complex coefficient of the |gn> state.  Must be pi/2 out of
        phase with `excited`.

    adjoint (Optional, False): Boolean --
        Whether we're finding an adjoint pulse or a regular pulse.

    Returns:
    angle: real float --
        The angle of the carrier pulse, divided by pi.

    coeff: complex float --
        The ending complex coefficient of the populated state.
    """
    _validate_target(internal, motional)
    delta = bound_angle((arg(ground) - arg(excited)) / pi)
    needs = [
        _feq(abs(excited) ** 2, 0.0) and internal is 'e',
        _feq(abs(ground) ** 2, 0.0) and internal is 'g',
        _feq(abs(delta), 0.5),
    ]
    if not any(needs):
        raise ValueError(
            "The phase difference between coefficients in a carrier pulse "
            "should be +-pi/2, but I have '{}' and '{}', giving '{}'."\
            .format(excited, ground, delta))
    neg = reduce(xor, [internal is 'e', _feq(delta, -0.5), adjoint], False)
    if internal is 'e':
        ground, excited = excited, ground
    angle = _find_angle(ground, excited, neg)
    coeff = ground * cos(pi * angle)\
            - 1.0j * excited * sin(pi * angle) * (-1.0 if adjoint else 1.0)
    return angle, coeff

def red_pulse(internal, motional, excited, ground, adjoint = False):
    """red_pulse(internal, motional, excited, ground, adjoint)
    -> (angle, coeff)

    Choose an angle for a red pulse such that the populations in the ground
    and excited states will be moved completely into one or the other.  In other
    words: Find an `x` and an `a` such that
        r(x) . (ex |e> + gr |g>) = e^(i a) sqrt(|ex|^2 + |gr|^2) |?n>,
    where `?` is `internal`, and `n` is `motional`.  There will only be
    solutions if `excited` and `ground` are in phase, or pi out of phase with
    each other (e.g. one is real positive, the other real negative).

    Arguments:
    internal: 'g' | 'e' --
        Whether we should end up in the ground or excited state at the end.

    motional: unsigned int --
        The motional level we should end up in.  If `internal` is 'g', then the
        coupled states will be |en-1> and |gn>, or if it's 'e', they'll be |en>
        and |gn+1>.

    excited: complex float --
        The starting complex coefficient of the |e> state.  Must be either in
        phase with, or pi out of phase with `ground`.

    ground: complex float --
        The starting complex coefficient of the |e> state.  Must be either in
        phase with, or pi out of phase with `excited`.

    adjoint (Optional, False): Boolean --
        Whether we're finding an adjoint pulse or a regular pulse.

    Returns:
    angle: real float --
        The angle of the pulse, divided by pi.

    coeff: complex float --
        The ending complex coefficient of the populated state.
    """
    _validate_target(internal, motional)
    if motional == 0 and internal is 'g':
        raise ValueError("The red pulse doesn't interact with |g0>.")
    delta = bound_angle((arg(ground) - arg(excited)) / pi)
    needs = [
        _feq(abs(excited) ** 2, 0.0) and internal is 'e',
        _feq(abs(ground) ** 2, 0.0) and internal is 'g',
        _feq(abs(delta), 0.0) or _feq(abs(delta), 1.0),
    ]
    if not any(needs):
        raise ValueError(
            "The phase difference between coefficients in a red pulse "
            "should be +-1 or 0, but I have '{}' and '{}', giving '{}'."\
            .format(excited, ground, delta))
    neg = reduce(xor, [internal is 'e', _feq(abs(delta), 1.0), adjoint], False)
    if internal is 'g':
        root = sqrt(motional)
        angle = _find_angle(ground, excited, neg) / root
        coeff = ground * cos(root * angle * pi)\
                + excited * sin(root * angle * pi) * (-1.0 if adjoint else 1.0)
    else: # internal is 'e'
        root = sqrt(motional + 1)
        angle = _find_angle(excited, ground, neg) / root
        coeff = excited * cos(root * angle * pi)\
                - ground * sin(root * angle * pi) * (-1.0 if adjoint else 1.0)
    return angle, coeff

def blue_pulse(internal, motional, excited, ground, adjoint = False):
    """blue_pulse(internal, motional, excited, ground, adjoint)
    -> (angle, coeff)

    Choose an angle for a red pulse such that the populations in the ground
    and excited states will be moved completely into one or the other.  In other
    words: Find an `x` and an `a` such that
        b(x) . (ex |e> + gr |g>) = e^(i a) sqrt(|ex|^2 + |gr|^2) |?n>,
    where `?` is `internal`, and `n` is `motional`.  There will only be
    solutions if `excited` and `ground` are in phase, or pi out of phase with
    each other (e.g. one is real positive, the other real negative).

    Arguments:
    internal: 'g' | 'e' --
        Whether we should end up in the ground or excited state at the end.

    motional: unsigned int --
        The motional level we should end up in.  If `internal` is 'g', then the
        coupled states will be |en+1> and |gn>, or if it's 'e', they'll be |en>
        and |gn-1>.

    excited: complex float --
        The starting complex coefficient of the |e> state.  Must be either in
        phase with, or pi out of phase with `ground`.

    ground: complex float --
        The starting complex coefficient of the |e> state.  Must be either in
        phase with, or pi out of phase with `excited`.

    adjoint (Optional, False): Boolean --
        Whether we're finding an adjoint pulse or a regular pulse.

    Returns:
    angle: real float --
        The angle of the pulse, divided by pi.

    coeff: complex float --
        The ending complex coefficient of the populated state.
    """
    _validate_target(internal, motional)
    if motional == 0 and internal is 'e':
        raise ValueError("The blue pulse doesn't interact with |e0>.")
    delta = bound_angle((arg(ground) - arg(excited)) / pi)
    needs = [
        _feq(abs(excited) ** 2, 0.0) and internal is 'e',
        _feq(abs(ground) ** 2, 0.0) and internal is 'g',
        _feq(abs(delta), 0.0) or _feq(abs(delta), 1.0),
    ]
    if not any(needs):
        raise ValueError(
            "The phase difference between coefficients in a red pulse "
            "should be +-1 or 0, but I have '{}' and '{}', giving '{}'."\
            .format(excited, ground, delta))
    neg = reduce(xor, [internal is 'e', _feq(abs(delta), 1.0), adjoint], False)
    if internal is 'g':
        root = sqrt(motional + 1)
        angle = _find_angle(ground, excited, neg) / root
        coeff = ground * cos(root * angle * pi)\
                + excited * sin(root * angle * pi) * (-1.0 if adjoint else 1.0)
    else: # internal is 'e'
        root = sqrt(motional)
        angle = _find_angle(excited, ground, neg) / root
        coeff = excited * cos(root * angle * pi)\
                - ground * sin(root * angle * pi) * (-1.0 if adjoint else 1.0)
    return angle, coeff

def is_populated(internal, motional, state_vector):
    ns = state_vector.shape[0] // 2
    coeff = state_vector[state.idx((motional, internal), ns)]
    return not _feq(abs(coeff) ** 2, 0.0)

def both_populated(motional, state_vector):
    return is_populated('g', motional, state_vector)\
           and is_populated('e', motional, state_vector)

def _try_fixed(target_spec):
    maxn = max(map(state.motional, target_spec))
    ns = maxn + 1
    cur = build_state_vector(target_spec, ns)
    carrier, red, blue = tuple(map(lambda x: ColourOperator(x, ns), "crb"))
    pulses = []
    counter = 2 * maxn + 1
    while counter > 0:
        if maxn == 0 and not is_populated('e', 0, cur):
            return pulses
        if maxn == 0 or both_populated(maxn, cur):
            gr = cur[state.idx((maxn, 'g'), ns)]
            ex = cur[state.idx((maxn, 'e'), ns)]
            op = carrier
            angle, _ = carrier_pulse('g', maxn, ex, gr, adjoint = True)
        elif is_populated('e', maxn, cur):
            gr = cur[state.idx((maxn - 1, 'g'), ns)]
            ex = cur[state.idx((maxn, 'e'), ns)]
            op = blue
            angle, _ = blue_pulse('g', maxn - 1, ex, gr, adjoint = True)
            maxn -= 1
        elif is_populated('g', maxn, cur):
            gr = cur[state.idx((maxn, 'g'), ns)]
            ex = cur[state.idx((maxn - 1, 'e'), ns)]
            op = red
            angle, _ = red_pulse('e', maxn - 1, ex, gr, adjoint = True)
            maxn -= 1
        else:
            assert False, "wtf."
        pulses.append((op.colour, angle))
        cur = np.dot(op.U(-angle), cur)
        counter -= 1
    assert False, "Failed to find the sequence."

from itertools import *

def _try(target_spec):
    phases_seq = product([0.0, 0.5], repeat = len(target_spec) - 1)\
                 if len(target_spec) > 1 else [[]]
    phases_seq = imap(lambda tup: [0.0] + list(tup), phases_seq)
    for phases in phases_seq:
        for i in xrange(len(phases)):
            target_spec[i] = state.set_phase(target_spec[i], phases[i])
        try:
            return _try_fixed(target_spec)
        except:
            pass
    assert False, "fuck."
