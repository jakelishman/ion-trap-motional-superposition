from __future__ import print_function
from pulse_sequence import PulseSequence, motional_states_needed
from functional import *
from itertools import *
import numpy as np
import math
import random

def colour_sequences(target):
    """
    An infinite iterator of all possible colour sequences, first with only one
    pulse, then all the ones with two, then three and so on.  Pulses which start
    with the red sideband do not exist, because a starting red sideband does
    nothing, so we will have already checked the corresponding (n - 1) length
    pulse before.  Also, pulses which have two consecutive sidebands of the same
    colour are disallowed, as this is equivalent to one pulse of different
    length, so will have been checked before.
    """
    max_n = max(target)
    rev = lambda small_iterator: list(small_iterator)[::-1]
    def allowed(colours):
        tests = [
            # First pulse mustn't be red.
            colours[-1] is not 'r',
            # Mustn't have two consecutive pulses of the same colour.
            not exists(lambda (a, b): a == b, pairs(colours)),
            # The colour sequence must reach the max occupied state exactly.
            motional_states_needed(colours) == max_n + 1,
            # Last pulse mustn't be blue (implied by previous test).
            colours[0] is not 'b' ]
        return not False in tests
    def possibles(ncols):
        """
        An iterator through all the pulse sequences of length `n`.  Pulses which
        start with the red sideband do not exist, because a starting red
        sideband does nothing, so we will have already checked the corresponding
        (n - 1) length pulse before.  Also, pulses which have two consecutive
        sidebands of the same colour are disallowed, as this is equivalent to
        one pulse of different length, so will have been checked before.
        """
        return ifilter(allowed, imap(rev, product("crb", repeat = ncols)))
    # There are no shorter sequences than `[ 'r', 'b' ] * max(n) // 2` for even
    # max(n), or `[ 'r', 'c' ] + [ 'r', 'b' ] * max(n) // 2` for odd max(n) that
    # can satisfy the rules (except for the trivial max(n) == 0).
    min_cols = max(max_n if max_n % 2 == 0 else max_n + 1, 1)
    # There are no longer sequences than `[ 'c' ] + [ 'b', 'c' ] * max(n)` that
    # can satisfy the rules, so we can cut off there.
    max_cols = 2 * max_n + 1
    return chain.from_iterable(imap(possibles, xrange(min_cols, max_cols + 1)))

def start_angles(colours):
    """Produce a set of angles to use to start for a certain colour sequence."""
    return map(lambda _: random.uniform(0, 1), colours)

def was_semi_successful(result):
    """Return True iff the result was mathematically successful, even if not
    physically so."""
    return result.fun < 1e-8 and result.success

def was_successful(result):
    """Return True iff we should consider the result a complete success."""
    return was_semi_successful(result)\
           and not any(map(lambda x: x < 0, result.x))

def bound_phase(phase):
    """Get the phase angle (divided by pi) bounded on (-1, 1]."""
    out = math.fmod(phase, 2.0)
    return out if -1.0 < out <= 1.0 else out - np.sign(out) * 2.0

def group_phases_and_angles(results):
    """Group by phase, remove duplicate angle sequences, and return as a set."""
    fst        = lambda tup:    tup[0]
    snd        = lambda tup:    tup[1]
    round_all  = lambda lst:    tuple(imap(lambda x: round(x, 5), lst))
    set_angles = lambda angles: set(imap(round_all, angles))
    by_phase = map(lambda t: (t[0], round_all(map(bound_phase, t[1]))), results)
    by_phase = sorted(by_phase, key = snd)
    by_phase = groupby(by_phase, key = snd)
    by_phase = imap(lambda t: (t[0], imap(fst, t[1])), by_phase)
    return dict(map(lambda t: (t[0], set_angles(t[1])), by_phase))

def try_colours(target, colours, before_success = 1, after_success = 0,
                max_retries = 0):
    """
    Try to optimise the given colour sequence to hit the specified target.
    Basically just a wrapper around PulseSequence.optimise().

    Arguments:
    target: 1D list of unsigned integer --
        The motional states which should be populated (in the ground state)
        after the pulse sequence.  These will be populated in equal proportions,
        in the same phase.

    colours: 1D list of 'c' | 'r' | 'b' --
        The sequence of colours pulses to apply, with 'c' being the carrier, 'r'
        being the first red sideband and 'b' being the first blue sideband.  If
        the array looks like
            ['r', 'b', 'c'],
        then the carrier will be applied first, then the blue, then the red,
        similar to how we'd write them in operator notation.

    before_success (optional, 5): unsigned integer --
        Number of attempts to optimise with randomised starting parameters
        before finding a success.  The function will exit with an empty set if
        no convergence is found after this number of attempts.

    after_success (optional, 0): unsigned integer --
        Number of attempts to find further angle sequences after we've seen a
        success already.  The maximum number of angle sequences which can be
        returned is `1 + after_success`.

    max_retries (optional, 0): unsigned integer --
        The number of additional retries allowed to fix semi-successful results
        (i.e. ones which are valid mathematically but not physically).

    Returns:
    set of tuples of doubles --
        A set of all the successful sequences of angles which will evolve the
        state |g0> into the target state.  In the tuples, the order of the
        angles is the same as the order of the colours input.
    """
    outs = []
    pulses = PulseSequence(colours, target)
    def get_result(result):
        return tuple(map(tuple, pulses.split_result(result)))
    tries = 0
    retries = 0
    while (tries - retries) < before_success:
        result = pulses.optimise(start_angles(colours))
        if not was_successful(result):
            if was_semi_successful(result):
                retries += 1
            tries += 1
            continue
        else:
            outs.append(get_result(result))
            tries = 0
            retries = 0
            while (tries - retries) < after_success:
                result = pulses.optimise(start_angles(colours))
                if was_successful(result):
                    outs.append(get_result(result))
                elif was_semi_successful(result):
                    retries += 1
                tries += 1
            break
    return group_phases_and_angles(outs)

def search(target, before_success = 1, after_success = 0, max_retries = 0,
           log_file = None):
    """
    Find all pulses sequences of the minimum length which will evolve the state
    |g0> into the target state.

    Arguments:
    target: 1D list of unsigned integer --
        The motional states which should be populated (in the ground state)
        after the pulse sequence.  These will be populated in equal proportions,
        in the same phase, e.g.
            search([0, 2, 3])
        will try to find a colour sequence which evolves to the state
            |g>(|0> + |2> + |3>)/sqrt(3).

    before_success (optional, 5): unsigned integer --
        Number of attempts to optimise with randomised starting parameters
        before finding a success.  The function will exit with an empty set if
        no convergence is found after this number of attempts.

    after_success (optional, 0): unsigned integer --
        Number of attempts to find further angle sequences after we've seen a
        success already.  The maximum number of angle sequences which can be
        returned is `1 + after_success`.

    max_retries (optional, 0): unsigned integer --
        The number of additional retries allowed to fix semi-successful results
        (i.e. ones which are valid mathematically but not physically).

    log (optional, None): nullable file --
        If not None, the colour sequences being tried will be logged to this
        file, along with whether they succeeded or not.

    Returns:
    1D list of (colours * angles) --
        colours: 1D list of 'c' | 'r' | 'b' --
            The sequence of coloured pulses to apply, with 'c' being the
            carrier, 'r' being the first red sideband and 'b' being the first
            blue sideband.  If the array looks like
                ['r', 'b', 'c'],
            then the carrier will be applied first, then the blue, then the red,
            similar to how we'd write them in operator notation.

        angles: set of tuples of doubles --
            A set of all the successful sequences of angles which will evolve
            the state |g0> into the target state for the attached colour
            sequence.  In the tuples, the order of the angles is the same as the
            order of the colours.

        A set of all the successful sequences of colours and the corresponding
        angles which will evolve the state |g0> into the target state.  All the
        colour sequences will be of the same length, and this will be minimum
        length for which there was a convergence.
    """
    log = (lambda *tup: print(*tup, file = log_file)) if log_file is not None\
          else (lambda *tup: None)
    cur_len = 0
    len_successes = 0
    all_outs = []
    for colours in colour_sequences(target):
        if len_successes > 0 and len(colours) > cur_len:
            return all_outs
        elif len(colours) != cur_len:
            cur_len = len(colours)
        dct = try_colours(target, colours, before_success, after_success,
                          max_retries)
        if len(dct) != 0:
            len_successes = len_successes + 1
            all_outs.append((colours, dct))
            log("  Success:", ", ".join(colours))
        else:
            log("  Failure:", ", ".join(colours))
    log("  Fatal: no more colour sequences to try.")
    return []
