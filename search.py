from __future__ import print_function
from pulse_sequence import PulseSequence, motional_states_needed
from functional import *
from itertools import *
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
    min_cols = max_n if max_n % 2 == 0 else max_n + 1
    # There are no longer sequences than `[ 'c' ] + [ 'b', 'c' ] * max(n)` that
    # can satisfy the rules, so we can cut off there.
    max_cols = 2 * max_n + 1
    return chain.from_iterable(imap(possibles, xrange(min_cols, max_cols + 1)))

def start_angles(colours):
    """Produce a set of angles to use to start for a certain colour sequence."""
    return map(lambda _: random.uniform(-1, 1), colours)

def was_successful(result):
    """Return True iff we should consider the optimiser result a success."""
    return result.fun < 1e-8 and result.success

def group_angles(angles):
    """Remove duplicate angle sequences, and return as a set."""
    return set(map(lambda lst: tuple(map(lambda x: round(x, 6), lst)), angles))

def try_colours(target, colours, before_success = 5, after_success = 0):
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

    Returns:
    set of tuples of doubles --
        A set of all the successful sequences of angles which will evolve the
        state |g0> into the target state.  In the tuples, the order of the
        angles is the same as the order of the colours input.
    """
    outs = []
    pulses = PulseSequence(colours, target)
    for _ in xrange(before_success):
        result = pulses.optimise(start_angles(colours))
        if not was_successful(result):
            continue
        else:
            outs.append(tuple(result.x))
            for _ in xrange(after_success):
                result = pulses.optimise(start_angles(colours))
                if was_successful(result):
                    outs.append(tuple(result.x))
            break
    return group_angles(outs)

def search(target, before_success = 5, after_success = 0, log = None):
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
    cur_len = 0
    len_successes = 0
    all_outs = []
    for colours in colour_sequences(target):
        if len_successes > 0 and len(colours) > cur_len:
            return all_outs
        elif len(colours) != cur_len:
            cur_len = len(colours)
        angles = try_colours(target, colours, before_success, after_success)
        if len(angles) != 0:
            len_successes = len_successes + 1
            all_outs.append((colours, angles))
            if log is not None:
                print("- Success:", ", ".join(colours), file = log)
        elif log is not None:
            print("- Failure:", ", ".join(colours), file = log)
    print("- Fatal: no more colour sequences to try.", file = log)
    return []
