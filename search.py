from __future__ import print_function
from pulse_sequence import PulseSequence, motional_states_needed
from functional import *
from itertools import *
import time
import operator
import random

start = time.clock()

ns = 6
max_superposition = 5
targets = [[ [x, y] for x in xrange(ns) for y in xrange(x + 1, ns) ]]
for i in xrange(3, max_superposition + 1):
    targets.append([prev + [x] for prev in targets[i - 3]
                               for x in xrange(max(prev) + 1, ns) ])
files = map(lambda t: open("motional_{}".format(len(t[0])), 'w'), targets)
errs  = map(lambda t: open("err_{}".format(len(t[0])), 'w'), targets)
targets = reduce(operator.__add__, targets)

attempts_to_succeed = 5
attempts_after_success = 4

def possible_sequences(n, target):
    """
    An iterator through all the pulse sequences of length `n`.  Pulses which
    start with the red sideband do not exist, because a starting red sideband
    does nothing, so we will have already checked the corresponding (n - 1)
    length pulse before.  Also, pulses which have two consecutive sidebands of
    the same colour are disallowed, as this is equivalent to one pulse of
    different length, so will have been checked before.
    """
    def allowed(colours):
        tests = [
            # First pulse mustn't be red.
            colours[-1] is not 'r',
            # Mustn't have two consecutive pulses of the same colour.
            not exists(lambda (a, b): a == b, pairs(colours)),
            # The colour sequence must reach the max occupied state exactly.
            motional_states_needed(colours) == max(target) + 1 ]
        return reduce(operator.__and__, tests, True)
    rev = lambda small_iterator: list(small_iterator)[::-1]
    return ifilter(allowed, imap(rev, product("crb", repeat = n)))

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
    possibles = lambda n: possible_sequences(n, target)
    return chain.from_iterable(imap(possibles, count(1)))

def start_angles(colours):
    return map(lambda _: random.uniform(-1, 1), colours)

def was_successful(result):
    return result.fun < 1e-8 and result.success

for i, target in enumerate(targets):
    out = files[len(target) - 2]
    err = errs[len(target) - 2]
    if i != 0:
        print("\n", file = out)
    print("Target: ", ", ".join(map(str, target)), file = out)
    print("Target: ", ", ".join(map(str, target)), file = err)
    cur_len = 0
    len_success = False
    for colours in colour_sequences(target):
        if len_success > 0 and len(colours) > cur_len:
            len_success = 0
            break
        elif len(colours) != cur_len:
            cur_len = len(colours)

        outs = []
        pulses = PulseSequence(colours, target)
        for _ in xrange(attempts_to_succeed):
            result = pulses.optimise(start_angles(colours))
            if not was_successful(result):
                continue
            outs.append(result.x)
            for _ in xrange(attempts_after_success):
                result = pulses.optimise(start_angles(colours))
                if was_successful(result):
                    outs.append(tuple(result.x))
            break

        if len(outs) > 0:
            if len_success > 0:
                print(file = out)
            len_success = len_success + 1
            print("- Colours: ", ", ".join(colours), file = out)
            print("- Success: ", ", ".join(colours), file = err)
            grouping = lambda lst: map(lambda x: round(x, 5), lst)
            for _, g in groupby(outs, grouping):
                print("-- Angles: ", ", ".join(map(str, g.next())), file = out)
        else:
            print("- Failure: ", ", ".join(colours), file = err)

print("Time elapsed: ", time.clock() - start)
