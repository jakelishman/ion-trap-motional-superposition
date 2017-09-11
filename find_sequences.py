from __future__ import print_function
import time
import operator
import search
import sys
from itertools import *

def handle_command_length(args):
    """
    !length size max_n

    Create targets for all superpositions of length `size`, upto a maximum
    occuppied motional state of `max_n`.  For example, `!length 2 3` will make
    the superpositions
        |0> + |1> ; |0> + |2> ; |0> + |3> ;
        |1> + |2> ; |1> + |3> ;
        |2> + |3>.

    See also `!lengths`.
    """
    try:
        args = map(int, args.split())
    except (TypeError, AttributeError):
        raise SyntaxError(handle_command_length.__doc__.rstrip())
    if len(args) is not 2 or args[0] <= 0 or args[1] < args[0]:
        raise SyntaxError(handle_command_length.__doc__.rstrip())
    return combinations(xrange(args[1] + 1), r = args[0])

def handle_command_lengths(args):
    """
    !lengths min_size max_size max_n

    Create targets for all superpositions of lengths between `min_size` and
    `max_size`, upto a maximum occuppied motional state of `max_n`.  For
    example, `!lengths 2 3 3` will make the superpositions
        |0> + |1> ; |0> + |2> ; |0> + |3> ;
        |1> + |2> ; |1> + |3> ;
        |2> + |3> ;
        |0> + |1> + |2> ; |0> + |1> + |2> ; |0> + |1> + |3> ; |0> + |2> + |3> ;
        |1> + |2> + |3>.
    This is equivalent to several lines of `!length` with increasing values of
    `size`.
    """
    try:
        args = map(int, args.split())
    except (TypeError, AttributeError):
        raise SyntaxError(handle_command_lengths.__doc__.rstrip())
    tests = [
        len(args) is 3,
        args[0] > 0,
        args[1] >= args[0],
        args[2] >= args[1],
    ]
    if False in tests:
        raise SyntaxError(handle_command_lengths.__doc__.rstrip())
    return \
        chain.from_iterable(
            imap(lambda n: combinations(xrange(args[2] + 1), r = n),
                 xrange(args[0], args[1] + 1)))

def handle_command(command, args = None):
    known_commands = {
        "length" : handle_command_length,
        "lengths" : handle_command_lengths,
    }
    if command in known_commands:
        return known_commands[command](args)
    raise NotImplementedError(
        "I don't know the command '!" + str(command) + "'.")

def parse_target_specification(line):
    line = line.strip()
    if line is "" or line[0] is '#':
        return []
    elif line[0] is '!':
        parts = line.split(" ", 1)
        com  = parts[0][1:]
        args = parts[1:]
        return handle_command(com, *args)
    else:
        return [ [ int(n) for n in line.split() ] ]

def parse_targets():
    targets = []
    for line in sys.stdin:
        targets.extend(parse_target_specification(line))
    return targets

def oprint(*args):
    return print(*args, file = sys.stdout)
def eprint(*args):
    return print(*args, file = sys.stderr)

def main():
    before_success = 1 if len(sys.argv) < 2 else int(sys.argv[1])
    after_success  = 0 if len(sys.argv) < 3 else int(sys.argv[2])

    if sys.stdin.isatty():
        sys.exit("Must pipe commands into stdin.")
    for n, target in enumerate(parse_targets()):
        if n is not 0:
            sys.stderr.flush()
            oprint()
            eprint()
        oprint("Target:", ", ".join(map(str, target)))
        eprint("Target:", ", ".join(map(str, target)))
        results = search.search(target, before_success, after_success,
                                log_file = sys.stderr)
        for colours, dct in results:
            oprint("  Colours:", ", ".join(colours))
            for phases, angles_set in dct.iteritems():
                oprint("    Phases:", ", ".join(map(str, phases)))
                for angles in angles_set:
                    oprint("      Angles:", ", ".join(map(str, angles)))
    return

if __name__ == "__main__":
    start = time.clock()
    main()
    eprint("\nTime elapsed:", time.clock() - start)
