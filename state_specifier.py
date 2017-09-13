"""
Functions operating on state_specifiers.

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
"""

import numbers

def motional(state):
    """Get the motional level from a state_specifier."""
    if isinstance(state, int) and state >= 0:
        return state
    elif len(state) >= 1 and isinstance(state[0], int) and state[0] >= 0:
        return state[0]
    base = (
        "The state should either be an unsigned integer or a "
        "'state_specifier' tuple which looks like "
        "'unsigned int * ?('g' | 'e') * ?double'.  "
        "This state is a {0}: '{1}'".format(type(state), state)
        )
    raise TypeError(base)

def internal(state):
    """Get the internal level from a state_specifier or 'g' if there is none."""
    try:
        if isinstance(state[1], str) and state[1] == 'g' or state[1] == 'e':
            return state[1]
    except:
        pass
    return 'g'


def phase(state):
    """Get the phase from a state_specifier as an angle (fraction of pi)."""
    try:
        if isinstance(state[1], str) and isinstance(state[2], numbers.Number):
            return state[2]
        elif isinstance(state[1], numbers.Number):
            return state[1]
    except:
        pass
    return 0.0

def set_motional(state, motional):
    return (motional, internal(state), phase(state))

def set_internal(state, internal):
    return (motional(state), internal, phase(state))

def set_phase(state, phase):
    return (motional(state), internal(state), phase)

def idx(state, ns):
    return {'e': 0, 'g': ns}[internal(state)] + motional(state)
