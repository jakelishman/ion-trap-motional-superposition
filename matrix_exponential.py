import numpy as np
from scipy.linalg import expm
from math import *
import operator

def is_sigma_plus(row, col, ns):
    return col >= ns and row < ns

def is_sigma_minus(row, col, ns):
    return row >= ns and col < ns

def start_n(_, col, ns):
    return col % ns

def end_n(row, _, ns):
    return row % ns

def is_ladder(ladder):
    return lambda row, col, ns:\
        end_n(row, col, ns) - start_n(row, col, ns) is ladder

def ladder_coefficient(ladder, start_n):
    ends = [ start_n + 1, start_n + ladder + 1 ]
    return sqrt(reduce(operator.mul, range(min(ends), max(ends)), 1))

def sideband_element(order, row, col, ns):
    """
    The `order` parameter is 0 for carrier, -1 for 1st red, +1 for 1st blue,
    and so on.
    """
    if is_sigma_plus(row, col, ns) and is_ladder(order)(row, col, ns):
        return ladder_coefficient(order, start_n(row, col, ns))
    elif is_sigma_minus(row, col, ns) and is_ladder(-order)(row, col, ns):
        return ladder_coefficient(-order, start_n(row, col, ns))
    else:
        return 0.0

def sideband_hamiltonian(order):
    return lambda ns: np.array([
        [sideband_element(order, row, col, ns) for col in xrange(2 * ns)]
        for row in xrange(2 * ns)])

def u(angle, hamiltonian):
    return expm(-1.0j * pi * angle * hamiltonian)

def du(angle, hamiltonian):
    return np.dot(-1.0j * hamiltonian, u(angle, hamiltonian))

def adj(op):
    return np.conj(op.T)

def overlap(final, initial):
    x = np.dot(adj(final), initial)
    return (x * np.conj(x)).real

class sideband(object):
    @property
    def ns(this):
        return this._ns

    @ns.setter
    def ns(this, ns):
        this._hamiltonian = this._generator(ns)
        this._ns = ns

    def __init__(this, hamiltonian_function, ns):
        this._generator = hamiltonian_function
        this.ns = ns

    def pulse(this, angle):
        return u(angle, this._hamiltonian)

    def dpulse(this, angle):
        return du(angle, this._hamiltonian)

def sbs(ns):
    return [ sideband(sideband_hamiltonian(x), ns).pulse for x in [0,-1,1] ]

def appl(lst):
    g = np.zeros(len(lst[0][0]))
    g[len(g)//2] = 1.0
    return np.dot(reduce(np.dot, lst), g)
