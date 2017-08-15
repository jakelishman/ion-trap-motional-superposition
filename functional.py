def scan(f, lst):
    """
    Returns the list
        [ lst[0], f(lst[0], lst[1]), f(f(lst[0], lst[1]), lst[2]), ... ].
    """
    out = lst[:1]
    prev = out[0] if out != [] else None
    for x in lst[1:]:
        prev = f(prev, x)
        out.append(prev)
    return out

def scan_back(f, lst):
    """
    For use with non-commutative `f`, this returns the list
        [ ..., f(lst[n-2], f(lst[n-1], lst[n])), f(lst[n-1], lst[n]), lst[n] ],
    which is different to `scan(f, lst[::-1])[::-1]` when `f(a, b) != f(b, a)`.

    Has particular use for associative `f`, which implies
        scan_back(f, lst)[0] == scan(f, lst)[-1],
    i.e. the element found by the total reduction of each scan is the same.
    """
    out = lst[-1:]
    prev = out[0] if out != [] else None
    for x in lst[-2::-1]:
        prev = f(x, prev) # note reversed order of parameters
        out.append(prev)
    return out[::-1]

def pairs(lst):
    """
    Make pairs out a list, so
        [1, 2, 3, 4, 5]
    becomes
        [(1, 2), (2, 3), (3, 4), (4, 5)].
    """
    if len(lst) <= 1:
        return []
    prev = lst[0]
    out = []
    for el in lst[1:]:
        out.append((prev, el))
        prev = el
    return out

def map2(f, lst1, lst2):
    """Map function extended to 2 lists, so f takes 2 arguments."""
    assert len(lst1) == len(lst2)
    out = []
    for i in xrange(len(lst1)):
        out.append(f(lst1[i], lst2[i]))
    return out

def exists(predicate, iterable):
    for element in iterable:
        if predicate(element):
            return True
    return False

def map3(f, lst1, lst2, lst3):
    """Map function extended to 3 lists, so f takes 3 arguments."""
    assert len(lst1) == len(lst2) == len(lst3)
    out = []
    for i in xrange(len(lst1)):
        out.append(f(lst1[i], lst2[i], lst3[i]))
    return out
