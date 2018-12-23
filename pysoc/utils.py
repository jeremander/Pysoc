import numpy as np

#############
# UTILITIES #
#############

def flatten(xs):
    """Takes a list of elements and/or lists, and flattens (recursively) to the element level."""
    res = []
    for x in xs:
        if isinstance(x, list):
            res += flatten(x)
        else:
            res.append(x)
    return res

##############
# DECORATORS #
##############

def memoize(f):
    """Memoization decorator for functions taking one or more args or kwargs."""
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args, **kwargs):
            kwargs = tuple(sorted(kwargs.items()))
            return self[(args, kwargs)]
        def __missing__(self, key):
            ret = self[key] = self.f(*key[0], **dict(key[1]))
            return ret
    return memodict(f)

########
# MATH #
########

def normalize_probs(probs):
    """Returns normalized probabilities."""
    probs = np.array(probs)
    assert all(probs >= 0.0), "Negative probabilities not allowed."
    assert any(probs > 0.0), "At least one probability must be positive."
    return probs / probs.sum()

