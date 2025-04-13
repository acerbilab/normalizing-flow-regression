import numpy as np


def logsumexp(x, axis=None, keepdims=False):
    M = np.amax(x, axis=axis, keepdims=keepdims)
    return M + np.log(np.sum(np.exp(x - M), axis=axis, keepdims=keepdims))
