"""Implementations of planar and radial flows.
"""
from typing import Union

from autograd import numpy as np


def planar_flow(z: np.ndarray,
                w: np.ndarray,
                u: np.ndarray,
                b: Union[int, float],
                h=np.tanh) -> np.ndarray:
    """Apply a planar flow to each element of `samples`

    :param z: numpy array, samples to be transformed
        Shape: (n_samples, n_dim)
    :param u: numpy array, parameter of flow (N, D)
    :param w: numpy array, parameter of flow (N, D)
    :param b: numeric, parameter of flow (N,)
    :param h: callable, non-linear function (default tanh)
    :returns: numpy array, transformed samples

    Transforms given samples according to the planar flow
    :math:`f(z) = z + uh(w^Tz + b)`
    """
    assert np.all(np.array([w.shape[0], u.shape[0], b.shape[0]]) == z.shape[0]), 'Incorrect first dimension'
    assert np.all(np.array([w.shape[1], u.shape[1]]) == z.shape[1]), "Incorrect second dimension"

    u = _get_uhat(u, w)
    assert np.all(np.sum(u * w, axis=1)) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
    assert np.all(np.sum(u * w, axis=1)) >= -1, f'Flow is not guaranteed to be invertible (u^Tw < -1: {w._value, u._value})'
    res = z + np.sum(u * np.tanh(np.sum(z * w, axis=1) + b).reshape(-1, 1), axis=1).reshape(z.shape[0], -1)
    assert res.shape == z.shape, f'Incorrect output shape: {(res.shape)}'
    return res


def _get_uhat(u, w):
    N, D = u.shape
    return u + (m(np.sum(w * u, axis=1).reshape(-1, 1)) - np.sum(w * u, axis=1).reshape(-1, 1)) * (w / (np.linalg.norm(w, axis=1).reshape(-1, 1) ** 2)).reshape(N, D)


def m(x):
    return -1 + np.log(1 + np.exp(x))
