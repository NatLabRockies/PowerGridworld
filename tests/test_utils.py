import numpy as np
import gymnasium as gym

from gridworld.utils import to_scaled, to_raw, maybe_rescale_box_space


LOW = np.array([0., -5., 10.])
HIGH = np.array([10., 5., 30.])


def test_to_scaled_maps_bounds_to_unit_interval():
    """Docstring: scale [low, high] to [-1, 1]."""
    assert np.allclose(to_scaled(LOW, LOW, HIGH), -1.)
    assert np.allclose(to_scaled(HIGH, LOW, HIGH), 1.)
    assert np.allclose(to_scaled((LOW + HIGH) / 2, LOW, HIGH), 0.)


def test_to_raw_maps_unit_interval_to_bounds():
    """Docstring: scale y in [-1, 1] to [low, high]."""
    assert np.allclose(to_raw(-np.ones(3), LOW, HIGH), LOW)
    assert np.allclose(to_raw(np.ones(3), LOW, HIGH), HIGH)


def test_to_raw_inverts_to_scaled():
    x = np.array([2.5, 0.1, 29.])
    assert np.allclose(to_raw(to_scaled(x, LOW, HIGH), LOW, HIGH), x)


def test_maybe_rescale_box_space():
    """New box has bounds [-1, 1] with the same shape and dtype."""
    box = gym.spaces.Box(low=LOW.astype(np.float32), high=HIGH.astype(np.float32))
    new = maybe_rescale_box_space(box)
    assert new.shape == box.shape
    assert new.dtype == box.dtype
    assert np.all(new.low == -1.) and np.all(new.high == 1.)


def test_maybe_rescale_box_space_pass_through():
    box = gym.spaces.Box(low=LOW, high=HIGH, dtype=np.float64)
    new = maybe_rescale_box_space(box, rescale=False)
    assert np.array_equal(new.low, LOW) and np.array_equal(new.high, HIGH)
