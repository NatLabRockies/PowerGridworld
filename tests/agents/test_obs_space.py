import numpy as np
import pytest

from gridworld.agents.buildings.obs_space import (
    make_obs_space, DEFAULT_OBS_CONFIG, MULTIZONE_KEYS, ZONE_TEMP_BOUNDS,
    OUTDOOR_TEMP_BOUNDS
)


def test_default_config_dims_and_labels():
    """Multizone keys get one dimension per zone, the rest get one each."""
    num_zones = 5
    space, labels = make_obs_space(num_zones, DEFAULT_OBS_CONFIG)

    expected_dim = (
        num_zones * len(MULTIZONE_KEYS)
        + len(DEFAULT_OBS_CONFIG) - len(MULTIZONE_KEYS)
    )
    assert space.shape == (expected_dim,)
    assert len(labels) == expected_dim
    assert len(set(labels)) == expected_dim


def test_bounds_follow_config():
    config = {"zone_temp": ZONE_TEMP_BOUNDS, "outdoor_temp": OUTDOOR_TEMP_BOUNDS}
    space, labels = make_obs_space(3, config)

    assert space.shape == (4,)
    zone_idx = [i for i, label in enumerate(labels) if label.startswith("zone_temp")]
    oa_idx = [i for i, label in enumerate(labels) if label == "outdoor_temp"]
    assert len(zone_idx) == 3 and len(oa_idx) == 1
    assert np.all(space.low[zone_idx] == ZONE_TEMP_BOUNDS[0])
    assert np.all(space.high[zone_idx] == ZONE_TEMP_BOUNDS[1])
    assert space.low[oa_idx[0]] == OUTDOOR_TEMP_BOUNDS[0]
    assert space.high[oa_idx[0]] == OUTDOOR_TEMP_BOUNDS[1]


def test_custom_bounds_override_defaults():
    space, _ = make_obs_space(2, {"zone_temp": (18., 34.)})
    assert np.all(space.low == 18.) and np.all(space.high == 34.)


def test_invalid_key_raises():
    with pytest.raises(Exception):
        make_obs_space(5, {"not_a_real_key": (0., 1.)})
