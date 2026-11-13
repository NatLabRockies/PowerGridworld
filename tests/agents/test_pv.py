import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import single_agent_episode_runner

from gridworld.agents.pv import PVEnv
from gridworld.agents.pv.pv_profile_env import PROFILE_DIR


def test_default_env(pv_config):
    """Creates an env instance and runs it (sanity check)."""
    env = PVEnv(**pv_config)
    assert single_agent_episode_runner(env)


def _profile(name):
    return pd.read_csv(os.path.join(PROFILE_DIR, name)).iloc[:, 0].values


def test_obs_is_scaled_profile(pv_config):
    """Obs magnitude is scaling_factor times the current csv row, regardless
    of the curtailment action."""
    env = PVEnv(rescale_spaces=False, **pv_config)
    profile = _profile(pv_config["profile_csv"]) * pv_config["scaling_factor"]
    env.reset()
    for k in range(1, 121):
        env.step(np.array([float(k % 2)]))
    obs, _ = env.get_obs()
    assert np.isclose(abs(obs[0]), profile[120])


def test_real_power_is_generation_bounded_by_profile(pv_config):
    env = PVEnv(rescale_spaces=False, **pv_config)
    max_power = pv_config["scaling_factor"] * _profile(pv_config["profile_csv"]).max()
    env.reset()
    done = False
    while not done:
        _, rew, done, _ = env.step(env.action_space.sample())
        assert rew == 0.
        assert -max_power <= env.real_power <= 0.


def test_episode_ends_at_end_of_data(pv_config):
    env = PVEnv(**pv_config)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(env.action_space.high)
        steps += 1
    assert steps == len(_profile(pv_config["profile_csv"])) - 1


def test_profile_path_overrides_profile_csv(pv_config):
    path = os.path.join(PROFILE_DIR, "constant.csv")
    env = PVEnv(rescale_spaces=False, profile_path=path, **pv_config)
    obs, _ = env.reset()
    # pv_profile.csv starts at 0 (night), constant.csv at 1.0.
    assert np.isclose(abs(obs[0]), pv_config["scaling_factor"] * _profile("constant.csv")[0])


def test_rescale_and_grid_aware_spaces(pv_config):
    env = PVEnv(grid_aware=True, rescale_spaces=True, **pv_config)
    assert "min_voltage" in env.obs_labels
    assert env.observation_space.shape == (len(env.obs_labels),)
    for space in [env.observation_space, env.action_space]:
        assert np.all(space.low == -1.) and np.all(space.high == 1.)
    obs, _ = env.reset(min_voltage=0.95)
    assert env.observation_space.contains(obs)


@pytest.mark.xfail(strict=True, reason="step doc: advance index then apply action; obs/real_power returned by step are from the previous csv row (get_obs() right after differs)")
def test_step_obs_matches_get_obs(pv_config):
    env = PVEnv(rescale_spaces=False, **pv_config)
    env.reset()
    for _ in range(110):
        obs, _, _, _ = env.step(np.array([1.]))
    assert np.allclose(obs, env.get_obs()[0])
    assert np.isclose(env.real_power, obs[0])
