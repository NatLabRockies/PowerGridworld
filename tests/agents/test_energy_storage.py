import numpy as np
import pytest

from tests.conftest import single_agent_episode_runner
from gridworld.agents.energy_storage import EnergyStorageEnv


def test_default_env(energy_storage_config):
    """Creates an env instance and runs it (sanity check)."""
    env = EnergyStorageEnv(**energy_storage_config)
    assert single_agent_episode_runner(env)


def _fixed_soc_env(soc, **kwargs):
    return EnergyStorageEnv(
        initial_storage_mean=soc, initial_storage_std=0., rescale_spaces=False,
        **kwargs)


def test_action_sign_convention():
    """Action -1 fully charges (load, real_power > 0), +1 fully discharges
    (generation, real_power < 0)."""
    env = _fixed_soc_env(30.)

    env.reset()
    obs, _, _, _ = env.step(np.array([-1.]))
    assert obs[0] > 30.
    assert env.real_power > 0.

    env.reset()
    obs, _, _, _ = env.step(np.array([1.]))
    assert obs[0] < 30.
    assert env.real_power < 0.


def test_reward_is_zero_and_soc_stays_in_range():
    env = EnergyStorageEnv(rescale_spaces=False, storage_range=(3., 50.))
    env.reset()
    done = False
    while not done:
        obs, rew, done, _ = env.step(env.action_space.sample())
        assert rew == 0.
        assert 3. <= obs[0] <= 50.
        assert env.observation_space.contains(obs)


def test_validate_power_at_soc_limits():
    """Positive power discharges, negative charges; neither is feasible past
    the corresponding SOC limit."""
    env = _fixed_soc_env(3.)
    env.reset()
    assert env.validate_power(10.) == 0.
    assert env.validate_power(-10.) == -10.

    env = _fixed_soc_env(50.)
    env.reset()
    assert env.validate_power(-10.) == 0.
    assert env.validate_power(10.) == 10.


def test_episode_length():
    env = EnergyStorageEnv(max_episode_steps=5)
    env.reset()
    dones = [env.step(np.array([0.]))[2] for _ in range(5)]
    assert dones == [False] * 4 + [True]


@pytest.mark.xfail(strict=True, reason="reset samples SOC from initial_storage_mean (default 30) without clipping to storage_range, so obs falls outside observation_space")
def test_reset_obs_in_space_with_small_storage_range():
    env = EnergyStorageEnv(storage_range=(3., 20.), rescale_spaces=False)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
