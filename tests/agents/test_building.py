import numpy as np
import pandas as pd
import pytest

from tests.conftest import single_agent_episode_runner

from gridworld.agents.buildings import FiveZoneROMEnv, FiveZoneROMThermalEnergyEnv
from gridworld.agents.buildings import five_zone_rom_dynamics as dynamics
from gridworld.agents.buildings.five_zone_rom_env import load_data


def test_default_env(building_config):
    """Creates an env instance and runs it (sanity check)."""
    env = FiveZoneROMThermalEnergyEnv(**building_config)
    assert single_agent_episode_runner(env)


def test_obs_in_space_and_power_is_load(building_config):
    """get_obs clips to the obs space; a building only consumes power."""
    env = FiveZoneROMEnv(**building_config)
    obs = env.reset()
    assert env.observation_space.contains(obs)
    for kind in ["low", "high"]:
        env.reset()
        for _ in range(20):
            action = getattr(env.action_space, kind)
            obs, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert env.real_power >= 0.


@pytest.mark.xfail(strict=True, reason="is_terminal docstring: done when max_episode_steps reached; max_episode_steps=10 ends after 9 steps")
def test_terminal_at_max_episode_steps(building_config):
    env = FiveZoneROMEnv(max_episode_steps=10, **building_config)
    env.reset()
    for _ in range(9):
        _, _, done, _ = env.step(env.action_space.low)
        assert not done
    _, _, done, _ = env.step(env.action_space.low)
    assert done


@pytest.mark.xfail(strict=True, reason="step_reward hint is Tuple[float, dict] and doc says soft comfort penalty; returns a positive shape-(5,) array")
def test_default_reward_is_scalar_penalty(building_config):
    env = FiveZoneROMEnv(**building_config)
    env.reset()
    env.step(env.action_space.high)
    rew, meta = env.step_reward()
    assert np.ndim(rew) == 0
    assert rew <= 0.
    assert isinstance(meta, dict)


def test_comfort_bounds_df_matches_tuple(building_config):
    env = FiveZoneROMEnv(comfort_bounds=(20., 25.), **building_config)
    df = env.make_comfort_bounds_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 2
    assert np.all(df.min(axis=1) == 20.)
    assert np.all(df.max(axis=1) == 25.)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_thermal_energy_alpha_weighting(building_config, alpha):
    env = FiveZoneROMThermalEnergyEnv(
        alpha=alpha, comfort_bounds=(18., 20.), **building_config)
    env.reset()
    env.step(env.action_space.low)
    rew, meta = env.step_reward()
    expected = 0.5 * alpha * meta["energy_rew"] + (1 - alpha) * meta["comfort_rew"]
    assert np.isclose(rew, expected)


def test_thermal_energy_alpha_zero_is_comfort_only(building_config):
    env = FiveZoneROMThermalEnergyEnv(
        alpha=0., comfort_bounds=(18., 20.), **building_config)
    env.reset()
    env.step(env.action_space.low)
    rew, meta = env.step_reward()
    assert meta["comfort_rew"] < 0.
    assert np.isclose(rew, meta["comfort_rew"])


@pytest.mark.xfail(strict=True, reason="step_ doc: returns the stage reward for the applied action; step returns the previous step's reward (first step is always 0)")
def test_thermal_energy_step_reward_matches_step(building_config):
    env = FiveZoneROMThermalEnergyEnv(alpha=1., **building_config)
    env.reset()
    for _ in range(3):
        _, rew, _, _ = env.step(env.action_space.high)
        assert env.real_power > 0.
        assert np.isclose(rew, env.step_reward()[0])


def test_load_data_window():
    df, models = load_data("08-12-2020 00:00:00", "08-13-2020 00:00:00")
    assert isinstance(df, pd.DataFrame)
    assert df.index[0] == pd.Timestamp("2020-08-12 00:00:00")
    assert df.index[-1] <= pd.Timestamp("2020-08-13 00:00:00")
    assert len(models) == dynamics.Z


def test_stage_cost_penalizes_violations_and_tou():
    action = np.array([1., 1., 1., 1., 1., 12.])
    bounds = (22., 28.)

    def cost(temp, tou=1., penalty=10.):
        return dynamics.stage_cost(
            np.ones(dynamics.Z) * temp, action, 30., bounds, tou, penalty)

    in_bounds = cost(25.)
    assert np.ndim(in_bounds) == 0
    assert cost(35.) > in_bounds
    assert cost(15.) > in_bounds
    assert cost(25., tou=2.) > in_bounds
