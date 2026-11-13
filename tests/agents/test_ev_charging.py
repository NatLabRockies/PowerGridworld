import pytest

from tests.conftest import single_agent_episode_runner
from gridworld.agents.vehicles import EVChargingEnv


def test_default_env(ev_charging_config):
    """Creates an env instance and runs it (sanity check)."""
    env = EVChargingEnv(**ev_charging_config)
    assert single_agent_episode_runner(env)


def test_real_power_is_load(ev_charging_config):
    env = EVChargingEnv(**{**ev_charging_config, "vehicle_multiplier": 1})
    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
        assert env.real_power >= 0.


@pytest.mark.xfail(strict=True, reason="is_terminal doc: True when max episode steps reached; episode ends after max_episode_steps - 2 steps (10 -> 8)")
def test_terminal_at_max_episode_steps(ev_charging_config):
    env = EVChargingEnv(**{**ev_charging_config, "max_episode_steps": 10})
    env.reset()
    dones = [env.step(env.action_space.low)[2] for _ in range(10)]
    assert dones == [False] * 9 + [True]


@pytest.mark.xfail(strict=True, reason="obs space bounds num_active_vehicles by num_vehicles, ignoring vehicle_multiplier; obs reaches 110 vs bound 100 with multiplier 5")
def test_obs_in_space_with_vehicle_multiplier(ev_charging_config):
    env = EVChargingEnv(**ev_charging_config)
    obs, _ = env.reset()
    done = False
    while not done:
        assert env.observation_space.contains(obs)
        obs, _, done, _ = env.step(env.action_space.high)
