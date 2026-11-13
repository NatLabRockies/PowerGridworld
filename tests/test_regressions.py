"""Targeted tests for specific bugs that the episode-runner smoke tests miss."""

import logging

import numpy as np

from gridworld import MultiAgentEnv
from gridworld.agents.buildings import FiveZoneROMEnv, FiveZoneROMThermalEnergyEnv
from gridworld.agents.pv import PVEnv
from gridworld.distribution_system import OpenDSSSolver
from gridworld.utils import to_scaled


BUILDING_CONFIG = {
    "start_time": "08-12-2020 00:00:00",
    "end_time": "08-13-2020 00:00:00",
}


def test_pv_reset_returns_obs(pv_config):
    obs, meta = PVEnv(**pv_config).reset()
    assert obs.shape == (1,)
    assert "real_power" in meta


def test_to_scaled_warns_above_high(caplog):
    with caplog.at_level(logging.WARNING, logger="default"):
        to_scaled(np.array([5.]), np.array([0.]), np.array([1.]))
    assert "out of bounds" in caplog.text


def test_to_scaled_silent_in_bounds(caplog):
    with caplog.at_level(logging.WARNING, logger="default"):
        to_scaled(np.array([.5]), np.array([0.]), np.array([1.]))
    assert caplog.text == ""


def test_building_step_reward_uses_lower_violations():
    env = FiveZoneROMEnv(**BUILDING_CONFIG)
    env.reset()
    env.state.update({f"zone_upper_viol_{i}": 0. for i in range(5)})
    env.state.update({f"zone_lower_viol_{i}": 2. for i in range(5)})
    rew, _ = env.step_reward()
    assert np.allclose(rew, 4.)


def test_thermal_energy_alpha_is_configurable():
    assert FiveZoneROMThermalEnergyEnv(**BUILDING_CONFIG).alpha == 0.2
    env = FiveZoneROMThermalEnergyEnv(alpha=0., **BUILDING_CONFIG)
    env.reset()
    _, meta = env.step_reward()
    rew, _ = env.step_reward()
    assert rew == meta["comfort_rew"]


def test_bus_voltage_by_name_replaces_only_phase_suffix(opendss_config):
    solver = OpenDSSSolver(**opendss_config)
    solver.bus_voltages = {"a1a.1": 1.01}
    assert solver.get_bus_voltage_by_name("a1aa") == 1.01


def test_multiagent_max_episode_steps(common_config, pf_config, pv_config):
    agents = [{"name": "pv", "bus": "675c", "cls": PVEnv, "config": pv_config}]
    env = MultiAgentEnv(
        common_config=common_config,
        pf_config=pf_config,
        agents=agents,
        max_episode_steps=5,
    )
    env.reset()
    steps = 0
    done = {"__all__": False}
    while not done["__all__"]:
        _, _, done, _ = env.step({"pv": env.action_space["pv"].sample()})
        steps += 1
    assert steps == 5
