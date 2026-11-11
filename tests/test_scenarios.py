import numpy as np
import pytest

from gridworld import MultiAgentEnv
from gridworld.scenarios import buildings, heterogeneous


def test_buildings_config_structure():
    config = buildings.make_env_config(
        building_config={}, pv_config={"profile_csv": "off-peak.csv"},
        storage_config={}, system_load_rescale_factor=0.5, num_buildings=4)
    assert len(config["agents"]) == 4
    assert config["pf_config"]["config"]["system_load_rescale_factor"] == 0.5
    for agent in config["agents"]:
        names = [c["name"] for c in agent["config"]["components"]]
        assert names == ["building", "pv", "storage"]


def test_buildings_config_omits_unconfigured_components():
    config = buildings.make_env_config(building_config={})
    for agent in config["agents"]:
        assert [c["name"] for c in agent["config"]["components"]] == ["building"]


@pytest.mark.xfail(strict=True, reason="default building_config=None is passed through as the component config, so MultiAgentEnv(**make_env_config()) raises TypeError")
def test_buildings_default_config_builds_env():
    env = MultiAgentEnv(**buildings.make_env_config())
    assert len(env.agent_dict) == 3


@pytest.mark.parametrize("rescale", [True, False])
def test_heterogeneous_config_builds_env(rescale):
    config = heterogeneous.make_env_config(
        system_load_rescale_factor=0.5, rescale_spaces=rescale)
    assert config["pf_config"]["config"]["system_load_rescale_factor"] == 0.5
    env = MultiAgentEnv(**config)
    spaces = []
    for name in env.agent_dict:
        for space in [env.observation_space[name], env.action_space[name]]:
            spaces.extend(space.values() if hasattr(space, "spaces") else [space])
    is_unit = [np.all(s.low == -1.) and np.all(s.high == 1.) for s in spaces]
    assert all(is_unit) if rescale else not all(is_unit)
