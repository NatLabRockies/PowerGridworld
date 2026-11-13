from tests.conftest import multi_agent_episode_runner

from gridworld import MultiAgentEnv, MultiComponentEnv
from gridworld.agents.energy_storage import EnergyStorageEnv
from gridworld.agents.vehicles import EVChargingEnv
from gridworld.agents.pv import PVEnv


def test_ev_charging_multiagent_env(common_config, ev_charging_config, pf_config):
    """Test multiagent env with 3 single-component EV charging agents."""

    agents = [
        {
            "name": "ev-charging-{}".format(i),
            "bus": "675c",
            "cls": EVChargingEnv,
            "config": ev_charging_config
        } for i in range(3)
    ]

    # Configuration of the multi-agent environment.
    env_config = {
        "common_config": common_config,
        "pf_config": pf_config,
        "agents": agents
    }

    env = MultiAgentEnv(**env_config)

    assert multi_agent_episode_runner(env)


def test_multi_component_building_multiagent_env(common_config, pf_config, multicomponent_building_config):
    """Test multiagent env with three multi-component building agents."""

    agents = [
        {
            "name": "building-{}".format(i),
            "bus": "675c",
            "cls": MultiComponentEnv,
            "config": {"components": multicomponent_building_config},
        } for i in range(3)
    ]

    # Configuration of the multi-agent environment.
    env_config = {
        "common_config": common_config,
        "pf_config": pf_config,
        "agents": agents
    }

    env = MultiAgentEnv(**env_config)

    assert multi_agent_episode_runner(env)


def test_heterogeneous_multiagent_env(
    multicomponent_building_config,
    ev_charging_config,
    pv_array_config,
    common_config,
    pf_config,
):
    """Test multiagent env with three heterogeneous agents."""

    building_agent ={
        "name": "building",
        "bus": "675c",
        "cls": MultiComponentEnv,
        "config": {"components": multicomponent_building_config}
    }

    # Next, build the PV and EV charging envs
    ev_agent = {
        "name": "ev-charging",
        "bus": "675c",
        "cls": EVChargingEnv,
        "config": ev_charging_config
    }

    pv_agent = {
        "name": "pv",
        "bus": "675c",
        "cls": PVEnv,
        "config": pv_array_config
    }

    agents = [building_agent, ev_agent, pv_agent]

    env_config = {
        "common_config": common_config,
        "pf_config": pf_config,
        "agents": agents
    }

    env = MultiAgentEnv(**env_config)

    assert multi_agent_episode_runner(env)


def _pv_storage_env(common_config, pf_config, pv_config, **kwargs):
    agents = [
        {"name": "pv", "bus": "675c", "cls": PVEnv,
         "config": {k: v for k, v in pv_config.items() if k != "name"}},
        {"name": "storage", "bus": "675c", "cls": EnergyStorageEnv, "config": {}},
    ]
    return MultiAgentEnv(
        common_config=common_config, pf_config=pf_config, agents=agents, **kwargs)


def test_step_dicts_and_max_episode_steps(common_config, pf_config, pv_config):
    env = _pv_storage_env(common_config, pf_config, pv_config, max_episode_steps=3)
    obs = env.reset()
    assert set(obs) == {"pv", "storage"}
    for i in range(3):
        action = {n: s.sample() for n, s in env.action_space.items()}
        obs, rew, done, meta = env.step(action)
        assert set(obs) == set(rew) == set(meta) == {"pv", "storage"}
        assert "__all__" in done
        assert done["__all__"] == (i == 2)
        for name, o in obs.items():
            assert env.observation_space[name].contains(o)


def test_transforms_are_pass_through(common_config, pf_config, pv_config):
    env = _pv_storage_env(common_config, pf_config, pv_config)
    rew = {"pv": 1., "storage": -2.}
    meta = {"pv": {"a": 1}, "storage": {}}
    assert env.reward_transform(rew) == rew
    assert env.meta_transform(meta) == meta
