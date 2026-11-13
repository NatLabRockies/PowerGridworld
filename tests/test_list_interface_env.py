import numpy as np
import pytest

from gridworld import MultiAgentEnv
from gridworld.multiagent_list_interface_env import MultiAgentListInterfaceEnv
from gridworld.scenarios import heterogeneous


@pytest.fixture
def list_env(buildings_scenario_config):
    return MultiAgentListInterfaceEnv(MultiAgentEnv, buildings_scenario_config)


def test_spaces_are_lists_of_flattened_agent_spaces(list_env):
    ma_env = list_env.ma_env
    agents = list(ma_env.agent_dict)
    assert isinstance(list_env.observation_space, list)
    assert isinstance(list_env.action_space, list)
    assert len(list_env.observation_space) == len(list_env.action_space) == len(agents)
    for i, name in enumerate(agents):
        obs_dim = sum(s.shape[0] for s in ma_env.observation_space[name].values())
        act_dim = sum(s.shape[0] for s in ma_env.action_space[name].values())
        assert list_env.observation_space[i].shape == (obs_dim,)
        assert list_env.action_space[i].shape == (act_dim,)


def test_convert_to_list_obs(list_env):
    obs_dict = list_env.ma_env.reset()
    obs_list = list_env.convert_to_list_obs(obs_dict)
    assert len(obs_list) == len(obs_dict)
    for obs, (name, agent_obs) in zip(obs_list, obs_dict.items()):
        assert np.array_equal(obs, np.concatenate(list(agent_obs.values())))


def test_convert_from_list_act_round_trip(list_env):
    acts = [s.sample() for s in list_env.action_space]
    act_dict = list_env.convert_from_list_act(acts)
    assert list(act_dict) == list(list_env.ma_env.agent_dict)
    for act, (name, agent_act) in zip(acts, act_dict.items()):
        assert list_env.ma_env.action_space[name].contains(agent_act)
        assert np.allclose(act, np.concatenate(list(agent_act.values())))


def test_reset_and_step_return_lists(list_env):
    obs = list_env.reset()
    assert all(s.contains(o) for s, o in zip(list_env.observation_space, obs))
    acts = [s.sample() for s in list_env.action_space]
    obs, rew, done, _ = list_env.step(acts)
    n = len(list_env.action_space)
    assert len(obs) == len(rew) == len(done) == n
    assert all(s.contains(o) for s, o in zip(list_env.observation_space, obs))


@pytest.mark.xfail(strict=True, reason="wrapper claims to wrap MultiAgentEnv (which accepts ComponentEnv agents) but raises KeyError('components') on any non-MultiComponentEnv agent")
def test_wraps_single_component_agents():
    env = MultiAgentListInterfaceEnv(
        MultiAgentEnv, heterogeneous.make_env_config())
    assert len(env.reset()) == 3
