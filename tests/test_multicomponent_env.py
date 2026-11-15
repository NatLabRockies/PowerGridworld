import numpy as np

from tests.conftest import multi_agent_episode_runner
from gridworld import MultiComponentEnv


def test_default_multicomponent_env(multicomponent_building_config):

    env = MultiComponentEnv(components=multicomponent_building_config)

    assert multi_agent_episode_runner(env)


def test_spaces_are_union_of_components(multicomponent_building_config):
    env = MultiComponentEnv(components=multicomponent_building_config)
    names = [c["name"] for c in multicomponent_building_config]
    assert list(env.env_dict.keys()) == names
    for name, comp in env.env_dict.items():
        assert env.observation_space[name] == comp.observation_space
        assert env.action_space[name] == comp.action_space
        assert env.obs_labels_dict[name] == comp.obs_labels
    assert set(env.obs_labels) == {
        label for comp in env.env_dict.values() for label in comp.obs_labels}


def test_step_composes_component_results(multicomponent_building_config):
    env = MultiComponentEnv(components=multicomponent_building_config)
    env.reset()
    for _ in range(5):
        obs, rew, done, meta = env.step(env.action_space.sample())
    assert set(obs) == set(env.env_dict) == set(meta)
    assert env.observation_space.contains(obs)
    assert isinstance(done, (bool, np.bool_))
    # Default step reward is the sum over components.
    total, _ = env.step_reward()
    assert np.isclose(rew, total)
    assert np.isclose(total, sum(c.step_reward()[0] for c in env.env_dict.values()))
