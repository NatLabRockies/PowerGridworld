# Examples of creating envs and running MARL training algorithms

## Environments

The notebooks in [`envs`](./envs) walk through building and stepping each
kind of environment.  They run in CI, so they should stay in sync with the
library.

| Notebook | What it shows |
| --- | --- |
| [`building.ipynb`](./envs/building.ipynb) | Five-zone building thermal model |
| [`pv.ipynb`](./envs/pv.ipynb) | PV array driven by a static generation profile |
| [`energy-storage.ipynb`](./envs/energy-storage.ipynb) | Battery energy storage |
| [`ev-charging.ipynb`](./envs/ev-charging.ipynb) | EV charging station |
| [`multi-component-building.ipynb`](./envs/multi-component-building.ipynb) | One agent composed of building + PV + storage (`MultiComponentEnv`) |
| [`multiagent-single-component.ipynb`](./envs/multiagent-single-component.ipynb) | Several single-component agents on a feeder (`MultiAgentEnv`) |
| [`multiagent-multi-component.ipynb`](./envs/multiagent-multi-component.ipynb) | Several multi-component agents on a feeder |
| [`multiagent-heterogeneous.ipynb`](./envs/multiagent-heterogeneous.ipynb) | Building, PV, and EV agents together |
| [`multiagent-list-interface-wrapper.ipynb`](./envs/multiagent-list-interface-wrapper.ipynb) | List-based interface for frameworks that don't take dicts |

Run them all with:

```
uv run pytest --nbmake examples/envs
```

## MARL training

- [`marl/rllib`](./marl/rllib):  PPO with RLlib on the heterogeneous scenario.
- [`marl/openai`](./marl/openai):  MADDPG with OpenAI's implementation.  This
  needs an old Python/TensorFlow stack; see the README there before trying it.
