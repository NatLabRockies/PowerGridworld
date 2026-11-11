import pytest

from gridworld.distribution_system.powerflow import PowerFlowSolver


def test_abstract_api_must_be_implemented():
    class Partial(PowerFlowSolver):
        def calculate_power_flow(self, p_controllable_consumed=None,
                                 q_controllable_consumed=None, **kwargs):
            pass

    with pytest.raises(TypeError):
        Partial()


def test_minimal_subclass():
    class Flat(PowerFlowSolver):
        def calculate_power_flow(self, p_controllable_consumed=None,
                                 q_controllable_consumed=None, **kwargs):
            self.voltages = {bus: 1. for bus in (p_controllable_consumed or {})}

        def get_bus_voltages(self):
            return self.voltages

        def get_bus_voltage_by_name(self, name):
            return self.voltages[name]

    solver = Flat(config={})
    solver.calculate_power_flow({"a": 1.}, {"a": 0.})
    assert solver.get_bus_voltages() == {"a": 1.}
    assert solver.get_bus_voltage_by_name("a") == 1.
