import numpy as np
import pytest

from gridworld.distribution_system import OpenDSSSolver


def test_default_opendss(opendss_config):

    opendss = OpenDSSSolver(**opendss_config)

    current_time = "01-01-2021 05:00:00"
    opendss.calculate_power_flow(current_time=current_time)
    
    assert opendss.get_bus_voltages()


TIME = "01-01-2021 05:00:00"


@pytest.fixture
def solved(opendss_config):
    solver = OpenDSSSolver(**opendss_config)
    solver.calculate_power_flow(current_time=TIME)
    return solver


def test_bus_voltages_are_per_unit_floats(solved):
    voltages = solved.get_bus_voltages()
    assert isinstance(voltages, dict)
    assert all(0.9 < float(v) < 1.1 for v in voltages.values())


def test_voltage_by_name_three_phase_and_single_phase(solved):
    voltages = solved.get_bus_voltages()
    v675 = solved.get_bus_voltage_by_name("675")
    assert isinstance(v675, list) and len(v675) == 3
    assert np.allclose(v675, [voltages[f"675.{i}"] for i in (1, 2, 3)])
    assert np.isclose(solved.get_bus_voltage_by_name("675c"), voltages["675.3"])


@pytest.mark.xfail(strict=True, reason="doc: non-3-phase bus returns a single float; bare single-phase bus '611' raises KeyError('611.1')")
def test_voltage_by_name_single_phase_bus(solved):
    v = solved.get_bus_voltage_by_name("611")
    assert np.isclose(v, solved.get_bus_voltages()["611.3"])


def test_controllable_load_sign(opendss_config):
    """p consumed > 0 is load (voltage drops), < 0 is generation (rises)."""
    solver = OpenDSSSolver(**opendss_config)
    solver.calculate_power_flow(current_time=TIME)
    base = solver.get_bus_voltage_by_name("675c")
    solver.calculate_power_flow(
        p_controllable_consumed={"675c": 500.},
        q_controllable_consumed={"675c": 0.}, current_time=TIME)
    assert solver.get_bus_voltage_by_name("675c") < base
    solver.calculate_power_flow(
        p_controllable_consumed={"675c": -500.},
        q_controllable_consumed={"675c": 0.}, current_time=TIME)
    assert solver.get_bus_voltage_by_name("675c") > base


def test_system_load_rescale_factor(opendss_config):
    min_v = []
    for factor in [0.3, 1.0]:
        solver = OpenDSSSolver(**{**opendss_config, "system_load_rescale_factor": factor})
        solver.calculate_power_flow(current_time=TIME)
        min_v.append(min(solver.get_bus_voltages().values()))
    assert min_v[0] > min_v[1]


@pytest.mark.xfail(strict=True, reason="q_controllable_consumed defaults to None but passing only p raises TypeError")
def test_p_only_controllable_load(opendss_config):
    solver = OpenDSSSolver(**opendss_config)
    solver.calculate_power_flow(
        p_controllable_consumed={"675c": 100.}, current_time=TIME)
    assert solver.get_bus_voltages()


@pytest.mark.xfail(strict=True, reason="current_time defaults to None but calling without it raises TypeError")
def test_default_current_time(opendss_config):
    solver = OpenDSSSolver(**opendss_config)
    solver.calculate_power_flow()
    assert solver.get_bus_voltages()
