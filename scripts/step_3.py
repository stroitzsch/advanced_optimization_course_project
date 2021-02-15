"""Step 3: Solving the problem as a bilevel program."""

import cvxpy as cp
import fledge
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shutil


def main():

    # Settings.
    scenario_name = 'course_project_step_3'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_3')

    # Clear / instantiate results directory.
    try:
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)
    except PermissionError:
        pass

    # STEP 3.0: SETUP MODELS.

    # Read scenario definition into FLEDGE.
    # - Data directory from this repository is first added as additional data path.
    fledge.config.config['paths']['additional_data'].append(
        os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'data')
    )
    fledge.data_interface.recreate_database()

    # Obtain data & models.

    # Flexible loads.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Thermal grid.
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_grid_model.cooling_plant_efficiency = 10.0  # Change model parameter to incentivize use of thermal grid.
    thermal_power_flow_solution_reference = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(thermal_grid_model, thermal_power_flow_solution_reference)
    )
    # Define arbitrary operation limits.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution_reference.node_head_vector
    branch_flow_vector_maximum = 10.0 * thermal_power_flow_solution_reference.branch_flow_vector

    # Electric grid.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution_reference = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(electric_grid_model, power_flow_solution_reference)
    )
    # Define arbitrary operation limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference

    # Energy price.
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain time step index shorthands.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    timesteps = scenario_data.timesteps
    timestep_interval_hours = (timesteps[1] - timesteps[0]) / pd.Timedelta('1h')

    # Invert sign of losses.
    # - Power values of loads are negative by convention. Hence, sign of losses should be negative for power balance.

    # Thermal grid.
    linear_thermal_grid_model.sensitivity_pump_power_by_der_power *= -1.0
    linear_thermal_grid_model.thermal_power_flow_solution.pump_power *= -1.0

    # Electric grid.
    linear_electric_grid_model.sensitivity_loss_active_by_der_power_active *= -1.0
    linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive *= -1.0
    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active *= -1.0
    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive *= -1.0
    linear_electric_grid_model.power_flow_solution.loss *= -1.0

    # Apply base power / voltage scaling.
    # - Scale values to avoid numerical issues.
    base_power = 1e6  # in MW.
    base_voltage = 1e3  # in kV.

    # Flexible loads.
    for der_model in der_model_set.flexible_der_models.values():
        der_model.mapping_active_power_by_output *= 1 / base_power
        der_model.mapping_reactive_power_by_output *= 1 / base_power
        der_model.mapping_thermal_power_by_output *= 1 / base_power

    # Thermal grid.
    linear_thermal_grid_model.sensitivity_node_head_by_der_power *= base_power
    linear_thermal_grid_model.sensitivity_branch_flow_by_der_power *= base_power
    linear_thermal_grid_model.sensitivity_pump_power_by_der_power *= 1

    # Electric grid.
    linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active *= base_power / base_voltage
    linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive *= base_power / base_voltage
    linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active *= 1
    linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive *= 1
    linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active *= 1
    linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive *= 1
    linear_electric_grid_model.sensitivity_loss_active_by_der_power_active *= 1
    linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive *= 1
    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active *= 1
    linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive *= 1
    linear_electric_grid_model.power_flow_solution.der_power_vector *= 1 / base_power
    linear_electric_grid_model.power_flow_solution.branch_power_vector_1 *= 1 / base_power
    linear_electric_grid_model.power_flow_solution.branch_power_vector_2 *= 1 / base_power
    linear_electric_grid_model.power_flow_solution.loss *= 1 / base_power
    linear_electric_grid_model.power_flow_solution.node_voltage_vector *= 1 / base_voltage

    # Limits
    node_voltage_magnitude_vector_minimum /= base_voltage
    node_voltage_magnitude_vector_maximum /= base_voltage
    branch_power_magnitude_vector_maximum /= base_power

    # Energy price.
    # - Conversion of price values from S$/kWh to S$/p.u. for convenience. Currency S$ is SGD.
    # - Power values of loads are negative by convention. Hence, sign of price values is inverted here.
    price_data.price_timeseries *= -1.0 * base_power / 1e3 * timestep_interval_hours

    # STEP 3.1: SOLVE MPEC PROBLEM.

    # Instantiate problem.
    # - Utility object for optimization problem definition with CVXPY.
    problem = fledge.utils.OptimizationProblem()

    # Define variables.

    # Flexible loads: State space vectors.
    # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
    problem.state_vector = dict.fromkeys(der_model_set.flexible_der_names)
    problem.control_vector = dict.fromkeys(der_model_set.flexible_der_names)
    problem.output_vector = dict.fromkeys(der_model_set.flexible_der_names)
    for der_name in der_model_set.flexible_der_names:
        problem.state_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].states)
            ))
        )
        problem.control_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].controls)
            ))
        )
        problem.output_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].outputs)
            ))
        )

    # Flexible loads: Power vectors.
    problem.der_thermal_power_vector = (
        cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
    )
    problem.der_active_power_vector = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )
    problem.der_reactive_power_vector = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )
    
    # Source variables.
    problem.source_thermal_power = cp.Variable((len(timesteps), 1))
    problem.source_active_power = cp.Variable((len(timesteps), 1))
    problem.source_reactive_power = cp.Variable((len(timesteps), 1))

    # Flexible loads: Power equations.
    problem.lambda_thermal_power_equation = (
        cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
    )
    problem.lambda_active_power_equation = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )
    problem.lambda_reactive_power_equation = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )

    # Thermal grid.
    problem.mu_node_head_minium = (
        cp.Variable((len(timesteps), len(thermal_grid_model.nodes)), nonneg=True)
    )
    problem.mu_branch_flow_maximum = (
        cp.Variable((len(timesteps), len(thermal_grid_model.branches)), nonneg=True)
    )
    problem.lambda_pump_power_equation = (
        cp.Variable((len(timesteps), 1))
    )

    # Electric grid.
    problem.mu_node_voltage_magnitude_minimum = (
        cp.Variable((len(timesteps), len(electric_grid_model.nodes)), nonneg=True)
    )
    problem.mu_node_voltage_magnitude_maximum = (
        cp.Variable((len(timesteps), len(electric_grid_model.nodes)), nonneg=True)
    )
    problem.mu_branch_power_magnitude_maximum_1 = (
        cp.Variable((len(timesteps), len(electric_grid_model.branches)), nonneg=True)
    )
    problem.mu_branch_power_magnitude_maximum_2 = (
        cp.Variable((len(timesteps), len(electric_grid_model.branches)), nonneg=True)
    )
    problem.lambda_loss_active_equation = cp.Variable((len(timesteps), 1))
    problem.lambda_loss_reactive_equation = cp.Variable((len(timesteps), 1))
    
    # Define complementarity binary variables.
    problem.psi_node_head_minium = cp.Variable(problem.mu_node_head_minium.shape, boolean=True)
    problem.psi_branch_flow_maximum = cp.Variable(problem.mu_branch_flow_maximum.shape, boolean=True)
    problem.psi_node_voltage_magnitude_minimum = cp.Variable(problem.mu_node_voltage_magnitude_minimum.shape, boolean=True)
    problem.psi_node_voltage_magnitude_maximum = cp.Variable(problem.mu_node_voltage_magnitude_maximum.shape, boolean=True)
    problem.psi_branch_power_magnitude_maximum_1 = cp.Variable(problem.mu_branch_power_magnitude_maximum_1.shape, boolean=True)
    problem.psi_branch_power_magnitude_maximum_2 = cp.Variable(problem.mu_branch_power_magnitude_maximum_2.shape, boolean=True)

    # Define complementarity big M parameters.
    # - Big M values are chosen based on expected order of magnitude of constraints from primal / dual solution.
    problem.big_m_node_head_minium = cp.Parameter(value=1e2)
    problem.big_m_branch_flow_maximum = cp.Parameter(value=1e3)
    problem.big_m_node_voltage_magnitude_minimum = cp.Parameter(value=1e2)
    problem.big_m_node_voltage_magnitude_maximum = cp.Parameter(value=1e2)
    problem.big_m_branch_power_magnitude_maximum_1 = cp.Parameter(value=1e3)
    problem.big_m_branch_power_magnitude_maximum_2 = cp.Parameter(value=1e3)

    # Define upper-level constraints.

    # Flexible loads.
    for der_model in der_model_set.flexible_der_models.values():

        # Initial state.
        problem.constraints.append(
            problem.state_vector[der_model.der_name][0, :]
            ==
            der_model.state_vector_initial.values
        )

        # State equation.
        problem.constraints.append(
            problem.state_vector[der_model.der_name][1:, :]
            ==
            cp.transpose(
                der_model.state_matrix.values
                @ cp.transpose(problem.state_vector[der_model.der_name][:-1, :])
                + der_model.control_matrix.values
                @ cp.transpose(problem.control_vector[der_model.der_name][:-1, :])
                + der_model.disturbance_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
            )
        )

        # Output equation.
        problem.constraints.append(
            problem.output_vector[der_model.der_name]
            ==
            cp.transpose(
                der_model.state_output_matrix.values
                @ cp.transpose(problem.state_vector[der_model.der_name])
                + der_model.control_output_matrix.values
                @ cp.transpose(problem.control_vector[der_model.der_name])
                + der_model.disturbance_output_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.values)
            )
        )

        # Output limits.
        problem.constraints.append(
            problem.output_vector[der_model.der_name]
            >=
            der_model.output_minimum_timeseries.values
        )
        problem.constraints.append(
            problem.output_vector[der_model.der_name]
            <=
            der_model.output_maximum_timeseries.replace(np.inf, 1e3).values
        )

        # Power mapping.
        der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
        problem.constraints.append(
            problem.der_active_power_vector[:, [der_index]]
            ==
            cp.transpose(
                der_model.mapping_active_power_by_output.values
                @ cp.transpose(problem.output_vector[der_model.der_name])
            )
        )
        problem.constraints.append(
            problem.der_reactive_power_vector[:, [der_index]]
            ==
            cp.transpose(
                der_model.mapping_reactive_power_by_output.values
                @ cp.transpose(problem.output_vector[der_model.der_name])
            )
        )
        problem.constraints.append(
            problem.der_thermal_power_vector[:, [der_index]]
            ==
            cp.transpose(
                der_model.mapping_thermal_power_by_output.values
                @ cp.transpose(problem.output_vector[der_model.der_name])
            )
        )

    # Define lower-level constraints.

    # Differential with respect to thermal power vector.
    problem.constraints.append(
        0.0
        ==
        (
            problem.lambda_thermal_power_equation
            - (
                problem.mu_node_head_minium
                @ linear_thermal_grid_model.sensitivity_node_head_by_der_power
            )
            + (
                problem.mu_branch_flow_maximum
                @ linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
            )
            - (
                problem.lambda_pump_power_equation
                @ (
                    thermal_grid_model.cooling_plant_efficiency ** -1
                    * np.ones(linear_thermal_grid_model.sensitivity_pump_power_by_der_power.shape)
                    + linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                )
            )
        )
    )

    # Differential with respect to active power vector.
    problem.constraints.append(
        0.0
        ==
        (
            problem.lambda_active_power_equation
            - (
                problem.mu_node_voltage_magnitude_minimum
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
            )
            + (
                problem.mu_node_voltage_magnitude_maximum
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
            )
            + (
                problem.mu_branch_power_magnitude_maximum_1
                @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
            )
            + (
                problem.mu_branch_power_magnitude_maximum_2
                @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
            )
            - (
                problem.lambda_loss_active_equation
                @ (
                    np.ones(linear_electric_grid_model.sensitivity_loss_active_by_der_power_active.shape)
                    + linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                )
            )
            - (
                problem.lambda_loss_reactive_equation
                @ linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            )
        )
    )

    # Differential with respect to reactive power vector.
    problem.constraints.append(
        0.0
        ==
        (
            problem.lambda_reactive_power_equation
            - (
                problem.mu_node_voltage_magnitude_minimum
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
            )
            + (
                problem.mu_node_voltage_magnitude_maximum
                @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
            )
            + (
                problem.mu_branch_power_magnitude_maximum_1
                @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
            )
            + (
                problem.mu_branch_power_magnitude_maximum_2
                @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
            )
            - (
                problem.lambda_loss_active_equation
                @ linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            )
            - (
                problem.lambda_loss_reactive_equation
                @ (
                    np.ones(linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive.shape)
                    + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                )
            )
        )
    )

    # Differential with respect to thermal source power.
    problem.constraints.append(
        0.0
        ==
        (
            np.transpose([price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values])
            + problem.lambda_pump_power_equation
        )
    )

    # Differential with respect to active source power.
    problem.constraints.append(
        0.0
        ==
        (
            np.transpose([price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values])
            + problem.lambda_loss_active_equation
        )
    )

    # Differential with respect to active source power.
    problem.constraints.append(
        0.0
        ==
        (
            problem.lambda_loss_reactive_equation
        )
    )

    # Thermal power balance.
    problem.constraints.append(
        thermal_grid_model.cooling_plant_efficiency ** -1
        * (
            problem.source_thermal_power
            + cp.sum(-1.0 * (
                problem.der_thermal_power_vector
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        ==
        cp.transpose(
            linear_thermal_grid_model.sensitivity_pump_power_by_der_power
            @ cp.transpose(problem.der_thermal_power_vector)
        )
    )

    # Active power balance.
    problem.constraints.append(
        problem.source_active_power
        + cp.sum(-1.0 * (
            problem.der_active_power_vector
        ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        ==
        np.real(linear_electric_grid_model.power_flow_solution.loss)
        + cp.transpose(
            linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
            @ cp.transpose(
                problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            @ cp.transpose(
                problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
    )

    # Reactive power balance.
    problem.constraints.append(
        problem.source_reactive_power
        + cp.sum(-1.0 * (
            problem.der_reactive_power_vector
        ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        ==
        np.imag(linear_electric_grid_model.power_flow_solution.loss)
        + cp.transpose(
            linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            @ cp.transpose(
                problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
            @ cp.transpose(
                problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
    )
    
    # Define complementarity constraints.

    # Thermal grid.

    # Node head limit.
    problem.constraints.append(
        -1.0
        * (
            np.array([node_head_vector_minimum.ravel()])
            - cp.transpose(
                linear_thermal_grid_model.sensitivity_node_head_by_der_power
                @ cp.transpose(problem.der_thermal_power_vector)
            )
        )
        <=
        problem.psi_node_head_minium
        * problem.big_m_node_head_minium
    )
    problem.constraints.append(
        problem.mu_node_head_minium
        <=
        (1 - problem.psi_node_head_minium)
        * problem.big_m_node_head_minium
    )

    # Branch flow limit.
    problem.constraints.append(
        -1.0
        * (
            cp.transpose(
                linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                @ cp.transpose(problem.der_thermal_power_vector)
            )
            - np.array([branch_flow_vector_maximum.ravel()])
        )
        <=
        problem.psi_branch_flow_maximum
        * problem.big_m_branch_flow_maximum
    )
    problem.constraints.append(
        problem.mu_branch_flow_maximum
        <=
        (1 - problem.psi_branch_flow_maximum)
        * problem.big_m_branch_flow_maximum
    )

    # Voltage limits.

    problem.constraints.append(
        -1.0
        * (
            np.array([node_voltage_magnitude_vector_minimum.ravel()])
            - np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            - cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )
        <=
        problem.psi_node_voltage_magnitude_minimum
        * problem.big_m_node_voltage_magnitude_minimum
    )
    problem.constraints.append(
        problem.mu_node_voltage_magnitude_minimum
        <=
        (1 - problem.psi_node_voltage_magnitude_minimum)
        * problem.big_m_node_voltage_magnitude_minimum
    )

    problem.constraints.append(
        -1.0
        * (
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            - np.array([node_voltage_magnitude_vector_maximum.ravel()])
        )
        <=
        problem.psi_node_voltage_magnitude_maximum
        * problem.big_m_node_voltage_magnitude_maximum
    )
    problem.constraints.append(
        problem.mu_node_voltage_magnitude_maximum
        <=
        (1 - problem.psi_node_voltage_magnitude_maximum)
        * problem.big_m_node_voltage_magnitude_maximum
    )

    # Branch flow limits.

    problem.constraints.append(
        -1.0
        * (
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(
                    problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(
                    problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            - np.array([branch_power_magnitude_vector_maximum.ravel()])
        )
        <=
        problem.psi_branch_power_magnitude_maximum_1
        * problem.big_m_branch_power_magnitude_maximum_1
    )
    problem.constraints.append(
        problem.mu_branch_power_magnitude_maximum_1
        <=
        (1 - problem.psi_branch_power_magnitude_maximum_1)
        * problem.big_m_branch_power_magnitude_maximum_1
    )

    problem.constraints.append(
        -1.0
        * (
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(
                    problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(
                    problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            - np.array([branch_power_magnitude_vector_maximum.ravel()])
        )
        <=
        problem.psi_branch_power_magnitude_maximum_2
        * problem.big_m_branch_power_magnitude_maximum_2
    )
    problem.constraints.append(
        problem.mu_branch_power_magnitude_maximum_2
        <=
        (1 - problem.psi_branch_power_magnitude_maximum_2)
        * problem.big_m_branch_power_magnitude_maximum_2
    )

    # Define objective.
    problem.objective += (
        price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ problem.source_thermal_power
        * thermal_grid_model.cooling_plant_efficiency ** -1
    )
    problem.objective += (
        price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ problem.source_active_power
    )

    # Solve problem.
    fledge.utils.log_time('primal solution')
    problem.solve()
    fledge.utils.log_time('primal solution')

    # Obtain results.

    # Flexible loads.
    state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
    control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
    output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
    for der_name in der_model_set.flexible_der_names:
        state_vector.loc[:, (der_name, slice(None))] = (
            problem.state_vector[der_name].value
        )
        control_vector.loc[:, (der_name, slice(None))] = (
            problem.control_vector[der_name].value
        )
        output_vector.loc[:, (der_name, slice(None))] = (
            problem.output_vector[der_name].value
        )

    # Thermal grid.
    der_thermal_power_vector = (
        pd.DataFrame(
            problem.der_thermal_power_vector.value,
            columns=linear_thermal_grid_model.thermal_grid_model.ders,
            index=timesteps
        )
    )
    source_thermal_power = (
        pd.DataFrame(
            problem.source_thermal_power.value,
            columns=['total'],
            index=timesteps
        )
    )

    # Electric grid.
    der_active_power_vector = (
        pd.DataFrame(
            problem.der_active_power_vector.value,
            columns=linear_electric_grid_model.electric_grid_model.ders,
            index=timesteps
        )
    )
    der_reactive_power_vector = (
        pd.DataFrame(
            problem.der_reactive_power_vector.value,
            columns=linear_electric_grid_model.electric_grid_model.ders,
            index=timesteps
        )
    )
    source_active_power = (
        pd.DataFrame(
            problem.source_active_power.value,
            columns=['total'],
            index=timesteps
        )
    )
    source_reactive_power = (
        pd.DataFrame(
            problem.source_reactive_power.value,
            columns=['total'],
            index=timesteps
        )
    )
    
    # Flexible loads: Power equations.
    lambda_thermal_power_equation = (
        pd.DataFrame(
            problem.lambda_thermal_power_equation.value,
            index=timesteps,
            columns=thermal_grid_model.ders
        )
    )
    lambda_active_power_equation = (
        pd.DataFrame(
            problem.lambda_active_power_equation.value,
            index=timesteps,
            columns=electric_grid_model.ders
        )
    )
    lambda_reactive_power_equation = (
        pd.DataFrame(
            problem.lambda_reactive_power_equation.value,
            index=timesteps,
            columns=electric_grid_model.ders
        )
    )

    # Thermal grid.
    mu_node_head_minium = (
        pd.DataFrame(
            problem.mu_node_head_minium.value,
            index=timesteps,
            columns=thermal_grid_model.nodes
        )
    )
    mu_branch_flow_maximum = (
        pd.DataFrame(
            problem.mu_branch_flow_maximum.value,
            index=timesteps,
            columns=thermal_grid_model.branches
        )
    )
    lambda_pump_power_equation = (
        pd.DataFrame(
            problem.lambda_pump_power_equation.value,
            index=timesteps,
            columns=['total']
        )
    )

    # Electric grid.
    mu_node_voltage_magnitude_minimum = (
        pd.DataFrame(
            problem.mu_node_voltage_magnitude_minimum.value,
            index=timesteps,
            columns=electric_grid_model.nodes
        )
    )
    mu_node_voltage_magnitude_maximum = (
        pd.DataFrame(
            problem.mu_node_voltage_magnitude_maximum.value,
            index=timesteps,
            columns=electric_grid_model.nodes
        )
    )
    mu_branch_power_magnitude_maximum_1 = (
        pd.DataFrame(
            problem.mu_branch_power_magnitude_maximum_1.value,
            index=timesteps,
            columns=electric_grid_model.branches
        )
    )
    mu_branch_power_magnitude_maximum_2 = (
        pd.DataFrame(
            problem.mu_branch_power_magnitude_maximum_2.value,
            index=timesteps,
            columns=electric_grid_model.branches
        )
    )
    lambda_loss_active_equation = (
        pd.DataFrame(
            problem.lambda_loss_active_equation.value,
            index=timesteps,
            columns=['total']
        )
    )
    lambda_loss_reactive_equation = (
        pd.DataFrame(
            problem.lambda_loss_reactive_equation.value,
            index=timesteps,
            columns=['total']
        )
    )

    # Store results.
    state_vector.to_csv(os.path.join(results_path, 'state_vector.csv'))
    control_vector.to_csv(os.path.join(results_path, 'control_vector.csv'))
    output_vector.to_csv(os.path.join(results_path, 'output_vector.csv'))
    der_thermal_power_vector.to_csv(os.path.join(results_path, 'der_thermal_power_vector.csv'))
    source_thermal_power.to_csv(os.path.join(results_path, 'source_thermal_power.csv'))
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    source_active_power.to_csv(os.path.join(results_path, 'source_active_power.csv'))
    source_reactive_power.to_csv(os.path.join(results_path, 'source_reactive_power.csv'))
    lambda_thermal_power_equation.to_csv(os.path.join(results_path, 'lambda_thermal_power_equation.csv'))
    lambda_active_power_equation.to_csv(os.path.join(results_path, 'lambda_active_power_equation.csv'))
    lambda_reactive_power_equation.to_csv(os.path.join(results_path, 'lambda_reactive_power_equation.csv'))
    mu_node_head_minium.to_csv(os.path.join(results_path, 'mu_node_head_minium.csv'))
    mu_branch_flow_maximum.to_csv(os.path.join(results_path, 'mu_branch_flow_maximum.csv'))
    lambda_pump_power_equation.to_csv(os.path.join(results_path, 'lambda_pump_power_equation.csv'))
    mu_node_voltage_magnitude_minimum.to_csv(os.path.join(results_path, 'mu_node_voltage_magnitude_minimum.csv'))
    mu_node_voltage_magnitude_maximum.to_csv(os.path.join(results_path, 'mu_node_voltage_magnitude_maximum.csv'))
    mu_branch_power_magnitude_maximum_1.to_csv(os.path.join(results_path, 'mu_branch_power_magnitude_maximum_1.csv'))
    mu_branch_power_magnitude_maximum_2.to_csv(os.path.join(results_path, 'mu_branch_power_magnitude_maximum_2.csv'))
    lambda_loss_active_equation.to_csv(os.path.join(results_path, 'lambda_loss_active_equation.csv'))
    lambda_loss_reactive_equation.to_csv(os.path.join(results_path, 'lambda_loss_reactive_equation.csv'))

    # Print objective.
    objective = pd.Series(problem.objective.value, index=['objective'])
    objective.to_csv(os.path.join(results_path, 'objective.csv'))
    print(f"objective = {objective.values}")

    # Store price timeseries for reference.
    price_data.price_timeseries.loc[
        :, [('active_power', 'source', 'source')]
    ].to_csv(os.path.join(results_path, 'price_timeseries.csv'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
