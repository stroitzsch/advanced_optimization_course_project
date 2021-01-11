"""Step 1: Solving the problem in a deterministic manner."""

import cvxpy as cp
import fledge
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
import shutil


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar_modified'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_1')

    # Clear / instantiate results directory.
    try:
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)
    except PermissionError:
        pass

    # Recreate / overwrite FLEDGE database, to incorporate changes in the scenario definition.
    # fledge.data_interface.recreate_database()

    # Obtain data & models.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Define shorthands.
    timesteps = scenario_data.timesteps
    timestep_interval_hours = (timesteps[1] - timesteps[0]) / pd.Timedelta('1h')

    # STEP 1.1: SOLVE PRIMAL PROBLEM.

    # Instantiate optimization problem.
    # - Utility object for optimization problem definition with CVXPY.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define variables.

    # Flexible loads: State space vectors.
    # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
    optimization_problem.state_vector = dict.fromkeys(der_model_set.flexible_der_names)
    optimization_problem.control_vector = dict.fromkeys(der_model_set.flexible_der_names)
    optimization_problem.output_vector = dict.fromkeys(der_model_set.flexible_der_names)
    for der_name in der_model_set.flexible_der_names:
        optimization_problem.state_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].states)
            ))
        )
        optimization_problem.control_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].controls)
            ))
        )
        optimization_problem.output_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].outputs)
            ))
        )

    # Flexible loads: Power vectors.
    optimization_problem.der_thermal_power_vector = (
        cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
    )
    optimization_problem.der_active_power_vector = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )
    optimization_problem.der_reactive_power_vector = (
        cp.Variable((len(timesteps), len(electric_grid_model.ders)))
    )

    # Thermal grid.
    optimization_problem.node_head_vector = (
        cp.Variable((len(timesteps), len(thermal_grid_model.nodes)))
    )
    optimization_problem.branch_flow_vector = (
        cp.Variable((len(timesteps), len(thermal_grid_model.branches)))
    )
    optimization_problem.pump_power = (
        cp.Variable((len(timesteps), 1))#, nonneg=True)
    )

    # Electric grid.
    optimization_problem.node_voltage_magnitude_vector = (
        cp.Variable((len(timesteps), len(electric_grid_model.nodes)))
    )
    optimization_problem.branch_power_magnitude_vector_1 = (
        cp.Variable((len(timesteps), len(electric_grid_model.branches)))
    )
    optimization_problem.branch_power_magnitude_vector_2 = (
        cp.Variable((len(timesteps), len(electric_grid_model.branches)))
    )
    optimization_problem.loss_active = cp.Variable((len(timesteps), 1))
    optimization_problem.loss_reactive = cp.Variable((len(timesteps), 1))

    # Source variables.
    optimization_problem.source_thermal_power = cp.Variable((len(timesteps), 1))
    optimization_problem.source_active_power = cp.Variable((len(timesteps), 1))
    optimization_problem.source_reactive_power = cp.Variable((len(timesteps), 1))

    # Define constraints.

    # Flexible loads.
    for der_model in der_model_set.flexible_der_models.values():

        # Initial state.
        optimization_problem.constraints.append(
            optimization_problem.state_vector[der_model.der_name][0, :]
            ==
            der_model.state_vector_initial.values
        )

        # State equation.
        optimization_problem.constraints.append(
            optimization_problem.state_vector[der_model.der_name][1:, :]
            ==
            cp.transpose(
                der_model.state_matrix.values
                @ cp.transpose(optimization_problem.state_vector[der_model.der_name][:-1, :])
                + der_model.control_matrix.values
                @ cp.transpose(optimization_problem.control_vector[der_model.der_name][:-1, :])
                + der_model.disturbance_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
            )
        )

        # Output equation.
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name]
            ==
            cp.transpose(
                der_model.state_output_matrix.values
                @ cp.transpose(optimization_problem.state_vector[der_model.der_name])
                + der_model.control_output_matrix.values
                @ cp.transpose(optimization_problem.control_vector[der_model.der_name])
                + der_model.disturbance_output_matrix.values
                @ np.transpose(der_model.disturbance_timeseries.values)
            )
        )

        # Output limits.
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name]
            >=
            der_model.output_minimum_timeseries.values
        )
        optimization_problem.constraints.append(
            optimization_problem.output_vector[der_model.der_name]
            <=
            der_model.output_maximum_timeseries.values
        )

        # Power mapping.
        der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
        optimization_problem.constraints.append(
            optimization_problem.der_active_power_vector[:, der_index]
            ==
            der_model.mapping_active_power_by_output.values
            @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
        )
        optimization_problem.constraints.append(
            optimization_problem.der_reactive_power_vector[:, der_index]
            ==
            der_model.mapping_reactive_power_by_output.values
            @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
        )
        optimization_problem.constraints.append(
            optimization_problem.der_thermal_power_vector[:, der_index]
            ==
            der_model.mapping_thermal_power_by_output.values
            @ cp.transpose(optimization_problem.output_vector[der_model.der_name])
        )

    # Thermal grid.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 10.0 * thermal_power_flow_solution.branch_flow_vector

    optimization_problem.constraints.append(
        optimization_problem.node_head_vector
        ==
        cp.transpose(
            linear_thermal_grid_model.sensitivity_node_head_by_der_power
            @ cp.transpose(optimization_problem.der_thermal_power_vector)
        )
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_flow_vector
        ==
        cp.transpose(
            linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
            @ cp.transpose(optimization_problem.der_thermal_power_vector)
        )
    )
    optimization_problem.constraints.append(
        optimization_problem.pump_power
        ==
        cp.transpose(
            linear_thermal_grid_model.sensitivity_pump_power_by_der_power
            @ cp.transpose(optimization_problem.der_thermal_power_vector)
        )
    )

    # Node head.
    optimization_problem.node_head_vector_minimum_constraint = (
        optimization_problem.node_head_vector
        - np.array([node_head_vector_minimum.ravel()])
        >=
        0.0
    )
    optimization_problem.constraints.append(optimization_problem.node_head_vector_minimum_constraint)

    # Branch flow.
    optimization_problem.branch_flow_vector_minimum_constraint = (
        optimization_problem.branch_flow_vector
        + np.array([branch_flow_vector_maximum.ravel()])
        >=
        0.0
    )
    optimization_problem.constraints.append(optimization_problem.branch_flow_vector_minimum_constraint)
    optimization_problem.branch_flow_vector_maximum_constraint = (
        optimization_problem.branch_flow_vector
        - np.array([branch_flow_vector_maximum.ravel()])
        <=
        0.0
    )
    optimization_problem.constraints.append(optimization_problem.branch_flow_vector_maximum_constraint)

    # Power balance.
    optimization_problem.constraints.append(
        thermal_grid_model.cooling_plant_efficiency ** -1
        * (
            optimization_problem.source_thermal_power
            + cp.sum(-1.0 * (
                optimization_problem.der_active_power_vector
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        )
        ==
        optimization_problem.pump_power
    )

    # Electric grid.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference

    # Voltage equation.
    optimization_problem.constraints.append(
        optimization_problem.node_voltage_magnitude_vector
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
            @ cp.transpose(
                optimization_problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
            @ cp.transpose(
                optimization_problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
        + np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
    )

    # Branch flow equation.
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
            @ cp.transpose(
                optimization_problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
            @ cp.transpose(
                optimization_problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
        + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
            @ cp.transpose(
                optimization_problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
            @ cp.transpose(
                optimization_problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
        + np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
    )

    # Loss equation.
    optimization_problem.constraints.append(
        optimization_problem.loss_active
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
            @ cp.transpose(
                optimization_problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
            @ cp.transpose(
                optimization_problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
        + np.real(linear_electric_grid_model.power_flow_solution.loss)
    )
    optimization_problem.constraints.append(
        optimization_problem.loss_reactive
        ==
        cp.transpose(
            linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
            @ cp.transpose(
                optimization_problem.der_active_power_vector
                - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
            + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
            @ cp.transpose(
                optimization_problem.der_reactive_power_vector
                - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
            )
        )
        + np.imag(linear_electric_grid_model.power_flow_solution.loss)
    )

    # Power balance.
    optimization_problem.constraints.append(
        optimization_problem.source_active_power
        + cp.sum(-1.0 * (
            optimization_problem.der_active_power_vector
        ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        ==
        optimization_problem.loss_active
    )
    optimization_problem.constraints.append(
        optimization_problem.source_reactive_power
        + cp.sum(-1.0 * (
            optimization_problem.der_reactive_power_vector
        ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
        ==
        optimization_problem.loss_reactive
    )

    # Voltage limits.
    # - Add dedicated constraints variables to enable retrieving dual variables.
    optimization_problem.voltage_magnitude_vector_minimum_constraint = (
        optimization_problem.node_voltage_magnitude_vector
        - np.array([node_voltage_magnitude_vector_minimum.ravel()])
        >=
        0.0
    )
    optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_minimum_constraint)
    optimization_problem.voltage_magnitude_vector_maximum_constraint = (
        optimization_problem.node_voltage_magnitude_vector
        - np.array([node_voltage_magnitude_vector_maximum.ravel()])
        <=
        0.0
    )
    optimization_problem.constraints.append(optimization_problem.voltage_magnitude_vector_maximum_constraint)

    # Branch flow limits.
    # - Add dedicated constraints variables to enable retrieving dual variables.
    optimization_problem.branch_power_magnitude_vector_1_minimum_constraint = (
        optimization_problem.branch_power_magnitude_vector_1
        + np.array([branch_power_magnitude_vector_maximum.ravel()])
        >=
        0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1_minimum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_1_maximum_constraint = (
        optimization_problem.branch_power_magnitude_vector_1
        - np.array([branch_power_magnitude_vector_maximum.ravel()])
        <=
        0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_1_maximum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_2_minimum_constraint = (
        optimization_problem.branch_power_magnitude_vector_2
        + np.array([branch_power_magnitude_vector_maximum.ravel()])
        >=
        0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2_minimum_constraint
    )
    optimization_problem.branch_power_magnitude_vector_2_maximum_constraint = (
        optimization_problem.branch_power_magnitude_vector_2
        - np.array([branch_power_magnitude_vector_maximum.ravel()])
        <=
        0.0
    )
    optimization_problem.constraints.append(
        optimization_problem.branch_power_magnitude_vector_2_maximum_constraint
    )

    # Define objective.
    optimization_problem.objective += (
        price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        * timestep_interval_hours  # In Wh.
        @ optimization_problem.source_thermal_power
        * thermal_grid_model.cooling_plant_efficiency ** -1
    )
    optimization_problem.objective += (
        price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        * timestep_interval_hours  # In Wh.
        @ optimization_problem.source_active_power
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.

    # Flexible loads.
    state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
    control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
    output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
    for der_name in der_model_set.flexible_der_names:
        state_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.state_vector[der_name].value
        )
        control_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.control_vector[der_name].value
        )
        output_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem.output_vector[der_name].value
        )

    # Thermal grid.
    der_thermal_power_vector = (
        pd.DataFrame(
            optimization_problem.der_thermal_power_vector.value,
            columns=linear_thermal_grid_model.thermal_grid_model.ders,
            index=timesteps
        )
    )
    branch_flow_vector = (
        pd.DataFrame(
            optimization_problem.branch_flow_vector.value,
            columns=linear_thermal_grid_model.thermal_grid_model.branches,
            index=timesteps
        )
    )
    node_head_vector = (
        pd.DataFrame(
            optimization_problem.node_head_vector.value,
            columns=linear_thermal_grid_model.thermal_grid_model.nodes,
            index=timesteps
        )
    )
    pump_power = (
        pd.DataFrame(
            optimization_problem.pump_power.value,
            columns=['total'],
            index=timesteps
        )
    )

    # Electric grid.
    der_active_power_vector = (
        pd.DataFrame(
            optimization_problem.der_active_power_vector.value,
            columns=linear_electric_grid_model.electric_grid_model.ders,
            index=timesteps
        )
    )
    der_reactive_power_vector = (
        pd.DataFrame(
            optimization_problem.der_reactive_power_vector.value,
            columns=linear_electric_grid_model.electric_grid_model.ders,
            index=timesteps
        )
    )
    node_voltage_magnitude_vector = (
        pd.DataFrame(
            optimization_problem.node_voltage_magnitude_vector.value,
            columns=linear_electric_grid_model.electric_grid_model.nodes,
            index=timesteps
        )
    )
    branch_power_magnitude_vector_1 = (
        pd.DataFrame(
            optimization_problem.branch_power_magnitude_vector_1.value,
            columns=linear_electric_grid_model.electric_grid_model.branches,
            index=timesteps
        )
    )
    branch_power_magnitude_vector_2 = (
        pd.DataFrame(
            optimization_problem.branch_power_magnitude_vector_2.value,
            columns=linear_electric_grid_model.electric_grid_model.branches,
            index=timesteps
        )
    )
    loss_active = (
        pd.DataFrame(
            optimization_problem.loss_active.value,
            columns=['total'],
            index=timesteps
        )
    )
    loss_reactive = (
        pd.DataFrame(
            optimization_problem.loss_reactive.value,
            columns=['total'],
            index=timesteps
        )
    )

    # Print some results.
    print(f"der_thermal_power_vector = {der_thermal_power_vector}")
    print(f"der_active_power_vector = {der_active_power_vector}")
    print(f"der_reactive_power_vector = {der_reactive_power_vector}")

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
