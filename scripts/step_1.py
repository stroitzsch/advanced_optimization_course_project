"""Step 1: Solving the problem in a deterministic manner."""

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
    scenario_name = 'singapore_tanjongpagar_modified'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_1')
    run_primal = True
    run_dual = True
    run_kkt = True

    # Clear / instantiate results directory.
    try:
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)
    except PermissionError:
        pass

    # STEP 1.0: SETUP MODELS.

    # Read scenario definition into FLEDGE.
    # fledge.data_interface.recreate_database()

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

    # STEP 1.1: SOLVE PRIMAL PROBLEM.
    if run_primal or run_kkt:  # Primal constraints are also needed for KKT problem.

        # Instantiate problem.
        # - Utility object for optimization problem definition with CVXPY.
        primal_problem = fledge.utils.OptimizationProblem()

        # Define variables.

        # Flexible loads: State space vectors.
        # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
        primal_problem.state_vector = dict.fromkeys(der_model_set.flexible_der_names)
        primal_problem.control_vector = dict.fromkeys(der_model_set.flexible_der_names)
        primal_problem.output_vector = dict.fromkeys(der_model_set.flexible_der_names)
        for der_name in der_model_set.flexible_der_names:
            primal_problem.state_vector[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            primal_problem.control_vector[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].controls)
                ))
            )
            primal_problem.output_vector[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )

        # Flexible loads: Power vectors.
        primal_problem.der_thermal_power_vector = (
            cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
        )
        primal_problem.der_active_power_vector = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )
        primal_problem.der_reactive_power_vector = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )

        # Source variables.
        primal_problem.source_thermal_power = cp.Variable((len(timesteps), 1))
        primal_problem.source_active_power = cp.Variable((len(timesteps), 1))
        primal_problem.source_reactive_power = cp.Variable((len(timesteps), 1))

        # Define constraints.

        # Flexible loads.
        for der_model in der_model_set.flexible_der_models.values():

            # Initial state.
            primal_problem.constraints.append(
                primal_problem.state_vector[der_model.der_name][0, :]
                ==
                der_model.state_vector_initial.values
            )

            # State equation.
            primal_problem.constraints.append(
                primal_problem.state_vector[der_model.der_name][1:, :]
                ==
                cp.transpose(
                    der_model.state_matrix.values
                    @ cp.transpose(primal_problem.state_vector[der_model.der_name][:-1, :])
                    + der_model.control_matrix.values
                    @ cp.transpose(primal_problem.control_vector[der_model.der_name][:-1, :])
                    + der_model.disturbance_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                )
            )

            # Output equation.
            primal_problem.constraints.append(
                primal_problem.output_vector[der_model.der_name]
                ==
                cp.transpose(
                    der_model.state_output_matrix.values
                    @ cp.transpose(primal_problem.state_vector[der_model.der_name])
                    + der_model.control_output_matrix.values
                    @ cp.transpose(primal_problem.control_vector[der_model.der_name])
                    + der_model.disturbance_output_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.values)
                )
            )

            # Output limits.
            primal_problem.constraints.append(
                primal_problem.output_vector[der_model.der_name]
                >=
                der_model.output_minimum_timeseries.values
            )
            primal_problem.constraints.append(
                primal_problem.output_vector[der_model.der_name]
                <=
                der_model.output_maximum_timeseries.replace(np.inf, 1e3).values
            )

            # Power mapping.
            der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
            primal_problem.constraints.append(
                primal_problem.der_active_power_vector[:, [der_index]]
                ==
                cp.transpose(
                    np.array([der_model.mapping_active_power_by_output.values])
                    @ cp.transpose(primal_problem.output_vector[der_model.der_name])
                )
            )
            primal_problem.constraints.append(
                primal_problem.der_reactive_power_vector[:, [der_index]]
                ==
                cp.transpose(
                    np.array([der_model.mapping_reactive_power_by_output.values])
                    @ cp.transpose(primal_problem.output_vector[der_model.der_name])
                )
            )
            primal_problem.constraints.append(
                primal_problem.der_thermal_power_vector[:, [der_index]]
                ==
                cp.transpose(
                    np.array([der_model.mapping_thermal_power_by_output.values])
                    @ cp.transpose(primal_problem.output_vector[der_model.der_name])
                )
            )

        # Thermal grid.

        # Node head limit.
        primal_problem.constraints.append(
            np.array([node_head_vector_minimum.ravel()])
            <=
            cp.transpose(
                linear_thermal_grid_model.sensitivity_node_head_by_der_power
                @ cp.transpose(primal_problem.der_thermal_power_vector)
            )
        )

        # Branch flow limit.
        primal_problem.constraints.append(
            cp.transpose(
                linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                @ cp.transpose(primal_problem.der_thermal_power_vector)
            )
            <=
            np.array([branch_flow_vector_maximum.ravel()])
        )
        # primal_problem.constraints.append(
        #     - 1.0
        #     * np.array([branch_flow_vector_maximum.ravel()])
        #     <=
        #     cp.transpose(
        #         linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
        #         @ cp.transpose(primal_problem.der_thermal_power_vector)
        #     )
        # )

        # Power balance.
        primal_problem.constraints.append(
            thermal_grid_model.cooling_plant_efficiency ** -1
            * (
                primal_problem.source_thermal_power
                + cp.sum(-1.0 * (
                    primal_problem.der_thermal_power_vector
                ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            )
            ==
            cp.transpose(
                linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                @ cp.transpose(primal_problem.der_thermal_power_vector)
            )
        )

        # Electric grid.

        # Voltage limits.
        primal_problem.constraints.append(
            np.array([node_voltage_magnitude_vector_minimum.ravel()])
            <=
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )
        primal_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([node_voltage_magnitude_vector_maximum.ravel()])
        )

        # Branch flow limits.
        primal_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )
        primal_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )

        # Power balance.
        primal_problem.constraints.append(
            primal_problem.source_active_power
            + cp.sum(-1.0 * (
                primal_problem.der_active_power_vector
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            ==
            np.real(linear_electric_grid_model.power_flow_solution.loss)
            + cp.transpose(
                linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )
        primal_problem.constraints.append(
            primal_problem.source_reactive_power
            + cp.sum(-1.0 * (
                primal_problem.der_reactive_power_vector
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            ==
            np.imag(linear_electric_grid_model.power_flow_solution.loss)
            + cp.transpose(
                linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                @ cp.transpose(
                    primal_problem.der_active_power_vector
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                @ cp.transpose(
                    primal_problem.der_reactive_power_vector
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )

        # Define objective.
        primal_problem.objective += (
            price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ primal_problem.source_thermal_power
            * thermal_grid_model.cooling_plant_efficiency ** -1
        )
        primal_problem.objective += (
            price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ primal_problem.source_active_power
        )

    if run_primal:

        # Solve problem.
        fledge.utils.log_time('primal solution')
        primal_problem.solve()
        fledge.utils.log_time('primal solution')

        # Obtain results.

        # Flexible loads.
        primal_state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
        primal_control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
        primal_output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        for der_name in der_model_set.flexible_der_names:
            primal_state_vector.loc[:, (der_name, slice(None))] = (
                primal_problem.state_vector[der_name].value
            )
            primal_control_vector.loc[:, (der_name, slice(None))] = (
                primal_problem.control_vector[der_name].value
            )
            primal_output_vector.loc[:, (der_name, slice(None))] = (
                primal_problem.output_vector[der_name].value
            )

        # Thermal grid.
        primal_der_thermal_power_vector = (
            pd.DataFrame(
                primal_problem.der_thermal_power_vector.value,
                columns=linear_thermal_grid_model.thermal_grid_model.ders,
                index=timesteps
            )
        )
        primal_source_thermal_power = (
            pd.DataFrame(
                primal_problem.source_thermal_power.value,
                columns=['total'],
                index=timesteps
            )
        )

        # Electric grid.
        primal_der_active_power_vector = (
            pd.DataFrame(
                primal_problem.der_active_power_vector.value,
                columns=linear_electric_grid_model.electric_grid_model.ders,
                index=timesteps
            )
        )
        primal_der_reactive_power_vector = (
            pd.DataFrame(
                primal_problem.der_reactive_power_vector.value,
                columns=linear_electric_grid_model.electric_grid_model.ders,
                index=timesteps
            )
        )
        primal_source_active_power = (
            pd.DataFrame(
                primal_problem.source_active_power.value,
                columns=['total'],
                index=timesteps
            )
        )
        primal_source_reactive_power = (
            pd.DataFrame(
                primal_problem.source_reactive_power.value,
                columns=['total'],
                index=timesteps
            )
        )

        # Store results.
        primal_state_vector.to_csv(os.path.join(results_path, 'primal_state_vector.csv'))
        primal_control_vector.to_csv(os.path.join(results_path, 'primal_control_vector.csv'))
        primal_output_vector.to_csv(os.path.join(results_path, 'primal_output_vector.csv'))
        primal_der_thermal_power_vector.to_csv(os.path.join(results_path, 'primal_der_thermal_power_vector.csv'))
        primal_source_thermal_power.to_csv(os.path.join(results_path, 'primal_source_thermal_power.csv'))
        primal_der_active_power_vector.to_csv(os.path.join(results_path, 'primal_der_active_power_vector.csv'))
        primal_der_reactive_power_vector.to_csv(os.path.join(results_path, 'primal_der_reactive_power_vector.csv'))
        primal_source_active_power.to_csv(os.path.join(results_path, 'primal_source_active_power.csv'))
        primal_source_reactive_power.to_csv(os.path.join(results_path, 'primal_source_reactive_power.csv'))

        # Obtain variable count / dimensions.
        primal_variable_count = (
            sum(np.multiply(*primal_problem.state_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*primal_problem.control_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*primal_problem.output_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + np.multiply(*primal_problem.der_thermal_power_vector.shape)
            + np.multiply(*primal_problem.der_active_power_vector.shape)
            + np.multiply(*primal_problem.der_reactive_power_vector.shape)
            + np.multiply(*primal_problem.source_thermal_power.shape)
            + np.multiply(*primal_problem.source_active_power.shape)
            + np.multiply(*primal_problem.source_reactive_power.shape)
        )
        print(f"primal_variable_count = {primal_variable_count}")

        # Print objective.
        primal_objective = pd.Series(primal_problem.objective.value, index=['primal_objective'])
        primal_objective.to_csv(os.path.join(results_path, 'primal_objective.csv'))
        print(f"primal_objective = {primal_objective.values}")

    # STEP 1.2: SOLVE DUAL PROBLEM.
    if run_dual or run_kkt:  # Primal constraints are also needed for KKT problem.

        # Instantiate problem.
        # - Utility object for optimization problem definition with CVXPY.
        dual_problem = fledge.utils.OptimizationProblem()

        # Define variables.

        # Flexible loads: State space equations.
        # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
        dual_problem.lambda_initial_state_equation = dict.fromkeys(der_model_set.flexible_der_names)
        dual_problem.lambda_state_equation = dict.fromkeys(der_model_set.flexible_der_names)
        dual_problem.lambda_output_equation = dict.fromkeys(der_model_set.flexible_der_names)
        dual_problem.mu_output_minimum = dict.fromkeys(der_model_set.flexible_der_names)
        dual_problem.mu_output_maximum = dict.fromkeys(der_model_set.flexible_der_names)
        for der_name in der_model_set.flexible_der_names:
            dual_problem.lambda_initial_state_equation[der_name] = (
                cp.Variable((
                    1,
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            dual_problem.lambda_state_equation[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps[:-1]),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            dual_problem.lambda_output_equation[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )
            dual_problem.mu_output_minimum[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ), nonneg=True)
            )
            dual_problem.mu_output_maximum[der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ), nonneg=True)
            )

        # Flexible loads: Power equations.
        dual_problem.lambda_thermal_power_equation = (
            cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
        )
        dual_problem.lambda_active_power_equation = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )
        dual_problem.lambda_reactive_power_equation = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )

        # Thermal grid.
        dual_problem.mu_node_head_minium = (
            cp.Variable((len(timesteps), len(thermal_grid_model.nodes)), nonneg=True)
        )
        dual_problem.mu_branch_flow_maximum = (
            cp.Variable((len(timesteps), len(thermal_grid_model.branches)), nonneg=True)
        )
        dual_problem.lambda_pump_power_equation = (
            cp.Variable((len(timesteps), 1))
        )

        # Electric grid.
        dual_problem.mu_node_voltage_magnitude_minimum = (
            cp.Variable((len(timesteps), len(electric_grid_model.nodes)), nonneg=True)
        )
        dual_problem.mu_node_voltage_magnitude_maximum = (
            cp.Variable((len(timesteps), len(electric_grid_model.nodes)), nonneg=True)
        )
        dual_problem.mu_branch_power_magnitude_maximum_1 = (
            cp.Variable((len(timesteps), len(electric_grid_model.branches)), nonneg=True)
        )
        dual_problem.mu_branch_power_magnitude_maximum_2 = (
            cp.Variable((len(timesteps), len(electric_grid_model.branches)), nonneg=True)
        )
        dual_problem.lambda_loss_active_equation = cp.Variable((len(timesteps), 1))
        dual_problem.lambda_loss_reactive_equation = cp.Variable((len(timesteps), 1))

        # Define constraints.

        for der_model in der_model_set.flexible_der_models.values():

            # Differential with respect to state vector.
            dual_problem.constraints.append(
                0.0
                ==
                (
                    dual_problem.lambda_initial_state_equation[der_model.der_name]
                    - (
                        dual_problem.lambda_state_equation[der_model.der_name][:1, :]
                        @ der_model.state_matrix.values
                    )
                    - (
                        dual_problem.lambda_output_equation[der_model.der_name][:1, :]
                        @ der_model.state_output_matrix.values
                    )
                )
            )
            dual_problem.constraints.append(
                0.0
                ==
                (
                    dual_problem.lambda_state_equation[der_model.der_name][0:-1, :]
                    - (
                        dual_problem.lambda_state_equation[der_model.der_name][1:, :]
                        @ der_model.state_matrix.values
                    )
                    - (
                        dual_problem.lambda_output_equation[der_model.der_name][1:-1, :]
                        @ der_model.state_output_matrix.values
                    )
                )
            )
            dual_problem.constraints.append(
                0.0
                ==
                (
                    dual_problem.lambda_state_equation[der_model.der_name][-1:, :]
                    - (
                        dual_problem.lambda_output_equation[der_model.der_name][-1:, :]
                        @ der_model.state_output_matrix.values
                    )
                )
            )

            # Differential with respect to control vector.
            dual_problem.constraints.append(
                0.0
                ==
                (
                    - (
                        dual_problem.lambda_state_equation[der_model.der_name]
                        @ der_model.control_matrix.values
                    )
                    - (
                        dual_problem.lambda_output_equation[der_model.der_name][:-1, :]
                        @ der_model.control_output_matrix.values
                    )
                )
            )
            dual_problem.constraints.append(
                0.0
                ==
                (
                    - (
                        dual_problem.lambda_output_equation[der_model.der_name][-1:, :]
                        @ der_model.control_output_matrix.values
                    )
                )
            )

            # Differential with respect to output vector.
            der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
            dual_problem.constraints.append(
                0.0
                ==
                (
                    dual_problem.lambda_output_equation[der_model.der_name]
                    - dual_problem.mu_output_minimum[der_model.der_name]
                    + dual_problem.mu_output_maximum[der_model.der_name]
                    - (
                        dual_problem.lambda_thermal_power_equation[:, [der_index]]
                        @ np.array([der_model.mapping_thermal_power_by_output.values])
                    )
                    - (
                        dual_problem.lambda_active_power_equation[:, [der_index]]
                        @ np.array([der_model.mapping_active_power_by_output.values])
                    )
                    - (
                        dual_problem.lambda_reactive_power_equation[:, [der_index]]
                        @ np.array([der_model.mapping_reactive_power_by_output.values])
                    )
                )
            )

        # Differential with respect to thermal power vector.
        dual_problem.constraints.append(
            0.0
            ==
            (
                dual_problem.lambda_thermal_power_equation
                - (
                    dual_problem.mu_node_head_minium
                    @ linear_thermal_grid_model.sensitivity_node_head_by_der_power
                )
                + (
                    dual_problem.mu_branch_flow_maximum
                    @ linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                )
                - (
                    dual_problem.lambda_pump_power_equation
                    @ (
                        thermal_grid_model.cooling_plant_efficiency ** -1
                        * np.ones(linear_thermal_grid_model.sensitivity_pump_power_by_der_power.shape)
                        + linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                    )
                )
            )
        )

        # Differential with respect to active power vector.
        dual_problem.constraints.append(
            0.0
            ==
            (
                dual_problem.lambda_active_power_equation
                - (
                    dual_problem.mu_node_voltage_magnitude_minimum
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                )
                + (
                    dual_problem.mu_node_voltage_magnitude_maximum
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                )
                + (
                    dual_problem.mu_branch_power_magnitude_maximum_1
                    @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                )
                + (
                    dual_problem.mu_branch_power_magnitude_maximum_2
                    @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                )
                - (
                    dual_problem.lambda_loss_active_equation
                    @ (
                        np.ones(linear_electric_grid_model.sensitivity_loss_active_by_der_power_active.shape)
                        + linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                    )
                )
                - (
                    dual_problem.lambda_loss_reactive_equation
                    @ linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                )
            )
        )

        # Differential with respect to reactive power vector.
        dual_problem.constraints.append(
            0.0
            ==
            (
                dual_problem.lambda_reactive_power_equation
                - (
                    dual_problem.mu_node_voltage_magnitude_minimum
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                )
                + (
                    dual_problem.mu_node_voltage_magnitude_maximum
                    @ linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                )
                + (
                    dual_problem.mu_branch_power_magnitude_maximum_1
                    @ linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                )
                + (
                    dual_problem.mu_branch_power_magnitude_maximum_2
                    @ linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                )
                - (
                    dual_problem.lambda_loss_active_equation
                    @ linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                )
                - (
                    dual_problem.lambda_loss_reactive_equation
                    @ (
                        np.ones(linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive.shape)
                        + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                    )
                )
            )
        )

        # Differential with respect to thermal source power.
        dual_problem.constraints.append(
            0.0
            ==
            (
                np.transpose([price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values])
                + dual_problem.lambda_pump_power_equation
            )
        )

        # Differential with respect to active source power.
        dual_problem.constraints.append(
            0.0
            ==
            (
                np.transpose([price_data.price_timeseries.loc[:, ('active_power', 'source', 'source')].values])
                + dual_problem.lambda_loss_active_equation
            )
        )

        # Differential with respect to active source power.
        dual_problem.constraints.append(
            0.0
            ==
            (
                dual_problem.lambda_loss_reactive_equation
            )
        )

    if run_dual:

        # Define objective.

        # Flexible loads.
        for der_model in der_model_set.flexible_der_models.values():
            dual_problem.objective += (
                -1.0
                * cp.sum(cp.multiply(
                    dual_problem.lambda_initial_state_equation[der_model.der_name],
                    np.array([der_model.state_vector_initial.values])
                ))
            )
            dual_problem.objective += (
                -1.0
                * cp.sum(cp.multiply(
                    dual_problem.lambda_state_equation[der_model.der_name],
                    cp.transpose(
                        der_model.disturbance_matrix.values
                        @ np.transpose(der_model.disturbance_timeseries.values[:-1, :])
                    )
                ))
            )
            dual_problem.objective += (
                -1.0
                * cp.sum(cp.multiply(
                    dual_problem.lambda_output_equation[der_model.der_name],
                    cp.transpose(
                        der_model.disturbance_output_matrix.values
                        @ np.transpose(der_model.disturbance_timeseries.values)
                    )
                ))
            )
            dual_problem.objective += (
                cp.sum(cp.multiply(
                    dual_problem.mu_output_minimum[der_model.der_name],
                    der_model.output_minimum_timeseries.values
                ))
            )
            dual_problem.objective += (
                -1.0
                * cp.sum(cp.multiply(
                    dual_problem.mu_output_maximum[der_model.der_name],
                    der_model.output_maximum_timeseries.replace(np.inf, 1e3).values
                ))
            )

        # Thermal grid.
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_node_head_minium,
                (
                    np.array([node_head_vector_minimum])
                    # - node_head_vector_reference
                    # + (
                    #     linear_thermal_grid_model.sensitivity_node_head_by_der_power
                    #     @ der_thermal_power_vector_reference
                    # )
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_branch_flow_maximum,
                (
                    # - branch_flow_vector_reference
                    # + (
                    #     linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                    #     @ der_thermal_power_vector_reference
                    # )
                    - 1.0
                    * np.array([branch_flow_vector_maximum])
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.lambda_pump_power_equation,
                (
                    0.0
                    # - pump_power_reference
                    # + (
                    #     linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                    #     @ der_thermal_power_vector_reference
                    # )
                )
            ))
        )

        # Electric grid.
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_node_voltage_magnitude_minimum,
                (
                    np.array([node_voltage_magnitude_vector_minimum])
                    - np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)])
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_node_voltage_magnitude_maximum,
                (
                    np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector)])
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.array([node_voltage_magnitude_vector_maximum])
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_branch_power_magnitude_maximum_1,
                (
                    np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1)])
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.array([branch_power_magnitude_vector_maximum])
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.mu_branch_power_magnitude_maximum_2,
                (
                    np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2)])
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.transpose(
                        linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    - np.array([branch_power_magnitude_vector_maximum])
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.lambda_loss_active_equation,
                (
                    -1.0
                    * np.array([np.real(linear_electric_grid_model.power_flow_solution.loss)])
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                )
            ))
        )
        dual_problem.objective += (
            cp.sum(cp.multiply(
                dual_problem.lambda_loss_reactive_equation,
                (
                    -1.0
                    * np.array([np.imag(linear_electric_grid_model.power_flow_solution.loss)])
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
                        @ np.transpose([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                    + np.transpose(
                        linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
                        @ np.transpose([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector)])
                    )
                )
            ))
        )

        # Invert sign of objective for maximisation.
        dual_problem.objective *= -1.0

        # Solve problem.
        fledge.utils.log_time('dual solution')
        dual_problem.solve()
        fledge.utils.log_time('dual solution')

        # Obtain results.

        # Flexible loads.
        dual_lambda_initial_state_equation = pd.DataFrame(0.0, index=der_model_set.timesteps[:1], columns=der_model_set.states)
        dual_lambda_state_equation = pd.DataFrame(0.0, index=der_model_set.timesteps[:-1], columns=der_model_set.states)
        dual_lambda_output_equation = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        dual_mu_output_minimum = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        dual_mu_output_maximum = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        for der_name in der_model_set.flexible_der_names:
            dual_lambda_initial_state_equation.loc[:, (der_name, slice(None))] = (
                dual_problem.lambda_initial_state_equation[der_name].value
            )
            dual_lambda_state_equation.loc[:, (der_name, slice(None))] = (
                dual_problem.lambda_state_equation[der_name].value
            )
            dual_lambda_output_equation.loc[:, (der_name, slice(None))] = (
                dual_problem.lambda_output_equation[der_name].value
            )
            dual_mu_output_minimum.loc[:, (der_name, slice(None))] = (
                dual_problem.mu_output_minimum[der_name].value
            )
            dual_mu_output_maximum.loc[:, (der_name, slice(None))] = (
                dual_problem.mu_output_maximum[der_name].value
            )

        # Flexible loads: Power equations.
        dual_lambda_thermal_power_equation = (
            pd.DataFrame(
                dual_problem.lambda_thermal_power_equation.value,
                index=timesteps,
                columns=thermal_grid_model.ders
            )
        )
        dual_lambda_active_power_equation = (
            pd.DataFrame(
                dual_problem.lambda_active_power_equation.value,
                index=timesteps,
                columns=electric_grid_model.ders
            )
        )
        dual_lambda_reactive_power_equation = (
            pd.DataFrame(
                dual_problem.lambda_reactive_power_equation.value,
                index=timesteps,
                columns=electric_grid_model.ders
            )
        )

        # Thermal grid.
        dual_mu_node_head_minium = (
            pd.DataFrame(
                dual_problem.mu_node_head_minium.value,
                index=timesteps,
                columns=thermal_grid_model.nodes
            )
        )
        dual_mu_branch_flow_maximum = (
            pd.DataFrame(
                dual_problem.mu_branch_flow_maximum.value,
                index=timesteps,
                columns=thermal_grid_model.branches
            )
        )
        dual_lambda_pump_power_equation = (
            pd.DataFrame(
                dual_problem.lambda_pump_power_equation.value,
                index=timesteps,
                columns=['total']
            )
        )

        # Electric grid.
        dual_mu_node_voltage_magnitude_minimum = (
            pd.DataFrame(
                dual_problem.mu_node_voltage_magnitude_minimum.value,
                index=timesteps,
                columns=electric_grid_model.nodes
            )
        )
        dual_mu_node_voltage_magnitude_maximum = (
            pd.DataFrame(
                dual_problem.mu_node_voltage_magnitude_maximum.value,
                index=timesteps,
                columns=electric_grid_model.nodes
            )
        )
        dual_mu_branch_power_magnitude_maximum_1 = (
            pd.DataFrame(
                dual_problem.mu_branch_power_magnitude_maximum_1.value,
                index=timesteps,
                columns=electric_grid_model.branches
            )
        )
        dual_mu_branch_power_magnitude_maximum_2 = (
            pd.DataFrame(
                dual_problem.mu_branch_power_magnitude_maximum_2.value,
                index=timesteps,
                columns=electric_grid_model.branches
            )
        )
        dual_lambda_loss_active_equation = (
            pd.DataFrame(
                dual_problem.lambda_loss_active_equation.value,
                index=timesteps,
                columns=['total']
            )
        )
        dual_lambda_loss_reactive_equation = (
            pd.DataFrame(
                dual_problem.lambda_loss_reactive_equation.value,
                index=timesteps,
                columns=['total']
            )
        )

        # Store results.
        dual_lambda_initial_state_equation.to_csv(os.path.join(results_path, 'dual_lambda_initial_state_equation.csv'))
        dual_lambda_state_equation.to_csv(os.path.join(results_path, 'dual_lambda_state_equation.csv'))
        dual_lambda_output_equation.to_csv(os.path.join(results_path, 'dual_lambda_output_equation.csv'))
        dual_mu_output_minimum.to_csv(os.path.join(results_path, 'dual_mu_output_minimum.csv'))
        dual_mu_output_maximum.to_csv(os.path.join(results_path, 'dual_mu_output_maximum.csv'))
        dual_lambda_thermal_power_equation.to_csv(os.path.join(results_path, 'dual_lambda_thermal_power_equation.csv'))
        dual_lambda_active_power_equation.to_csv(os.path.join(results_path, 'dual_lambda_active_power_equation.csv'))
        dual_lambda_reactive_power_equation.to_csv(os.path.join(results_path, 'dual_lambda_reactive_power_equation.csv'))
        dual_mu_node_head_minium.to_csv(os.path.join(results_path, 'dual_mu_node_head_minium.csv'))
        dual_mu_branch_flow_maximum.to_csv(os.path.join(results_path, 'dual_mu_branch_flow_maximum.csv'))
        dual_lambda_pump_power_equation.to_csv(os.path.join(results_path, 'dual_lambda_pump_power_equation.csv'))
        dual_mu_node_voltage_magnitude_minimum.to_csv(os.path.join(results_path, 'dual_mu_node_voltage_magnitude_minimum.csv'))
        dual_mu_node_voltage_magnitude_maximum.to_csv(os.path.join(results_path, 'dual_mu_node_voltage_magnitude_maximum.csv'))
        dual_mu_branch_power_magnitude_maximum_1.to_csv(os.path.join(results_path, 'dual_mu_branch_power_magnitude_maximum_1.csv'))
        dual_mu_branch_power_magnitude_maximum_2.to_csv(os.path.join(results_path, 'dual_mu_branch_power_magnitude_maximum_2.csv'))
        dual_lambda_loss_active_equation.to_csv(os.path.join(results_path, 'dual_lambda_loss_active_equation.csv'))
        dual_lambda_loss_reactive_equation.to_csv(os.path.join(results_path, 'dual_lambda_loss_reactive_equation.csv'))

        # Obtain variable count / dimensions.
        dual_variable_count = (
            sum(np.multiply(*dual_problem.lambda_initial_state_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*dual_problem.lambda_state_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*dual_problem.lambda_output_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*dual_problem.mu_output_minimum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*dual_problem.mu_output_maximum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + np.multiply(*dual_problem.lambda_thermal_power_equation.shape)
            + np.multiply(*dual_problem.lambda_active_power_equation.shape)
            + np.multiply(*dual_problem.lambda_reactive_power_equation.shape)
            + np.multiply(*dual_problem.mu_node_head_minium.shape)
            + np.multiply(*dual_problem.mu_branch_flow_maximum.shape)
            + np.multiply(*dual_problem.lambda_pump_power_equation.shape)
            + np.multiply(*dual_problem.mu_node_voltage_magnitude_minimum.shape)
            + np.multiply(*dual_problem.mu_node_voltage_magnitude_maximum.shape)
            + np.multiply(*dual_problem.mu_branch_power_magnitude_maximum_1.shape)
            + np.multiply(*dual_problem.mu_branch_power_magnitude_maximum_2.shape)
            + np.multiply(*dual_problem.lambda_loss_active_equation.shape)
            + np.multiply(*dual_problem.lambda_loss_reactive_equation.shape)
        )
        print(f"dual_variable_count = {dual_variable_count}")

        # Print objective.
        dual_objective = pd.Series(-1.0 * dual_problem.objective.value, index=['dual_objective'])
        dual_objective.to_csv(os.path.join(results_path, 'dual_objective.csv'))
        print(f"dual_objective = {dual_objective.values}")

    # STEP 1.3: SOLVE KKT CONDITIONS.
    if run_kkt:

        # Instantiate problem.
        # - Utility object for optimization problem definition with CVXPY.
        kkt_problem = fledge.utils.OptimizationProblem()

        # Obtain primal and dual variables.
        # - Since primal and dual variables are part of the KKT conditions, the previous definitions are recycled.
        kkt_problem.state_vector = primal_problem.state_vector
        kkt_problem.control_vector = primal_problem.control_vector
        kkt_problem.output_vector = primal_problem.output_vector
        kkt_problem.der_thermal_power_vector = primal_problem.der_thermal_power_vector
        kkt_problem.der_active_power_vector = primal_problem.der_active_power_vector
        kkt_problem.der_reactive_power_vector = primal_problem.der_reactive_power_vector
        kkt_problem.source_thermal_power = primal_problem.source_thermal_power
        kkt_problem.source_active_power = primal_problem.source_active_power
        kkt_problem.source_reactive_power = primal_problem.source_reactive_power
        kkt_problem.lambda_initial_state_equation = dual_problem.lambda_initial_state_equation
        kkt_problem.lambda_state_equation = dual_problem.lambda_state_equation
        kkt_problem.lambda_output_equation = dual_problem.lambda_output_equation
        kkt_problem.mu_output_minimum = dual_problem.mu_output_minimum
        kkt_problem.mu_output_maximum = dual_problem.mu_output_maximum
        kkt_problem.lambda_thermal_power_equation = dual_problem.lambda_thermal_power_equation
        kkt_problem.lambda_active_power_equation = dual_problem.lambda_active_power_equation
        kkt_problem.lambda_reactive_power_equation = dual_problem.lambda_reactive_power_equation
        kkt_problem.mu_node_head_minium = dual_problem.mu_node_head_minium
        kkt_problem.mu_branch_flow_maximum = dual_problem.mu_branch_flow_maximum
        kkt_problem.lambda_pump_power_equation = dual_problem.lambda_pump_power_equation
        kkt_problem.mu_node_voltage_magnitude_minimum = dual_problem.mu_node_voltage_magnitude_minimum
        kkt_problem.mu_node_voltage_magnitude_maximum = dual_problem.mu_node_voltage_magnitude_maximum
        kkt_problem.mu_branch_power_magnitude_maximum_1 = dual_problem.mu_branch_power_magnitude_maximum_1
        kkt_problem.mu_branch_power_magnitude_maximum_2 = dual_problem.mu_branch_power_magnitude_maximum_2
        kkt_problem.lambda_loss_active_equation = dual_problem.lambda_loss_active_equation
        kkt_problem.lambda_loss_reactive_equation = dual_problem.lambda_loss_reactive_equation

        # Obtain primal and dual constraints.
        # - Since primal and dual constraints are part of the KKT conditions, the previous definitions are recycled.
        kkt_problem.constraints.extend(primal_problem.constraints)
        kkt_problem.constraints.extend(dual_problem.constraints)

        # Obtain primal and dual problem objective.
        # - For testing / debugging only, since the KKT problem does not technically have any objective.
        # kkt_problem.objective = primal_problem.objective
        # kkt_problem.objective = dual_problem.objective

        # Define complementarity binary variables.
        kkt_problem.psi_output_minimum = dict.fromkeys(der_model_set.flexible_der_names)
        kkt_problem.psi_output_maximum = dict.fromkeys(der_model_set.flexible_der_names)
        for der_name in der_model_set.flexible_der_names:
            kkt_problem.psi_output_minimum[der_name] = (
                cp.Variable(kkt_problem.mu_output_minimum[der_name].shape, boolean=True)
            )
            kkt_problem.psi_output_maximum[der_name] = (
                cp.Variable(kkt_problem.mu_output_maximum[der_name].shape, boolean=True)
            )
        kkt_problem.psi_node_head_minium = cp.Variable(kkt_problem.mu_node_head_minium.shape, boolean=True)
        kkt_problem.psi_branch_flow_maximum = cp.Variable(kkt_problem.mu_branch_flow_maximum.shape, boolean=True)
        kkt_problem.psi_node_voltage_magnitude_minimum = cp.Variable(kkt_problem.mu_node_voltage_magnitude_minimum.shape, boolean=True)
        kkt_problem.psi_node_voltage_magnitude_maximum = cp.Variable(kkt_problem.mu_node_voltage_magnitude_maximum.shape, boolean=True)
        kkt_problem.psi_branch_power_magnitude_maximum_1 = cp.Variable(kkt_problem.mu_branch_power_magnitude_maximum_1.shape, boolean=True)
        kkt_problem.psi_branch_power_magnitude_maximum_2 = cp.Variable(kkt_problem.mu_branch_power_magnitude_maximum_2.shape, boolean=True)

        # Define complementarity big M parameters.
        # - Big M values are chosen based on expected order of magnitude of constraints from primal / dual solution.
        kkt_problem.big_m_output_minimum = cp.Parameter(value=2e4)
        kkt_problem.big_m_output_maximum = cp.Parameter(value=2e4)
        kkt_problem.big_m_node_head_minium = cp.Parameter(value=1e2)
        kkt_problem.big_m_branch_flow_maximum = cp.Parameter(value=1e3)
        kkt_problem.big_m_node_voltage_magnitude_minimum = cp.Parameter(value=1e2)
        kkt_problem.big_m_node_voltage_magnitude_maximum = cp.Parameter(value=1e2)
        kkt_problem.big_m_branch_power_magnitude_maximum_1 = cp.Parameter(value=1e3)
        kkt_problem.big_m_branch_power_magnitude_maximum_2 = cp.Parameter(value=1e3)

        # Define complementarity constraints.

        # Flexible loads.
        for der_model in der_model_set.flexible_der_models.values():

            # Output limits.

            kkt_problem.constraints.append(
                -1.0
                * (
                    der_model.output_minimum_timeseries.values
                    - kkt_problem.output_vector[der_model.der_name]
                )
                <=
                kkt_problem.psi_output_minimum[der_model.der_name]
                * kkt_problem.big_m_output_minimum
            )
            kkt_problem.constraints.append(
                kkt_problem.mu_output_minimum[der_model.der_name]
                <=
                (1 - kkt_problem.psi_output_minimum[der_model.der_name])
                * kkt_problem.big_m_output_minimum
            )

            kkt_problem.constraints.append(
                -1.0
                * (
                    kkt_problem.output_vector[der_model.der_name]
                    - der_model.output_maximum_timeseries.replace(np.inf, 1e4).values
                )
                <=
                kkt_problem.psi_output_maximum[der_model.der_name]
                * kkt_problem.big_m_output_maximum
            )
            kkt_problem.constraints.append(
                kkt_problem.mu_output_maximum[der_model.der_name]
                <=
                (1 - kkt_problem.psi_output_maximum[der_model.der_name])
                * kkt_problem.big_m_output_maximum
            )

        # Thermal grid.

        # Node head limit.
        kkt_problem.constraints.append(
            -1.0
            * (
                np.array([node_head_vector_minimum.ravel()])
                - cp.transpose(
                    linear_thermal_grid_model.sensitivity_node_head_by_der_power
                    @ cp.transpose(kkt_problem.der_thermal_power_vector)
                )
            )
            <=
            kkt_problem.psi_node_head_minium
            * kkt_problem.big_m_node_head_minium
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_node_head_minium
            <=
            (1 - kkt_problem.psi_node_head_minium)
            * kkt_problem.big_m_node_head_minium
        )

        # Branch flow limit.
        kkt_problem.constraints.append(
            -1.0
            * (
                cp.transpose(
                    linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                    @ cp.transpose(kkt_problem.der_thermal_power_vector)
                )
                - np.array([branch_flow_vector_maximum.ravel()])
            )
            <=
            kkt_problem.psi_branch_flow_maximum
            * kkt_problem.big_m_branch_flow_maximum
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_branch_flow_maximum
            <=
            (1 - kkt_problem.psi_branch_flow_maximum)
            * kkt_problem.big_m_branch_flow_maximum
        )

        # Voltage limits.

        kkt_problem.constraints.append(
            -1.0
            * (
                np.array([node_voltage_magnitude_vector_minimum.ravel()])
                - np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
                - cp.transpose(
                    linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                    @ cp.transpose(
                        kkt_problem.der_active_power_vector
                        - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                    + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                    @ cp.transpose(
                        kkt_problem.der_reactive_power_vector
                        - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                )
            )
            <=
            kkt_problem.psi_node_voltage_magnitude_minimum
            * kkt_problem.big_m_node_voltage_magnitude_minimum
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_node_voltage_magnitude_minimum
            <=
            (1 - kkt_problem.psi_node_voltage_magnitude_minimum)
            * kkt_problem.big_m_node_voltage_magnitude_minimum
        )

        kkt_problem.constraints.append(
            -1.0
            * (
                np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
                + cp.transpose(
                    linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                    @ cp.transpose(
                        kkt_problem.der_active_power_vector
                        - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                    + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                    @ cp.transpose(
                        kkt_problem.der_reactive_power_vector
                        - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                )
                - np.array([node_voltage_magnitude_vector_maximum.ravel()])
            )
            <=
            kkt_problem.psi_node_voltage_magnitude_maximum
            * kkt_problem.big_m_node_voltage_magnitude_maximum
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_node_voltage_magnitude_maximum
            <=
            (1 - kkt_problem.psi_node_voltage_magnitude_maximum)
            * kkt_problem.big_m_node_voltage_magnitude_maximum
        )

        # Branch flow limits.

        kkt_problem.constraints.append(
            -1.0
            * (
                np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
                + cp.transpose(
                    linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                    @ cp.transpose(
                        kkt_problem.der_active_power_vector
                        - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                    + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                    @ cp.transpose(
                        kkt_problem.der_reactive_power_vector
                        - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                )
                - np.array([branch_power_magnitude_vector_maximum.ravel()])
            )
            <=
            kkt_problem.psi_branch_power_magnitude_maximum_1
            * kkt_problem.big_m_branch_power_magnitude_maximum_1
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_branch_power_magnitude_maximum_1
            <=
            (1 - kkt_problem.psi_branch_power_magnitude_maximum_1)
            * kkt_problem.big_m_branch_power_magnitude_maximum_1
        )

        kkt_problem.constraints.append(
            -1.0
            * (
                np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
                + cp.transpose(
                    linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                    @ cp.transpose(
                        kkt_problem.der_active_power_vector
                        - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                    + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                    @ cp.transpose(
                        kkt_problem.der_reactive_power_vector
                        - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                    )
                )
                - np.array([branch_power_magnitude_vector_maximum.ravel()])
            )
            <=
            kkt_problem.psi_branch_power_magnitude_maximum_2
            * kkt_problem.big_m_branch_power_magnitude_maximum_2
        )
        kkt_problem.constraints.append(
            kkt_problem.mu_branch_power_magnitude_maximum_2
            <=
            (1 - kkt_problem.psi_branch_power_magnitude_maximum_2)
            * kkt_problem.big_m_branch_power_magnitude_maximum_2
        )

        # Solve problem.
        fledge.utils.log_time('KKT solution')
        kkt_problem.solve()
        fledge.utils.log_time('KKT solution')

        # Obtain results.

        # Flexible loads.
        kkt_state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
        kkt_control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
        kkt_output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        kkt_lambda_initial_state_equation = pd.DataFrame(0.0, index=der_model_set.timesteps[:1], columns=der_model_set.states)
        kkt_lambda_state_equation = pd.DataFrame(0.0, index=der_model_set.timesteps[:-1], columns=der_model_set.states)
        kkt_lambda_output_equation = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        kkt_mu_output_minimum = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        kkt_mu_output_maximum = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
        for der_name in der_model_set.flexible_der_names:
            kkt_state_vector.loc[:, (der_name, slice(None))] = (
                kkt_problem.state_vector[der_name].value
            )
            kkt_control_vector.loc[:, (der_name, slice(None))] = (
                kkt_problem.control_vector[der_name].value
            )
            kkt_output_vector.loc[:, (der_name, slice(None))] = (
                kkt_problem.output_vector[der_name].value
            )
            kkt_lambda_initial_state_equation.loc[:, (der_name, slice(None))] = (
                kkt_problem.lambda_initial_state_equation[der_name].value
            )
            kkt_lambda_state_equation.loc[:, (der_name, slice(None))] = (
                kkt_problem.lambda_state_equation[der_name].value
            )
            kkt_lambda_output_equation.loc[:, (der_name, slice(None))] = (
                kkt_problem.lambda_output_equation[der_name].value
            )
            kkt_mu_output_minimum.loc[:, (der_name, slice(None))] = (
                kkt_problem.mu_output_minimum[der_name].value
            )
            kkt_mu_output_maximum.loc[:, (der_name, slice(None))] = (
                kkt_problem.mu_output_maximum[der_name].value
            )

        # Flexible loads: Power equations.
        kkt_lambda_thermal_power_equation = (
            pd.DataFrame(
                kkt_problem.lambda_thermal_power_equation.value,
                index=timesteps,
                columns=thermal_grid_model.ders
            )
        )
        kkt_lambda_active_power_equation = (
            pd.DataFrame(
                kkt_problem.lambda_active_power_equation.value,
                index=timesteps,
                columns=electric_grid_model.ders
            )
        )
        kkt_lambda_reactive_power_equation = (
            pd.DataFrame(
                kkt_problem.lambda_reactive_power_equation.value,
                index=timesteps,
                columns=electric_grid_model.ders
            )
        )

        # Thermal grid.
        kkt_der_thermal_power_vector = (
            pd.DataFrame(
                kkt_problem.der_thermal_power_vector.value,
                columns=linear_thermal_grid_model.thermal_grid_model.ders,
                index=timesteps
            )
        )
        kkt_source_thermal_power = (
            pd.DataFrame(
                kkt_problem.source_thermal_power.value,
                columns=['total'],
                index=timesteps
            )
        )
        kkt_mu_node_head_minium = (
            pd.DataFrame(
                kkt_problem.mu_node_head_minium.value,
                index=timesteps,
                columns=thermal_grid_model.nodes
            )
        )
        kkt_mu_branch_flow_maximum = (
            pd.DataFrame(
                kkt_problem.mu_branch_flow_maximum.value,
                index=timesteps,
                columns=thermal_grid_model.branches
            )
        )
        kkt_lambda_pump_power_equation = (
            pd.DataFrame(
                kkt_problem.lambda_pump_power_equation.value,
                index=timesteps,
                columns=['total']
            )
        )

        # Electric grid.
        kkt_der_active_power_vector = (
            pd.DataFrame(
                kkt_problem.der_active_power_vector.value,
                columns=linear_electric_grid_model.electric_grid_model.ders,
                index=timesteps
            )
        )
        kkt_der_reactive_power_vector = (
            pd.DataFrame(
                kkt_problem.der_reactive_power_vector.value,
                columns=linear_electric_grid_model.electric_grid_model.ders,
                index=timesteps
            )
        )
        kkt_source_active_power = (
            pd.DataFrame(
                kkt_problem.source_active_power.value,
                columns=['total'],
                index=timesteps
            )
        )
        kkt_source_reactive_power = (
            pd.DataFrame(
                kkt_problem.source_reactive_power.value,
                columns=['total'],
                index=timesteps
            )
        )
        kkt_mu_node_voltage_magnitude_minimum = (
            pd.DataFrame(
                kkt_problem.mu_node_voltage_magnitude_minimum.value,
                index=timesteps,
                columns=electric_grid_model.nodes
            )
        )
        kkt_mu_node_voltage_magnitude_maximum = (
            pd.DataFrame(
                kkt_problem.mu_node_voltage_magnitude_maximum.value,
                index=timesteps,
                columns=electric_grid_model.nodes
            )
        )
        kkt_mu_branch_power_magnitude_maximum_1 = (
            pd.DataFrame(
                kkt_problem.mu_branch_power_magnitude_maximum_1.value,
                index=timesteps,
                columns=electric_grid_model.branches
            )
        )
        kkt_mu_branch_power_magnitude_maximum_2 = (
            pd.DataFrame(
                kkt_problem.mu_branch_power_magnitude_maximum_2.value,
                index=timesteps,
                columns=electric_grid_model.branches
            )
        )
        kkt_lambda_loss_active_equation = (
            pd.DataFrame(
                kkt_problem.lambda_loss_active_equation.value,
                index=timesteps,
                columns=['total']
            )
        )
        kkt_lambda_loss_reactive_equation = (
            pd.DataFrame(
                kkt_problem.lambda_loss_reactive_equation.value,
                index=timesteps,
                columns=['total']
            )
        )

        # Store results.
        kkt_state_vector.to_csv(os.path.join(results_path, 'kkt_state_vector.csv'))
        kkt_control_vector.to_csv(os.path.join(results_path, 'kkt_control_vector.csv'))
        kkt_output_vector.to_csv(os.path.join(results_path, 'kkt_output_vector.csv'))
        kkt_der_thermal_power_vector.to_csv(os.path.join(results_path, 'kkt_der_thermal_power_vector.csv'))
        kkt_source_thermal_power.to_csv(os.path.join(results_path, 'kkt_source_thermal_power.csv'))
        kkt_der_active_power_vector.to_csv(os.path.join(results_path, 'kkt_der_active_power_vector.csv'))
        kkt_der_reactive_power_vector.to_csv(os.path.join(results_path, 'kkt_der_reactive_power_vector.csv'))
        kkt_source_active_power.to_csv(os.path.join(results_path, 'kkt_source_active_power.csv'))
        kkt_source_reactive_power.to_csv(os.path.join(results_path, 'kkt_source_reactive_power.csv'))
        kkt_lambda_initial_state_equation.to_csv(os.path.join(results_path, 'kkt_lambda_initial_state_equation.csv'))
        kkt_lambda_state_equation.to_csv(os.path.join(results_path, 'kkt_lambda_state_equation.csv'))
        kkt_lambda_output_equation.to_csv(os.path.join(results_path, 'kkt_lambda_output_equation.csv'))
        kkt_mu_output_minimum.to_csv(os.path.join(results_path, 'kkt_mu_output_minimum.csv'))
        kkt_mu_output_maximum.to_csv(os.path.join(results_path, 'kkt_mu_output_maximum.csv'))
        kkt_lambda_thermal_power_equation.to_csv(os.path.join(results_path, 'kkt_lambda_thermal_power_equation.csv'))
        kkt_lambda_active_power_equation.to_csv(os.path.join(results_path, 'kkt_lambda_active_power_equation.csv'))
        kkt_lambda_reactive_power_equation.to_csv(os.path.join(results_path, 'kkt_lambda_reactive_power_equation.csv'))
        kkt_mu_node_head_minium.to_csv(os.path.join(results_path, 'kkt_mu_node_head_minium.csv'))
        kkt_mu_branch_flow_maximum.to_csv(os.path.join(results_path, 'kkt_mu_branch_flow_maximum.csv'))
        kkt_lambda_pump_power_equation.to_csv(os.path.join(results_path, 'kkt_lambda_pump_power_equation.csv'))
        kkt_mu_node_voltage_magnitude_minimum.to_csv(os.path.join(results_path, 'kkt_mu_node_voltage_magnitude_minimum.csv'))
        kkt_mu_node_voltage_magnitude_maximum.to_csv(os.path.join(results_path, 'kkt_mu_node_voltage_magnitude_maximum.csv'))
        kkt_mu_branch_power_magnitude_maximum_1.to_csv(os.path.join(results_path, 'kkt_mu_branch_power_magnitude_maximum_1.csv'))
        kkt_mu_branch_power_magnitude_maximum_2.to_csv(os.path.join(results_path, 'kkt_mu_branch_power_magnitude_maximum_2.csv'))
        kkt_lambda_loss_active_equation.to_csv(os.path.join(results_path, 'kkt_lambda_loss_active_equation.csv'))
        kkt_lambda_loss_reactive_equation.to_csv(os.path.join(results_path, 'kkt_lambda_loss_reactive_equation.csv'))

        # Obtain variable count / dimensions.
        kkt_variable_count = (
            sum(np.multiply(*kkt_problem.state_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.control_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.output_vector[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + np.multiply(*kkt_problem.der_thermal_power_vector.shape)
            + np.multiply(*kkt_problem.der_active_power_vector.shape)
            + np.multiply(*kkt_problem.der_reactive_power_vector.shape)
            + np.multiply(*kkt_problem.source_thermal_power.shape)
            + np.multiply(*kkt_problem.source_active_power.shape)
            + np.multiply(*kkt_problem.source_reactive_power.shape)
            + sum(np.multiply(*kkt_problem.lambda_initial_state_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.lambda_state_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.lambda_output_equation[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.mu_output_minimum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.mu_output_maximum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + np.multiply(*kkt_problem.lambda_thermal_power_equation.shape)
            + np.multiply(*kkt_problem.lambda_active_power_equation.shape)
            + np.multiply(*kkt_problem.lambda_reactive_power_equation.shape)
            + np.multiply(*kkt_problem.mu_node_head_minium.shape)
            + np.multiply(*kkt_problem.mu_branch_flow_maximum.shape)
            + np.multiply(*kkt_problem.lambda_pump_power_equation.shape)
            + np.multiply(*kkt_problem.mu_node_voltage_magnitude_minimum.shape)
            + np.multiply(*kkt_problem.mu_node_voltage_magnitude_maximum.shape)
            + np.multiply(*kkt_problem.mu_branch_power_magnitude_maximum_1.shape)
            + np.multiply(*kkt_problem.mu_branch_power_magnitude_maximum_2.shape)
            + np.multiply(*kkt_problem.lambda_loss_active_equation.shape)
            + np.multiply(*kkt_problem.lambda_loss_reactive_equation.shape)
            + sum(np.multiply(*kkt_problem.psi_output_minimum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + sum(np.multiply(*kkt_problem.psi_output_maximum[der_name].shape) for der_name in der_model_set.flexible_der_names)
            + np.multiply(*kkt_problem.psi_node_head_minium.shape)
            + np.multiply(*kkt_problem.psi_branch_flow_maximum.shape)
            + np.multiply(*kkt_problem.psi_node_voltage_magnitude_minimum.shape)
            + np.multiply(*kkt_problem.psi_node_voltage_magnitude_maximum.shape)
            + np.multiply(*kkt_problem.psi_branch_power_magnitude_maximum_1.shape)
            + np.multiply(*kkt_problem.psi_branch_power_magnitude_maximum_2.shape)
        )
        print(f"kkt_variable_count = {kkt_variable_count}")

        # Print objective.
        # - The primal objective is evaluated based on the KKT solution,
        #   because the KKT problem itself does not have an objective.
        kkt_objective = pd.Series(primal_problem.objective.value, index=['kkt_objective'])
        kkt_objective.to_csv(os.path.join(results_path, 'kkt_objective.csv'))
        print(f"kkt_objective = {kkt_objective.values}")

    # Store price timeseries for reference.
    price_data.price_timeseries.loc[
        :, [('active_power', 'source', 'source')]
    ].to_csv(os.path.join(results_path, 'price_timeseries.csv'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
