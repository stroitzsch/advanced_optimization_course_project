"""Step 2: Solving the problem under uncertainty."""

import cvxpy as cp
import fledge
import cobmo
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shutil
import tslearn.utils
import tslearn.clustering


def main():

    # Settings.
    # scenario_name = 'singapore_tanjongpagar_modified'
    scenario_name = 'paper_2020_2_scenario_13'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_1')
    scenario_in_sample_number = 10

    # Clear / instantiate results directory.
    try:
        if os.path.isdir(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)
    except PermissionError:
        pass

    # STEP 2.0: SETUP MODELS.

    # Read scenario definition into FLEDGE.
    # fledge.data_interface.recreate_database()

    # Obtain data & models.

    # Flexible DERs.
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
    price_data_day_ahead = fledge.data_interface.PriceData(scenario_name)
    price_data_real_time = fledge.data_interface.PriceData(scenario_name)

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

    # Flexible DERs.
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
    price_data_day_ahead.price_timeseries *= -1.0 * base_power / 1e3 * timestep_interval_hours
    price_data_real_time.price_timeseries *= -2.0 * base_power / 1e3 * timestep_interval_hours

    # STEP 2.1: GENERATE SCENARIOS.

    # Load irradiation data from CoBMo.
    irradiation_timeseries = (
        pd.read_sql(
            "SELECT time, irradiation_horizontal FROM weather_timeseries WHERE weather_type = 'singapore_iwec'",
            con=cobmo.data_interface.connect_database(),
            parse_dates=['time'],
            index_col='time'
        )
    )
    # Resample / down-sample if needed.
    irradiation_timeseries = (
        irradiation_timeseries.resample(
            pd.Timedelta(f'{timestep_interval_hours}h'),
            label='left'  # Using zero-order hold in the simulation.
        ).mean()
    )
    # Interpolate / up-sample if needed.
    irradiation_timeseries = (
        irradiation_timeseries.reindex(
            pd.date_range(
                irradiation_timeseries.index[0],
                irradiation_timeseries.index[-1],
                freq=pd.Timedelta(f'{timestep_interval_hours}h')
            )
        ).interpolate(method='linear')
    )
    # Drop last time step (first hour of next year).
    irradiation_timeseries = irradiation_timeseries.iloc[0:-1, :]
    # Normalize.
    irradiation_timeseries /= irradiation_timeseries.max().max()

    # Obtain out-of-sample scenarios.
    # - Pivot irradiation timeseries into table with column for each day of the year.
    irradiation_timeseries.loc[:, 'dayofyear'] = irradiation_timeseries.index.dayofyear
    irradiation_timeseries.loc[:, 'time_string'] = irradiation_timeseries.index.strftime('%H:%M')
    irradiation_out_of_sample = (
        irradiation_timeseries.pivot_table(
            index='time_string',
            columns='dayofyear',
            values='irradiation_horizontal',
            aggfunc=np.nanmean,
            fill_value=0.0
        )
    )
    # Append time step to match length of scenario time horizon.
    irradiation_out_of_sample.loc['24:00', :] = 0.0
    # Obtain scenario index short-hand.
    out_of_sample_scenarios = irradiation_out_of_sample.columns

    # Obtain in-sample scenarios.
    # - Select representative scenarios by time series clustering.
    clustering = tslearn.clustering.TimeSeriesKMeans(n_clusters=scenario_in_sample_number)
    clustering = clustering.fit((tslearn.utils.to_time_series_dataset(irradiation_out_of_sample.transpose())))
    # irradiation_cluster_prediction = (
    #     pd.Index(
    #         clustering.predict(tslearn.utils.to_time_series_dataset(irradiation_out_of_sample.transpose()))
    #     )
    # )
    irradiation_in_sample = (
        pd.DataFrame(
            clustering.cluster_centers_[:, :, 0].transpose(),
            index=irradiation_out_of_sample.index,
            columns=range(clustering.cluster_centers_.shape[0])
        )
    )
    # Obtain scenario index short-hand.
    in_sample_scenarios = irradiation_in_sample.columns

    # STEP 2.2: SOLVE STOCHASTIC PROBLEM.

    # Instantiate problem.
    # - Utility object for optimization problem definition with CVXPY.
    in_sample_problem = fledge.utils.OptimizationProblem()

    # Define variables.
    # - Scenario dimension is added by using dicts.
    in_sample_problem.state_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.control_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.output_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.der_thermal_power_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.der_active_power_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.der_reactive_power_vector = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.source_thermal_power_real_time = dict.fromkeys(in_sample_scenarios)
    in_sample_problem.source_active_power_real_time = dict.fromkeys(in_sample_scenarios)

    for scenario in in_sample_scenarios:

        # Flexible DERs: State space vectors.
        # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
        in_sample_problem.state_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        in_sample_problem.control_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        in_sample_problem.output_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        for der_name in der_model_set.flexible_der_names:
            in_sample_problem.state_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            in_sample_problem.control_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].controls)
                ))
            )
            in_sample_problem.output_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )

        # Flexible DERs: Power vectors.
        in_sample_problem.der_thermal_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
        )
        in_sample_problem.der_active_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )
        in_sample_problem.der_reactive_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )

        # Source variables: Real time.
        in_sample_problem.source_thermal_power_real_time[scenario] = cp.Variable((len(timesteps), 1), nonpos=True)
        in_sample_problem.source_active_power_real_time[scenario] = cp.Variable((len(timesteps), 1), nonpos=True)

    # Source variables: Day ahead.
    in_sample_problem.source_thermal_power_day_ahead = cp.Variable((len(timesteps), 1), nonpos=True)
    in_sample_problem.source_active_power_day_ahead = cp.Variable((len(timesteps), 1), nonpos=True)

    # Define constraints.

    for scenario in in_sample_scenarios:

        # Flexible DERs.
        for der_model in der_model_set.flexible_der_models.values():

            # Initial state.
            in_sample_problem.constraints.append(
                in_sample_problem.state_vector[scenario][der_model.der_name][0, :]
                ==
                der_model.state_vector_initial.values
            )

            # State equation.
            in_sample_problem.constraints.append(
                in_sample_problem.state_vector[scenario][der_model.der_name][1:, :]
                ==
                cp.transpose(
                    der_model.state_matrix.values
                    @ cp.transpose(in_sample_problem.state_vector[scenario][der_model.der_name][:-1, :])
                    + der_model.control_matrix.values
                    @ cp.transpose(in_sample_problem.control_vector[scenario][der_model.der_name][:-1, :])
                    + der_model.disturbance_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                )
            )

            # Output equation.
            in_sample_problem.constraints.append(
                in_sample_problem.output_vector[scenario][der_model.der_name]
                ==
                cp.transpose(
                    der_model.state_output_matrix.values
                    @ cp.transpose(in_sample_problem.state_vector[scenario][der_model.der_name])
                    + der_model.control_output_matrix.values
                    @ cp.transpose(in_sample_problem.control_vector[scenario][der_model.der_name])
                    + der_model.disturbance_output_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.values)
                )
            )

            # Output limits.
            in_sample_problem.constraints.append(
                in_sample_problem.output_vector[scenario][der_model.der_name]
                >=
                der_model.output_minimum_timeseries.values
            )
            # For PV power plant, adjust maximum generation limit according to scenario.
            if der_model.der_type == 'flexible_generator':
                output_maximum_timeseries = (
                    pd.concat([
                        der_model.active_power_nominal * irradiation_in_sample.loc[:, scenario].rename('active_power'),
                        der_model.reactive_power_nominal * irradiation_in_sample.loc[:, scenario].rename('reactive_power')
                    ], axis='columns')
                )
                in_sample_problem.constraints.append(
                    in_sample_problem.output_vector[scenario][der_model.der_name]
                    <=
                    output_maximum_timeseries.replace(np.inf, 1e3).values
                )
            else:
                in_sample_problem.constraints.append(
                    in_sample_problem.output_vector[scenario][der_model.der_name]
                    <=
                    der_model.output_maximum_timeseries.replace(np.inf, 1e3).values
                )

            # Power mapping.
            der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
            in_sample_problem.constraints.append(
                in_sample_problem.der_active_power_vector[scenario][:, [der_index]]
                ==
                cp.transpose(
                    der_model.mapping_active_power_by_output.values
                    @ cp.transpose(in_sample_problem.output_vector[scenario][der_model.der_name])
                )
            )
            in_sample_problem.constraints.append(
                in_sample_problem.der_reactive_power_vector[scenario][:, [der_index]]
                ==
                cp.transpose(
                    der_model.mapping_reactive_power_by_output.values
                    @ cp.transpose(in_sample_problem.output_vector[scenario][der_model.der_name])
                )
            )
            # - Thermal grid power mapping only for DERs which are connected to the thermal grid.
            if der_model.der_name in thermal_grid_model.ders.get_level_values('der_name'):
                der_index = int(fledge.utils.get_index(thermal_grid_model.ders, der_name=der_model.der_name))
                in_sample_problem.constraints.append(
                    in_sample_problem.der_thermal_power_vector[scenario][:, [der_index]]
                    ==
                    cp.transpose(
                        der_model.mapping_thermal_power_by_output.values
                        @ cp.transpose(in_sample_problem.output_vector[scenario][der_model.der_name])
                    )
                )

        # Thermal grid.

        # Node head limit.
        in_sample_problem.constraints.append(
            np.array([node_head_vector_minimum.ravel()])
            <=
            cp.transpose(
                linear_thermal_grid_model.sensitivity_node_head_by_der_power
                @ cp.transpose(in_sample_problem.der_thermal_power_vector[scenario])
            )
        )

        # Branch flow limit.
        in_sample_problem.constraints.append(
            cp.transpose(
                linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                @ cp.transpose(in_sample_problem.der_thermal_power_vector[scenario])
            )
            <=
            np.array([branch_flow_vector_maximum.ravel()])
        )

        # Power balance.
        in_sample_problem.constraints.append(
            thermal_grid_model.cooling_plant_efficiency ** -1
            * (
                in_sample_problem.source_thermal_power_day_ahead
                + in_sample_problem.source_thermal_power_real_time[scenario]
                + cp.sum(-1.0 * (
                    in_sample_problem.der_thermal_power_vector[scenario]
                ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            )
            ==
            cp.transpose(
                linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                @ cp.transpose(in_sample_problem.der_thermal_power_vector[scenario])
            )
        )

        # Electric grid.

        # Voltage limits.
        in_sample_problem.constraints.append(
            np.array([node_voltage_magnitude_vector_minimum.ravel()])
            <=
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    in_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    in_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )
        in_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    in_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    in_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([node_voltage_magnitude_vector_maximum.ravel()])
        )

        # Branch flow limits.
        in_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(
                    in_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(
                    in_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )
        in_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(
                    in_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(
                    in_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )

        # Power balance.
        in_sample_problem.constraints.append(
            in_sample_problem.source_active_power_day_ahead
            + in_sample_problem.source_active_power_real_time[scenario]
            + cp.sum(-1.0 * (
                in_sample_problem.der_active_power_vector[scenario]
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            ==
            np.real(linear_electric_grid_model.power_flow_solution.loss)
            + cp.transpose(
                linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                @ cp.transpose(
                    in_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                @ cp.transpose(
                    in_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )

    # Define objective.

    # Day-ahead.
    in_sample_problem.objective += (
        price_data_day_ahead.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ in_sample_problem.source_thermal_power_day_ahead
        * thermal_grid_model.cooling_plant_efficiency ** -1
    )
    in_sample_problem.objective += (
        price_data_day_ahead.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ in_sample_problem.source_active_power_day_ahead
    )

    # Real-time.
    for scenario in in_sample_scenarios:
        in_sample_problem.objective += (
            len(in_sample_scenarios) ** -1  # Assuming equal probability.
            * price_data_real_time.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ in_sample_problem.source_thermal_power_real_time[scenario]
            * thermal_grid_model.cooling_plant_efficiency ** -1
        )
        in_sample_problem.objective += (
            len(in_sample_scenarios) ** -1  # Assuming equal probability.
            * price_data_real_time.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ in_sample_problem.source_active_power_real_time[scenario]
        )

    # Solve problem.
    fledge.utils.log_time('in sample solution')
    in_sample_problem.solve()
    fledge.utils.log_time('in sample solution')

    # Print objective.
    in_sample_objective = pd.Series(in_sample_problem.objective.value, index=['in_sample_objective'])
    in_sample_objective.to_csv(os.path.join(results_path, 'in_sample_objective.csv'))
    print(f"in_sample_objective = {in_sample_objective.values}")

    # STEP 2.2: OUT-OF-SAMPLE ANALYSIS.

    # Instantiate problem.
    # - Utility object for optimization problem definition with CVXPY.
    out_of_sample_problem = fledge.utils.OptimizationProblem()

    # Define variables.
    # - Scenario dimension is added by using dicts.
    out_of_sample_problem.state_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.control_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.output_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.der_thermal_power_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.der_active_power_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.der_reactive_power_vector = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.source_thermal_power_real_time = dict.fromkeys(out_of_sample_scenarios)
    out_of_sample_problem.source_active_power_real_time = dict.fromkeys(out_of_sample_scenarios)

    for scenario in out_of_sample_scenarios:

        # Flexible DERs: State space vectors.
        # - CVXPY only allows for 2-dimensional variables. Using dicts below to represent 3rd dimension.
        out_of_sample_problem.state_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        out_of_sample_problem.control_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        out_of_sample_problem.output_vector[scenario] = dict.fromkeys(der_model_set.flexible_der_names)
        for der_name in der_model_set.flexible_der_names:
            out_of_sample_problem.state_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].states)
                ))
            )
            out_of_sample_problem.control_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].controls)
                ))
            )
            out_of_sample_problem.output_vector[scenario][der_name] = (
                cp.Variable((
                    len(der_model_set.flexible_der_models[der_name].timesteps),
                    len(der_model_set.flexible_der_models[der_name].outputs)
                ))
            )

        # Flexible DERs: Power vectors.
        out_of_sample_problem.der_thermal_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(thermal_grid_model.ders)))
        )
        out_of_sample_problem.der_active_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )
        out_of_sample_problem.der_reactive_power_vector[scenario] = (
            cp.Variable((len(timesteps), len(electric_grid_model.ders)))
        )

        # Source variables: Real time.
        out_of_sample_problem.source_thermal_power_real_time[scenario] = cp.Variable((len(timesteps), 1), nonpos=True)
        out_of_sample_problem.source_active_power_real_time[scenario] = cp.Variable((len(timesteps), 1), nonpos=True)

    # Source variables: Day ahead.
    # - For the out-of-sample, this is a fixed parameter based on the solution of the in-sample problem.
    out_of_sample_problem.source_thermal_power_day_ahead = (
        cp.Parameter((len(timesteps), 1), value=in_sample_problem.source_thermal_power_day_ahead.value)
    )
    out_of_sample_problem.source_active_power_day_ahead = (
        cp.Parameter((len(timesteps), 1), value=in_sample_problem.source_active_power_day_ahead.value)
    )

    # Define constraints.

    for scenario in out_of_sample_scenarios:

        # Flexible DERs.
        for der_model in der_model_set.flexible_der_models.values():

            # Initial state.
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.state_vector[scenario][der_model.der_name][0, :]
                ==
                der_model.state_vector_initial.values
            )

            # State equation.
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.state_vector[scenario][der_model.der_name][1:, :]
                ==
                cp.transpose(
                    der_model.state_matrix.values
                    @ cp.transpose(out_of_sample_problem.state_vector[scenario][der_model.der_name][:-1, :])
                    + der_model.control_matrix.values
                    @ cp.transpose(out_of_sample_problem.control_vector[scenario][der_model.der_name][:-1, :])
                    + der_model.disturbance_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
                )
            )

            # Output equation.
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.output_vector[scenario][der_model.der_name]
                ==
                cp.transpose(
                    der_model.state_output_matrix.values
                    @ cp.transpose(out_of_sample_problem.state_vector[scenario][der_model.der_name])
                    + der_model.control_output_matrix.values
                    @ cp.transpose(out_of_sample_problem.control_vector[scenario][der_model.der_name])
                    + der_model.disturbance_output_matrix.values
                    @ np.transpose(der_model.disturbance_timeseries.values)
                )
            )

            # Output limits.
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.output_vector[scenario][der_model.der_name]
                >=
                der_model.output_minimum_timeseries.values
            )
            # For PV power plant, adjust maximum generation limit according to scenario.
            if der_model.der_type == 'flexible_generator':
                output_maximum_timeseries = (
                    pd.concat([
                        der_model.active_power_nominal * irradiation_out_of_sample.loc[:, scenario].rename('active_power'),
                        der_model.reactive_power_nominal * irradiation_out_of_sample.loc[:, scenario].rename('reactive_power')
                    ], axis='columns')
                )
                out_of_sample_problem.constraints.append(
                    out_of_sample_problem.output_vector[scenario][der_model.der_name]
                    <=
                    output_maximum_timeseries.replace(np.inf, 1e3).values
                )
            else:
                out_of_sample_problem.constraints.append(
                    out_of_sample_problem.output_vector[scenario][der_model.der_name]
                    <=
                    der_model.output_maximum_timeseries.replace(np.inf, 1e3).values
                )

            # Power mapping.
            der_index = int(fledge.utils.get_index(electric_grid_model.ders, der_name=der_model.der_name))
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.der_active_power_vector[scenario][:, [der_index]]
                ==
                cp.transpose(
                    der_model.mapping_active_power_by_output.values
                    @ cp.transpose(out_of_sample_problem.output_vector[scenario][der_model.der_name])
                )
            )
            out_of_sample_problem.constraints.append(
                out_of_sample_problem.der_reactive_power_vector[scenario][:, [der_index]]
                ==
                cp.transpose(
                    der_model.mapping_reactive_power_by_output.values
                    @ cp.transpose(out_of_sample_problem.output_vector[scenario][der_model.der_name])
                )
            )
            # - Thermal grid power mapping only for DERs which are connected to the thermal grid.
            if der_model.der_name in thermal_grid_model.ders.get_level_values('der_name'):
                der_index = int(fledge.utils.get_index(thermal_grid_model.ders, der_name=der_model.der_name))
                out_of_sample_problem.constraints.append(
                    out_of_sample_problem.der_thermal_power_vector[scenario][:, [der_index]]
                    ==
                    cp.transpose(
                        der_model.mapping_thermal_power_by_output.values
                        @ cp.transpose(out_of_sample_problem.output_vector[scenario][der_model.der_name])
                    )
                )

        # Thermal grid.

        # Node head limit.
        out_of_sample_problem.constraints.append(
            np.array([node_head_vector_minimum.ravel()])
            <=
            cp.transpose(
                linear_thermal_grid_model.sensitivity_node_head_by_der_power
                @ cp.transpose(out_of_sample_problem.der_thermal_power_vector[scenario])
            )
        )

        # Branch flow limit.
        out_of_sample_problem.constraints.append(
            cp.transpose(
                linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                @ cp.transpose(out_of_sample_problem.der_thermal_power_vector[scenario])
            )
            <=
            np.array([branch_flow_vector_maximum.ravel()])
        )

        # Power balance.
        out_of_sample_problem.constraints.append(
            thermal_grid_model.cooling_plant_efficiency ** -1
            * (
                out_of_sample_problem.source_thermal_power_day_ahead
                + out_of_sample_problem.source_thermal_power_real_time[scenario]
                + cp.sum(-1.0 * (
                    out_of_sample_problem.der_thermal_power_vector[scenario]
                ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            )
            ==
            cp.transpose(
                linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                @ cp.transpose(out_of_sample_problem.der_thermal_power_vector[scenario])
            )
        )

        # Electric grid.

        # Voltage limits.
        out_of_sample_problem.constraints.append(
            np.array([node_voltage_magnitude_vector_minimum.ravel()])
            <=
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    out_of_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    out_of_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )
        out_of_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.node_voltage_vector.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
                @ cp.transpose(
                    out_of_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
                @ cp.transpose(
                    out_of_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([node_voltage_magnitude_vector_maximum.ravel()])
        )

        # Branch flow limits.
        out_of_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_1.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
                @ cp.transpose(
                    out_of_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
                @ cp.transpose(
                    out_of_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )
        out_of_sample_problem.constraints.append(
            np.array([np.abs(linear_electric_grid_model.power_flow_solution.branch_power_vector_2.ravel())])
            + cp.transpose(
                linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
                @ cp.transpose(
                    out_of_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
                @ cp.transpose(
                    out_of_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
            <=
            np.array([branch_power_magnitude_vector_maximum.ravel()])
        )

        # Power balance.
        out_of_sample_problem.constraints.append(
            out_of_sample_problem.source_active_power_day_ahead
            + out_of_sample_problem.source_active_power_real_time[scenario]
            + cp.sum(-1.0 * (
                out_of_sample_problem.der_active_power_vector[scenario]
            ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
            ==
            np.real(linear_electric_grid_model.power_flow_solution.loss)
            + cp.transpose(
                linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
                @ cp.transpose(
                    out_of_sample_problem.der_active_power_vector[scenario]
                    - np.array([np.real(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
                + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
                @ cp.transpose(
                    out_of_sample_problem.der_reactive_power_vector[scenario]
                    - np.array([np.imag(linear_electric_grid_model.power_flow_solution.der_power_vector.ravel())])
                )
            )
        )

    # Define objective.

    # Day-ahead.
    out_of_sample_problem.objective += (
        price_data_day_ahead.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ out_of_sample_problem.source_thermal_power_day_ahead
        * thermal_grid_model.cooling_plant_efficiency ** -1
    )
    out_of_sample_problem.objective += (
        price_data_day_ahead.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
        @ out_of_sample_problem.source_active_power_day_ahead
    )

    # Real-time.
    for scenario in out_of_sample_scenarios:
        out_of_sample_problem.objective += (
            len(out_of_sample_scenarios) ** -1  # Assuming equal probability.
            * price_data_real_time.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ out_of_sample_problem.source_thermal_power_real_time[scenario]
            * thermal_grid_model.cooling_plant_efficiency ** -1
        )
        out_of_sample_problem.objective += (
            len(out_of_sample_scenarios) ** -1  # Assuming equal probability.
            * price_data_real_time.price_timeseries.loc[:, ('active_power', 'source', 'source')].values.T
            @ out_of_sample_problem.source_active_power_real_time[scenario]
        )

    # Solve problem.
    fledge.utils.log_time('in sample solution')
    out_of_sample_problem.solve()
    fledge.utils.log_time('in sample solution')

    # Print objective.
    out_of_sample_objective = pd.Series(out_of_sample_problem.objective.value, index=['out_of_sample_objective'])
    out_of_sample_objective.to_csv(os.path.join(results_path, 'out_of_sample_objective.csv'))
    print(f"out_of_sample_objective = {out_of_sample_objective.values}")

    # Plot selected results.

    # Plot irradiation timeseries.
    figure = go.Figure()
    for column in irradiation_out_of_sample.columns:
        figure.add_scatter(
            x=irradiation_out_of_sample.index,
            y=irradiation_out_of_sample.loc[:, column].values,
            name=column
        )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'irradiation_out_of_sample'))

    # Plot irradiation timeseries clusters.
    figure = go.Figure()
    for column in irradiation_in_sample.columns:
        figure.add_scatter(
            x=irradiation_in_sample.index,
            y=irradiation_in_sample.loc[:, column].values,
            name=column
        )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, 'irradiation_in_sample'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
