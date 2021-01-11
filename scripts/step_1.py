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
    scenario_name = 'singapore_tanjongpagar'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_1')

    # Clear / instantiate results directory.
    if os.path.isdir(results_path):
        shutil.rmtree(results_path)
    os.mkdir(results_path)

    # Recreate / overwrite FLEDGE database, to incorporate changes in the scenario definition.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain models.
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

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define constraints.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 10.0 * thermal_power_flow_solution.branch_flow_vector
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
    )
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # Define objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
    )
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
    )
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            scenario_data.timesteps
        )
    )
    results.update(
        linear_thermal_grid_model.get_optimization_results(
            optimization_problem,
            scenario_data.timesteps
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = fledge.problems.Results()
    dlmps.update(
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )
    dlmps.update(
        linear_thermal_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs to CSV.
    dlmps.save(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
