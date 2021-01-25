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

    # STEP 2.0: LOAD DATA & SETUP MODELS.

    # Obtain time step index shorthands.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    timesteps = scenario_data.timesteps
    timestep_interval_hours = (timesteps[1] - timesteps[0]) / pd.Timedelta('1h')

    # Load weather data from CoBMo.
    weather_timeseries = (
        pd.read_sql(
            "SELECT time, irradiation_horizontal FROM weather_timeseries WHERE weather_type = 'singapore_iwec'",
            con=cobmo.data_interface.connect_database(),
            parse_dates=['time'],
            index_col='time'
        )
    )
    # Resample / down-sample if needed.
    weather_timeseries = (
        weather_timeseries.resample(
            pd.Timedelta(f'{timestep_interval_hours}h'),
            label='left'  # Using zero-order hold in the simulation.
        ).mean()
    )
    # Interpolate / up-sample if needed.
    weather_timeseries = (
        weather_timeseries.reindex(
            pd.date_range(
                weather_timeseries.index[0],
                weather_timeseries.index[-1],
                freq=pd.Timedelta(f'{timestep_interval_hours}h')
            )
        ).interpolate(method='linear')
    )
    # Drop last time step (first hour of next year).
    weather_timeseries = weather_timeseries.iloc[0:-1, :]
    # Pivot weather timeseries into table with column for each day of the year.
    weather_timeseries.loc[:, 'day'] = weather_timeseries.index.day
    weather_timeseries.loc[:, 'time_string'] = weather_timeseries.index.strftime('%H:%M')
    weather_timeseries = (
        weather_timeseries.pivot_table(
            index='time_string',
            columns='day',
            values='irradiation_horizontal',
            aggfunc=np.nanmean,
            fill_value=0.0
        )
    )

    # STEP 2.1: GENERATE SCENARIOS.
    pass

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
