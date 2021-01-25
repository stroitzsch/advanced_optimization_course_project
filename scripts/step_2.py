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
    scenario_name = 'singapore_tanjongpagar_modified'
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'results', 'step_1')
    scenario_in_sample_number = 10

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
