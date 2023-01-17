"""Module to plot metrics from benchmark results."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_metrics_vs_time(data: pd.DataFrame, metric: str, dataset: str, group: str):
    """Plots the metric vs. elapsed time for each model.

    Metric is plotted on the y-axis and elapsed time on the x-axis.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the results of the benchmark.
    metric : str
        The metric to plot on the y-axis, e.g. 'smape'
        There must be a column with this name in the data
    dataset : str
        Dataset for which to plot the results, e.g. 'M3'.
        Used for the title of the plot.
    group : str
        Dataset for which to plot the results,
        e.g. 'Other', 'Quarterly', 'Yearly', 'Monthly', etc.
        Used for the title of the plot.
    """
    data = data.copy()
    data.dropna(subset=["norm_time_cpu_model", metric], inplace=True)

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"'{metric}' vs. Elapsed Time"],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    data["count_ts"] = data["count_ts"].astype(str)
    data["time"] = data["time"].astype(str)

    # Add individual model metrics ----
    for i in range(len(data)):
        library = data.iloc[i]["library"]
        library_version = data.iloc[i]["library_version"]
        model = data.iloc[i]["model"]
        model_engine = data.iloc[i]["model_engine"]
        model_engine_version = data.iloc[i]["model_engine_version"]
        execution_engine = data.iloc[i]["execution_engine"]
        execution_engine_version = data.iloc[i]["execution_engine_version"]
        execution_mode = data.iloc[i]["execution_mode"]
        execution_mode_version = data.iloc[i]["execution_mode_version"]
        num_cpus = data.iloc[i]["num_cpus"]
        backup_model = data.iloc[i]["backup_model"]
        python_version = data.iloc[i]["python_version"]
        os = data.iloc[i]["os"]
        num_models = data.iloc[i]["count_ts"]
        time = data.iloc[i]["time"]
        hovertext = (
            f"<br>Library: '{library}' Version: '{library_version}'"
            f"<br>Model: '{model}' "
            f"Engine: '{model_engine}' Version: '{model_engine_version}'"
            f"<br>Execution Engine: '{execution_engine}' "
            f"Version: '{execution_engine_version}'"
            f"<br>Execution Mode: '{execution_mode}' "
            f"Version: '{execution_mode_version}'"
            f"<br>Number of CPUs: '{num_cpus}' "
            f"Total Models: '{num_models}' Total Time (mins): '{time}'"
            f"<br>Backup Model: '{backup_model}'"
            f"<br>Python Version: '{python_version}' OS: '{os}'"
        )
        fig.add_scattergl(
            x=[data.iloc[i]["norm_time_cpu_model"]],
            y=[data.iloc[i][metric]],
            mode="markers",
            marker_size=10,
            row=1,
            col=1,
            name=data.iloc[i]["name"],
            hovertext=hovertext,
        )

    with fig.batch_update():
        fig.update_xaxes(
            title_text="Elapsed Time (normalized per core and per model)", row=1, col=1
        )

        # Only on first column
        fig.update_yaxes(title_text=f"{metric}", row=1, col=1)
        template = "ggplot2"
        fig.update_layout(showlegend=True, template=template)
        fig.update_traces(marker={"size": 10})
        fig.update_layout(title=f"Dataset: '{dataset}' Group: '{group}'")

    fig.show()


def plot_metrics(data: pd.DataFrame, metric: str, dataset: str, group: str):
    """Plots the metric for all models (bar chart).

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the results of the benchmark.
    metric : str
        The metric to plot, e.g. 'smape'
        There must be a column with this name in the data
    dataset : str
        Dataset for which to plot the results, e.g. 'M3'.
        Used for the title of the plot.
    group : str
        Dataset for which to plot the results,
        e.g. 'Other', 'Quarterly', 'Yearly', 'Monthly', etc.
        Used for the title of the plot.
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"'{metric}'"],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    fig.add_trace(go.Bar(y=data["name"], x=data[metric], orientation="h"))

    with fig.batch_update():
        fig.update_xaxes(title_text=f"{metric}", row=1, col=1)
        template = "ggplot2"
        fig.update_layout(showlegend=False, template=template)
        fig.update_layout(title=f"Dataset: '{dataset}' Group: '{group}'")

    fig.show()


if __name__ == "__main__":
    plot_metrics_vs_time(
        metrics=[4.5, 3.5, 5.5, 7],
        elapsed_times=[1, 2, 10, 0.5],
        model_keys=["X-s", "a-e", "a", "g"],
        metric_name="smape",
    )
