"""Module to plot metrics from benchmark results."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator


def _return_color_map(models: list) -> dict:
    """Returns a color map for the models.

    Parameters
    ----------
    models : list
        List of models for which to return the color map.

    Returns
    -------
    dict
        Dictionary with model as key and color as value.
    """
    # Add color map for models ----
    color_map = {}
    cmap = px.colors.sequential.Turbo  # px.colors.sequential.Viridis
    total_colors = len(cmap)
    for i, model in enumerate(models):
        color_map[model] = cmap[i % total_colors]
    return color_map


def _return_symbol_map(models: list) -> dict:
    """Returns a symbol map for the models.

    Parameters
    ----------
    models : list
        List of models for which to return the symbol map.

    Returns
    -------
    dict
        Dictionary with model as key and symbol as value.
    """
    # Ref: https://plotly.com/python/marker-style/
    symbols = []
    name_stems = []
    # name_variants = []
    raw_symbols = SymbolValidator().values
    for i in range(0, len(raw_symbols), 3):
        name = raw_symbols[i + 2]
        symbols.append(raw_symbols[i])
        name_stems.append(name.replace("-open", "").replace("-dot", ""))
        # name_variants.append(name[len(name_stems[-1]) :])

    if len(models) <= len(set(name_stems)):
        symbols_to_use = list(set(name_stems))
    else:
        symbols_to_use = symbols
    symbols_to_use.sort()

    symbol_map = {}
    total_symbols = len(symbols_to_use)
    for i, model in enumerate(models):
        symbol_map[model] = symbols_to_use[i % total_symbols]

    return symbol_map


def plot_metrics_vs_time(
    data: pd.DataFrame, metric: str, dataset: str, group: str, name_col: str = "key"
):
    """Plots the metric vs. elapsed time for each model.

    Metric is plotted on the y-axis and elapsed time on the x-axis.
    NOTE: Time passed should be in seconds.

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
    name_col : str, optional
        The column name to use to name the data points, by default "key"
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

    models = data["model"].unique()
    color_map = _return_color_map(models)
    symbol_map = _return_symbol_map(models)

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
        primary_model_per = data.iloc[i]["primary_model_per"]
        backup_model_per = data.iloc[i]["backup_model_per"]
        no_model_per = data.iloc[i]["no_model_per"]
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
            f"Total Models: '{num_models}' Total Time (seconds): '{time}'"
            f"<br>Backup Model: '{backup_model}' Backup Model %: '{backup_model_per}'"
            f"<br>Primary Model %: '{primary_model_per}' No Model %: '{no_model_per}'"
            f"<br>Python Version: '{python_version}' OS: '{os}'"
        )
        fig.add_scattergl(
            x=[data.iloc[i]["norm_time_cpu_model"]],
            y=[data.iloc[i][metric]],
            mode="markers",
            marker=dict(
                color=color_map.get(model, "ffffff"),
                symbol=symbol_map.get(model, "circle"),
            ),
            marker_line_width=2,
            marker_size=15,
            row=1,
            col=1,
            name=data.iloc[i][name_col],
            hovertext=hovertext,
        )

    with fig.batch_update():
        fig.update_xaxes(
            title_text="Elapsed Time in Seconds (normalized per core and per model)",
            row=1,
            col=1,
        )

        # Only on first column
        fig.update_yaxes(title_text=f"{metric}", row=1, col=1)
        template = "ggplot2"
        fig.update_layout(showlegend=True, template=template)
        fig.update_layout(title=f"Dataset: '{dataset}' Group: '{group}'")

    return fig


def plot_metrics(
    data: pd.DataFrame, metric: str, dataset: str, group: str, name_col: str = "key"
):
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
    name_col : str, optional
        The column name to use to name the data points, by default "key"
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[f"'{metric}'"],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    fig.add_trace(go.Bar(y=data[name_col], x=data[metric], orientation="h"))

    with fig.batch_update():
        fig.update_xaxes(title_text=f"{metric}", row=1, col=1)
        # tickmode added so that all y-axis labels are shown
        fig.update_yaxes(tickmode="linear", row=1, col=1)
        template = "ggplot2"
        fig.update_layout(showlegend=False, template=template)
        fig.update_layout(title=f"Dataset: '{dataset}' Group: '{group}'")

    return fig


if __name__ == "__main__":
    plot_metrics_vs_time(
        metrics=[4.5, 3.5, 5.5, 7],
        elapsed_times=[1, 2, 10, 0.5],
        model_keys=["X-s", "a-e", "a", "g"],
        metric_name="smape",
    )
