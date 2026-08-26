#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"  # Use the browser for rendering plots


def plot(df: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Plot the benchmark results from a JSON file."""
    fig = px.line(
        df,
        x="params.x",
        y="stats.mean",
        color="package",
        error_y=df["stats.stddev"],
        labels={
            "stats.mean": "Mean Time (s)",
            "params.x": "Number of Atoms",
        },
    )
    fig.update_layout(
        title=title,
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
        xaxis_type="log",
        yaxis_type="log",
    )
    return fig


def json2df(data: dict) -> pd.DataFrame:
    df = pd.json_normalize(data["benchmarks"])
    tmp = pd.DataFrame()
    tmp["name"] = df["name"].str.extract(r"^test_(.*?)(?:\[.*\])?$")
    df[["package", "function_name"]] = tmp["name"].str.rsplit("_", n=1, expand=True)
    return df


def main(base: str, show: bool, save: bool) -> None:
    """Main function to plot benchmark results."""
    # Recursive search for JSON files in the base path
    base_path = Path(base)
    files = list(base_path.glob("**/*.json"))
    if not files:
        print("No JSON files found in the specified path.")
        sys.exit(1)
    file_map = {}
    files.sort(key=lambda x: x.stem)
    for f in files:
        _, stem = f.stem.split("_", 1)
        file_map[stem] = f

    figs = []
    for f in file_map.values():
        with open(f, encoding="utf-8") as file:
            data = json.load(file)

        df = json2df(data)
        # Plot by function name
        for function_name, group_df in df.groupby("function_name"):
            fig = plot(group_df, title=str(function_name))
            figs.append((function_name, fig))

    for function_name, fig in figs:
        if show:
            fig.show()

        if save:
            _, file_stem = f.stem.split("_", 1)
            fig.write_image(f"{file_stem}_{function_name}.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show the plot in the browser.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the plot as SVG files instead of showing.",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".benchmarks",
        help="Base path to search for JSON files.",
    )
    args = parser.parse_args()
    main(base=args.base_path, show=args.show, save=args.save)
