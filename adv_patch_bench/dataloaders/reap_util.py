"""Util functions for REAP benchmark."""

import ast

import pandas as pd

from hparams import RELIGHT_METHODS


def load_annotation_df(
    tgt_csv_filepath: str, keep_others: bool = False
) -> pd.DataFrame:
    """Load CSV annotation (transforms and sign class) into pd.DataFrame."""
    dataframe: pd.DataFrame = pd.read_csv(tgt_csv_filepath)
    # Converts 'tgt_final' from string to list format
    dataframe["tgt_points"] = dataframe["tgt_points"].apply(ast.literal_eval)
    for method in RELIGHT_METHODS:
        column_name = f"{method}_coeffs"
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].apply(
                lambda x: ast.literal_eval(x) if pd.notnull(x) else x
            )

    if not keep_others:
        # Exclude shapes to which we do not apply the transform to
        dataframe = dataframe[dataframe["final_shape"] != "other-0.0-0.0"]

    return dataframe
