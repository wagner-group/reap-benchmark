"""Util functions for REAP benchmark."""

import ast

import pandas as pd


def load_annotation_df(tgt_csv_filepath: str) -> pd.DataFrame:
    """Load CSV annotation (transforms and sign class) into pd.DataFrame."""
    dataframe: pd.DataFrame = pd.read_csv(tgt_csv_filepath)
    # Converts 'tgt_final' from string to list format
    dataframe["tgt_points"] = dataframe["tgt_points"].apply(ast.literal_eval)
    # dataframe["relight_coeffs"] = dataframe["relight_coeffs"].apply(
    #     ast.literal_eval
    # )
    # Exclude shapes to which we do not apply the transform to
    dataframe = dataframe[dataframe["final_shape"] != "other-0.0-0.0"]
    return dataframe
