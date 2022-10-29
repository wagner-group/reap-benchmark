"""Util functions for REAP benchmark."""

import ast
import pandas as pd


def load_annotation_df(tgt_csv_filepath: str) -> pd.DataFrame:
    """Load CSV annotation (transforms and sign class) into pd.DataFrame."""
    df: pd.DataFrame = pd.read_csv(tgt_csv_filepath)
    # Converts 'tgt_final' from string to list format
    df["tgt_points"] = df["tgt_points"].apply(ast.literal_eval)
    # Exclude shapes to which we do not apply the transform to
    df = df[df["final_shape"] != "other-0.0-0.0"]
    return df
