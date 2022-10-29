import numpy as np
import pandas as pd
import ast
from adv_patch_bench.transforms import verifier

df = pd.read_csv('mapillary_vistas_final_merged.csv')
shape_df = pd.read_csv('shapes_df.csv')
df = df.merge(shape_df, on=['filename'], how='inner')

new_dataframe = pd.DataFrame()

for index, df_row in df.iterrows():

    h0 = df_row['h0']
    w0 = df_row['w0']
    final_shape = df_row['final_shape']

    if not pd.isna(df_row["points"]):
        tgt_points = np.array(ast.literal_eval(df_row["points"]), dtype=np.float32)

    else:
        tgt_points = (
            df_row["tgt"]
            if pd.isna(df_row["tgt_polygon"])
            else df_row["tgt_polygon"]
        )
        tgt_points = np.array(ast.literal_eval(tgt_points), dtype=np.float32)

        offset_x_ratio = df_row["xmin_ratio"]
        offset_y_ratio = df_row["ymin_ratio"]

        pad_size = int(max(h0, w0) * 0.25)
        x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size
        # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        tgt_points[:, 1] = (tgt_points[:, 1] + y_min) 
        tgt_points[:, 0] = (tgt_points[:, 0] + x_min) 

    shape = final_shape.split("-")[0]
    if shape != "octagon":
        tgt_points = verifier.sort_polygon_vertices(tgt_points)

    tgt_points = tgt_points.tolist()
    row = {
        'filename': df_row['filename'],
        'object_id': df_row['object_id'],
        'final_shape': final_shape,
        'alpha': df_row['alpha'],
        'beta': df_row['beta'],
        'tgt_points': tgt_points
        }
    
    new_dataframe = new_dataframe.append(row, ignore_index=True)

new_dataframe.to_csv('reap_annotations.csv', index=False)