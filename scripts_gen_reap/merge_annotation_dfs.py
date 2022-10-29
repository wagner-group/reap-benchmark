import argparse
import json
import os
from ast import literal_eval

import numpy as np
import pandas as pd
from tqdm import tqdm


CLASS_LIST = [
    'circle-750.0',
    'triangle-900.0',
    'triangle_inverted-1220.0',
    'diamond-600.0',
    'diamond-915.0',
    'square-600.0',
    'rect-458.0-610.0',
    'rect-762.0-915.0',
    'rect-915.0-1220.0',
    'pentagon-915.0',
    'octagon-915.0',
    'other-0.0-0.0',
]


def main(args):
    split = args.split
    if split not in ['training', 'validation']:
        raise Exception('Please enter a valid split')
    
    # getting points from manual labeling (instance segmentation) and storing in a dataframe
    manual_inst_seg_anno_df = pd.DataFrame()
    json_files = []

    # for edit_path in ['traffic_signs_wrong_transform', 'traffic_signs_todo', 'traffic_signs_final_check', 'traffic_signs_use_polygon', 'traffic_signs_wrong_octagons']:
    for edit_path in ['traffic_signs_wrong_transform', 'traffic_signs_todo', 'traffic_signs_final_check', 'traffic_signs_use_polygon', 'traffic_signs_wrong_octagons', 'traffic_signs_patch_errors']:
        for group in ['1', '2', '3']:
            try:
                path_to_json = f'/data/shared/mapillary_vistas/{split}/hand_annotated_signs/{edit_path}/{group}/'
                curr_json_files = [path_to_json + p for p in os.listdir(path_to_json) if p.endswith('.json')]
                json_files.extend(curr_json_files)
                print(path_to_json)
            except:
                pass
    df_filenames = []
    df_points = []
    for json_path in json_files:
        filename = json_path.split('/')[-1].split('.json')[0] + '.png'
        df_filenames.append(filename)
        with open(json_path) as f:
            json_data = json.load(f)

        assert len(json_data['shapes']) == 1
        for annotation in json_data['shapes']:
            df_points.append(annotation['points'])

    manual_inst_seg_anno_df['filename'] = df_filenames
    manual_inst_seg_anno_df['points'] = df_points

    # loading df with manual class annotations
    if split == 'training':
        manual_annotated_df = pd.read_csv('/data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_train.csv')
    elif split == 'validation':
        manual_annotated_df = pd.read_csv('/data/shared/mtsd_v2_fully_annotated/traffic_sign_annotation_validation.csv')
    
    # merging with instance labeling df
    manual_annotated_df = manual_annotated_df.merge(
        manual_inst_seg_anno_df, left_on='filename', right_on='filename', how='left')

    # relabeling shapes in df
    final_shapes = []
    for index, row in manual_annotated_df.iterrows():
        # if reannotated then choose our annotation
        if not np.isnan(row['new_class']):
            final_shapes.append(CLASS_LIST[int(row['new_class'])])
        # if group 1 then choose agreed shape
        elif row['group'] == 1:
            final_shapes.append(row['predicted_class'])
        elif row['group'] == 2:
            # resnet mostly correct on group 2
            final_shapes.append(row['predicted_class'])
        elif row['group'] == 3:
            # use annotation: if no annotation, we do not know the traffic sign and it is actually other
            final_shapes.append('other-0.0-0.0')

        # if shape is rect and use_rect flag is 1 then we use use regular src, tgt
        # if shape is NOT rect and use_rect flag is 1 then we

    manual_annotated_df['final_shape'] = final_shapes
    # if len(manual_annotated_df) != 28404:
    #     qqq
    # read df with tgt, alpha, beta and merge
    df = pd.read_csv(f'mapillaryvistas_{split}_data.csv')
    # manual_annotated_df['filename_x'] = manual_annotated_df['filename'].apply(
        # lambda x: '_'.join(x.split('.png')[0].split('_')[: -1]) + '.jpg')
    # manual_annotated_df = manual_annotated_df.merge(df, on=['filename_x', 'object_id'], how='left')
    manual_annotated_df['filename'] = manual_annotated_df['filename'].apply(lambda x: '_'.join(x.split('_')[:-1]) + '.jpg')
    manual_annotated_df = manual_annotated_df.merge(df, on=['filename', 'object_id'], how='left')
    
    # if len(manual_annotated_df) != 28404:
    #     qqq
    final_df = manual_annotated_df[[
        'filename', 'object_id', 'shape_x', 'predicted_shape_x', 'predicted_class_x',
        'group_x', 'batch_number_x', 'row_x', 'column_x', 'new_class', 'todo',
        'use_rect_for_contour', 'wrong_transform', 'use_polygon', 'occlusion',
        'final_shape', 'points', 'tgt', 'alpha', 'beta', 'filename_png', 'xmin', 'ymin',
        'xmin_ratio', 'ymin_ratio'
    ]]

    final_df = final_df[final_df['occlusion'].isna()]
    tgt_final_values = []

    # TODO: remove. only used for debugging
    errors = []
    indices = []

    for index, row in tqdm(final_df.iterrows()):
        shape = row['final_shape'].split('-')[0]
        try:
            # curr_tgt = literal_eval(row['tgt_final'])
            curr_tgt = literal_eval(row['tgt'])
            curr_tgt = np.array(curr_tgt)
        except:
            pass

        if not isinstance(row['points'], float):
            tgt_final_values.append(row['points'])
            continue

        # offset_x_ratio = row['xmin_ratio']
        # offset_y_ratio = row['ymin_ratio']
        # h0, w0 = row['h0'], row['w0']
        # h_ratio, w_ratio = row['h_ratio'], row['w_ratio']
        # h_pad, w_pad = row['h_pad'], row['w_pad']

        # # Have to correct for the padding when df is saved (TODO: this should be simplified)
        # pad_size = int(max(h0, w0) * 0.25)
        # x_min = offset_x_ratio * (w0 + pad_size * 2) - pad_size
        # y_min = offset_y_ratio * (h0 + pad_size * 2) - pad_size

        # # Order of coordinate in tgt is inverted, i.e., (x, y) instead of (y, x)
        # curr_tgt[:, 1] = (curr_tgt[:, 1] + y_min) * h_ratio + h_pad
        # curr_tgt[:, 0] = (curr_tgt[:, 0] + x_min) * w_ratio + w_pad

        tgt_final_values.append(curr_tgt.tolist())

        if shape in ['triangle', 'triangle_inverted']:
            if len(curr_tgt) != 3:
                errors.append(row)
                indices.append(index)

        elif shape in ['square', 'diamond', 'octagon', 'circle', 'pentagon', 'rect']:
            if len(curr_tgt) != 4:
                errors.append(row)
                indices.append(index)

    print('[INFO] num errors', len(errors))

    error_df = final_df.loc[indices]

    error_df['final_check'] = 1
    error_df['group'] = 1
    final_df['tgt_final'] = tgt_final_values

    final_df = final_df.rename(columns={
        'shape_x': 'shape', 'predicted_shape_x': 'predicted_shape',
        'predicted_class_x': 'predicted_class', 'batch_number_x': 'batch_number',
        'row_x': 'row', 'column_x': 'column', 'filename_x': 'filename'
    })
    missed_alpha_beta_df = pd.read_csv(f'mapillary_vistas_{split}_alpha_beta.csv')
    alpha_list = []
    beta_list = []
    for index, row in tqdm(final_df.iterrows()):
        filename_and_obj_id = row['filename'].split('.jpg')[0] + '_' + str(row['object_id']) + '.png'
        if filename_and_obj_id in missed_alpha_beta_df['filename'].values:
            idx = missed_alpha_beta_df['filename'] == filename_and_obj_id
            alpha_list.append(missed_alpha_beta_df[idx]['alpha'].item())
            beta_list.append(missed_alpha_beta_df[idx]['beta'].item())
        else:
            alpha_list.append(row['alpha'])
            beta_list.append(row['beta'])
    final_df['alpha'] = alpha_list
    final_df['beta'] = beta_list
    final_df.to_csv(f'mapillary_vistas_{split}_final_merged.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preperation', add_help=False)
    parser.add_argument('--split', default='training', type=str)
    args = parser.parse_args()
    main(args)
