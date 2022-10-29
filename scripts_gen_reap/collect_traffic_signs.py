import argparse
import csv
import json
from os import listdir, makedirs
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm.auto import tqdm

from adv_patch_bench.utils import get_box, pad_image

DATASET = 'mapillaryvistas'
# DATASET = 'bdd100k'

if DATASET == 'mapillaryvistas':
    TRAFFIC_SIGN_LABEL = 95
elif DATASET == 'bdd100k':
    TRAFFIC_SIGN_LABEL = 'traffic sign'

CLASS_LIST = [
    'circle-750.0',
    'triangle-900.0',
    'octagon-915.0',
    'other-0.0-0.0',
    'triangle_inverted-1220.0',
    'diamond-600.0',
    'diamond-915.0',
    'square-600.0',
    'rect-458.0-610.0',
    'rect-762.0-915.0',
    'rect-915.0-1220.0',
    'pentagon-915.0'
]

SHAPE_LIST = [
    'circle',
    'triangle',
    'triangle_inverted',
    'diamond',
    'square',
    'rect',
    'pentagon',
    'octagon',
    'other'
]

# CLASS_LIST = ['octagon-915.0-915.0',
#               'diamond-915.0-915.0',
#               'pentagon-915.0-915.0',
#               'rect-915.0-1220.0',
#               'rect-762.0-915.0',
#               'triangle-900.0',
#               'circle-750.0',
#               'triangle_inverted-1220.0-1220.0',
#               'rect-458.0-610.0',
#               'other-0.0-0.0']


def crop_traffic_signs(filename, panoptic_per_image_id, img_path, label_path,
                       min_area=0, pad=0.1):

    # Load Mapillary Vistas image
    img_id = filename.split('.')[0]
    segment = panoptic_per_image_id[img_id]['segments_info']
    panoptic = np.array(Image.open(join(label_path, f'{img_id}.png')))

    img_pil = Image.open(join(img_path, filename))
    img = np.array(img_pil)[:, :, :3]
    img_height, img_width, _ = img.shape

    # Pad image to avoid cutting varying shapes due to boundary
    img_padded, pad_size = pad_image(img, pad_mode='constant', return_pad_size=True, pad_size=0.25)
    id_padded = pad_image(panoptic[:, :, 0], pad_mode='constant', pad_size=0.25)
    outputs = {
        'images': [],
        'masks': [],
        'bbox': [],
        'obj_id': [],
        'offset_x': [],
        'offset_y': [],
        'offset_x_ratio': [],
        'offset_y_ratio': []
    }
    # Crop the specified object
    for obj in segment:
        # Check if bounding box is cut off at the image boundary
        xmin, ymin, width, height = obj['bbox']
        is_oob = (xmin == 0) or (ymin == 0) or \
            ((xmin + width) >= img_width - 1) or ((ymin + height) >= img_height - 1)
        if obj['category_id'] != TRAFFIC_SIGN_LABEL or is_oob:
            continue

        # Collect mask
        extra_pad = int(max(width, height) * 0.2)
        ymin, xmin = max(0, ymin + pad_size - extra_pad), max(0, xmin + pad_size - extra_pad)
        temp_mask = (id_padded[ymin:ymin + height + 2 * extra_pad,
                               xmin:xmin + width + 2 * extra_pad] == obj['id']).astype(np.uint8)
        if temp_mask.sum() < min_area:
            continue

        # Get refined crop patch
        ymin_, ymax_, xmin_, xmax_ = get_box(temp_mask, pad)

        ymin, ymax, xmin, xmax = ymin + ymin_, ymin + ymax_, xmin + xmin_, xmin + xmax_
        bool_mask = (id_padded[ymin:ymax, xmin:xmax] == obj['id']).astype(np.uint8)
        height, width = bool_mask.shape
        if height != width:
            print('height != width')

        # assert height == width
        image = img_padded[ymin:ymax, xmin:xmax].astype(np.uint8)

        outputs['images'].append(image)
        outputs['masks'].append(bool_mask)
        outputs['obj_id'].append(obj['id'])
        outputs['bbox'].append((ymin, ymax, xmin, xmax))

        outputs['offset_x'].append(xmin)
        outputs['offset_y'].append(ymin)

        assert xmin/img_padded.shape[1] <= 1
        assert ymin/img_padded.shape[0] <= 1
        outputs['offset_x_ratio'].append(xmin/img_padded.shape[1])
        outputs['offset_y_ratio'].append(ymin/img_padded.shape[0])

        # FIXME
        # if DATASET == 'bdd100k':
        #     xmin_, ymin_, width_, height_ = obj['bbox']
        #     bool_mask[:max(0, ymin_-10), :] = 0
        #     bool_mask[min(ymin_+height_+10, img_height):, :] = 0
        #     bool_mask[:, :max(0, xmin_-10)] = 0
        #     bool_mask[:, min(xmin_+width_+10, img_width):] = 0

    return outputs


def main():

    # Arguments
    min_area = 1600

    if DATASET == 'mapillaryvistas':
        if SPLIT == 'training':
            data_dir = '/data/shared/mapillary_vistas/training/'
        elif SPLIT == 'validation':
            data_dir = '/data/shared/mapillary_vistas/validation/'
    elif DATASET == 'bdd100k':
        data_dir = '/data/shared/bdd100k/images/10k/train/'
    else:
        raise NotImplementedError(f'{DATASET} dataset is not recognized')

    # data_dir = '/data/shared/mtsd_v2_fully_annotated/'
    # model_path = '/home/nab_126/adv-patch-bench/model_weights/resnet18_cropped_signs_good_resolution_and_not_edge_10_labels.pth'

    cudnn.benchmark = True

    # Read in panoptic file
    if DATASET == 'mapillaryvistas':
        panoptic_json_path = f'{data_dir}/v2.0/panoptic/panoptic_2020.json'
    elif DATASET == 'bdd100k':
        panoptic_json_path = '/data/shared/bdd100k/labels/pan_seg/polygons/pan_seg_train.json'

    with open(panoptic_json_path) as panoptic_file:
        panoptic = json.load(panoptic_file)

    if DATASET == 'mapillaryvistas':
        panoptic_per_image_id = {}
        for annotation in panoptic['annotations']:
            # print(annotation)
            # print()
            # print()
            # print(annotation.keys())
            # qq
            panoptic_per_image_id[annotation['image_id']] = annotation

        # Convert category infos to category_id indexed dictionary
        panoptic_category_per_id = {}
        for category in panoptic['categories']:
            panoptic_category_per_id[category['id']] = category

    elif DATASET == 'bdd100k':
        # creating same mapping for bdd100k
        panoptic_per_image_id = {}
        for image_annotation in tqdm(panoptic):
            filename = image_annotation['name']
            image_id = filename.split('.jpg')[0]
            annotation = {}
            annotation['filename'] = filename
            annotation['image_id'] = image_id
            segments_info = []
            for label in image_annotation['labels']:
                label_dict = {}

                # TODO: check if occluded and exclude if True
                if label['category'] == 'traffic sign':
                    # if label['category'] == 'traffic sign' or label['category'] == 'traffic sign frame':
                    # label_dict['id'] = label['id']
                    label_dict['id'] = 26
                    label_dict['category_id'] = label['category']
                    for sign in label['poly2d']:
                        vertices = sign['vertices']
                        vertices = np.array(vertices)

                        x_cords, y_cords = vertices[:, 0], vertices[:, 1]
                        xmin = min(x_cords)
                        xmax = max(x_cords)
                        ymin = min(y_cords)
                        ymax = max(y_cords)
                        width = xmax - xmin
                        height = ymax - ymin

                        label_dict['area'] = int(width) * int(height)
                        label_dict['bbox'] = [int(xmin), int(ymin), int(width), int(height)]
                        segments_info.append(label_dict)
            annotation['segments_info'] = segments_info
            panoptic_per_image_id[image_id] = annotation

    # mapillary
    if DATASET == 'mapillaryvistas':
        img_path = join(data_dir, 'images')
        label_path = join(data_dir, 'v2.0/panoptic/')
        filenames = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    elif DATASET == 'bdd100k':
        # data_dir = '/data/shared/bdd100k/images/10k/train/'
        label_path = '/data/shared/bdd100k/labels/pan_seg/bitmasks/train/'
        filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        img_path = data_dir

    filenames.sort()

    print('[INFO] running detection algorithm')
    save_paths = [join(data_dir, 'traffic_signs'), join(data_dir, 'masks')]
    for p in save_paths:
        makedirs(p, exist_ok=True)

    offset_df = pd.DataFrame(columns=['filename', 'obj_id', 'xmin', 'ymin', 'xmin_ratio', 'ymin_ratio'])
    for filename in tqdm(filenames):
        output = crop_traffic_signs(
            filename, panoptic_per_image_id, img_path, label_path,
            min_area=min_area, pad=0.)
        save_images(output, filename.split('.')[0], save_paths)
        offset_df = save_offset(output, filename.split('.')[0], save_paths, offset_df)
    offset_df.to_csv(f'offset_{SPLIT}.csv', index=False)


def save_images(output, filename, paths):
    for img, mask, obj_id in zip(output['images'], output['masks'], output['obj_id']):
        Image.fromarray(img, 'RGB').save(join(paths[0], f'{filename}_{obj_id}.png'))
        Image.fromarray(mask * 255).save(join(paths[1], f'{filename}_{obj_id}.png'))


def save_offset(output, filename, paths, offset_df):
    # for obj_id in output['obj_id']:
    for obj_id, xmin, ymin, xmin_ratio, ymin_ratio in zip(
            output['obj_id'],
            output['offset_x'],
            output['offset_y'],
            output['offset_x_ratio'],
            output['offset_y_ratio']):
        offset_df = offset_df.append(
            {'filename': f'{filename}.png', 'obj_id': obj_id, 'xmin': xmin, 'ymin': ymin, 'xmin_ratio': xmin_ratio,
             'ymin_ratio': ymin_ratio},
            ignore_index=True)
    return offset_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Preperation', add_help=False)
    parser.add_argument('--split', default='training', type=str)
    args = parser.parse_args()
    SPLIT = args.split
    main()
