import numpy as np
import pandas as pd
import os
from PIL import Image
import imagehash
from tqdm import tqdm

mapillary_hashes_df = pd.DataFrame()
mapillary_filenames = []
mapillary_splits = []
mapillary_a_hashes = []
mapillary_p_hashes = []
mapillary_d_hashes = []
mapillary_w_hashes = []

HASH_SIZE = 64

for split in ['training', 'validation']:
    mapillary_image_path = f'/data/shared/mapillary_vistas/{split}/images_original/'
    mapillary_vistas_files = os.listdir(mapillary_image_path)
    print(f'num mapillary {split} files', len(mapillary_vistas_files))
    for filename in tqdm(mapillary_vistas_files):
        filepath = os.path.join(mapillary_image_path, filename)
        img = Image.open(filepath)
        a_hash = imagehash.average_hash(img, hash_size=HASH_SIZE)
        p_hash = imagehash.phash(img, hash_size=HASH_SIZE)
        d_hash = imagehash.dhash(img, hash_size=HASH_SIZE)
        w_hash = imagehash.whash(img, hash_size=HASH_SIZE)

        mapillary_filenames.append(filename)
        mapillary_splits.append(split)
        mapillary_a_hashes.append(a_hash)
        mapillary_p_hashes.append(p_hash)
        mapillary_d_hashes.append(d_hash)
        mapillary_w_hashes.append(w_hash)

mapillary_hashes_df['filename'] = mapillary_filenames
mapillary_hashes_df['split'] = mapillary_splits
mapillary_hashes_df['a_hash'] = mapillary_a_hashes
mapillary_hashes_df['p_hash'] = mapillary_p_hashes
mapillary_hashes_df['d_hash'] = mapillary_d_hashes
mapillary_hashes_df['w_hash'] = mapillary_w_hashes

print('[INFO] saving mapillary df')
mapillary_hashes_df.to_csv('mapillary_vistas_hashes.csv', index=False)

# overlap_images = 0
# overlapping_image_filename = []

mtsd_hashes_df = pd.DataFrame()
mtsd_filenames = []
mtsd_splits = []
mtsd_a_hashes = []
mtsd_p_hashes = []
mtsd_d_hashes = []
mtsd_w_hashes = []

for split in ['train', 'val']:
    mtsd_image_path = f'/data/shared/mtsd_v2_fully_annotated/images/{split}'
    mtsd_files = os.listdir(mtsd_image_path)
    print(f'num mtsd {split} files', len(mtsd_files))
    for filename in tqdm(mtsd_files):
        filepath = os.path.join(mtsd_image_path, filename)
        img = Image.open(filepath)
        a_hash = imagehash.average_hash(img, hash_size=HASH_SIZE)
        p_hash = imagehash.phash(img, hash_size=HASH_SIZE)
        d_hash = imagehash.dhash(img, hash_size=HASH_SIZE)
        w_hash = imagehash.whash(img, hash_size=HASH_SIZE)

        mtsd_filenames.append(filename)
        mtsd_splits.append(split)
        mtsd_a_hashes.append(a_hash)
        mtsd_p_hashes.append(p_hash)
        mtsd_d_hashes.append(d_hash)
        mtsd_w_hashes.append(w_hash)

mtsd_hashes_df['filename'] = mtsd_filenames
mtsd_hashes_df['split'] = mtsd_splits
mtsd_hashes_df['a_hash'] = mtsd_a_hashes
mtsd_hashes_df['p_hash'] = mtsd_p_hashes
mtsd_hashes_df['d_hash'] = mtsd_d_hashes
mtsd_hashes_df['w_hash'] = mtsd_w_hashes

print('[INFO] saving mtsd df')
mtsd_hashes_df.to_csv('mtsd_hashes.csv', index=False)

        # if hash in mapillary_hashes:
        #     overlapping_image_filename.append(filename)
        #     overlap_images += 1
# print('num overlap images', overlap_images)

# print('[INFO] saving images')
# with open("mtsd_overlapping_filenames.txt", "w") as output:
#     for filename in overlapping_image_filename:
#         output.write(str(filename) + '\n')
