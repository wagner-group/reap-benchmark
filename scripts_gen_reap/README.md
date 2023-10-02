# Dataset Preparation

This instruction is outdated.

## Recreate REAP from Mapillary Vistas and MTSD

Coming soon!

- `mtsd_label_metadata.csv` is a mapping between the original MTSD classes to classes in REAP. It contains shapes and sizes for each MTSD class.

## Mapillary Vistas

- `prep_mapillary.py`: Prepare Vistas dataset for YOLOv5 using a pre-trained classifier to determine classes of the signs. May require substantial memory to run. Insufficient memory can lead to the script getting killed with no error message.

The original dataset should have the following structure:

```text
mapillary_vistas/
|-- testing
|   `-- images
|-- training
|   |-- images
|   |-- masks
|   `-- v2.0
|       |-- instances
|       |-- labels
|       |-- panoptic
|       `-- polygons
`-- validation
    |-- images
    |-- masks
    `-- v2.0
        |-- instances
        |-- labels
        |-- panoptic
        `-- polygons
```

After running all the preparation scripts, the dataset should have the following structure:

```text
mapillary_vistas/
|-- no_color
|   `-- combined
|       |-- images
|       `-- labels
|-- testing
|   `-- images
|-- training
|   |-- images
|   |-- labels_no_color
|   |-- masks
|   `-- v2.0
|       |-- instances
|       |-- labels
|       |-- panoptic
|       `-- polygons
`-- validation
    |-- images
    |-- labels_no_color
    |-- masks
    `-- v2.0
        |-- instances
        |-- labels
        |-- panoptic
        `-- polygons
```

```bash
MODIFIER="no_color"

# Dataset should be extracted to ~/data/mapillary_vistas (use symlink if needed)
python prep_mapillary.py --split train --resume PATH_TO_CLASSIFIER
python prep_mapillary.py --split val --resume PATH_TO_CLASSIFIER

# Combined train and val partition into "combined"
BASE_DIR=~/data/mapillary_vistas
cd $BASE_DIR && mkdir $MODIFIER && cd $MODIFIER
mkdir combined && cd combined && mkdir images labels
ln -s $BASE_DIR/training/images/* images/
ln -s $BASE_DIR/validation/images/* images/
ln -s $BASE_DIR/training/labels_$MODIFIER/* labels/
ln -s $BASE_DIR/validation/labels_$MODIFIER/* labels/
```
