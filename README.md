# REAP: A Large-Scale Realistic Adversarial Patch Benchmark

Nabeel Hingun\* (UC Berkeley), Chawin Sitawarin\* (UC Berkeley), Jerry Li (Microsoft), David Wagner (UC Berkeley)

## Abstract

Machine learning models are known to be susceptible to adversarial perturbation.
One famous attack is the _adversarial patch_, a sticker with a particularly crafted pattern that makes the model incorrectly predict the object it is placed on.
This attack presents a critical threat to cyber-physical systems that rely on cameras such as autonomous cars.
Despite the significance of the problem, conducting research in this setting has been difficult;
evaluating attacks and defenses in the real world is exceptionally costly while synthetic data are unrealistic.
In this work, we propose the REAP (REalistic Adversarial Patch) benchmark, a digital benchmark that allows the user to evaluate patch attacks on real images, and under real-world conditions.
Built on top of the Mapillary Vistas dataset, our benchmark contains over 14,000 traffic signs.
Each sign is augmented with a pair of geometric and lighting transformations, which can be used to apply a digitally generated patch realistically onto the sign.
Using our benchmark, we perform the first large-scale assessments of adversarial patch attacks under realistic conditions.
Our experiments suggest that adversarial patch attacks may present a smaller threat than previously believed and that the success rate of an attack on simpler digital simulations is not predictive of its actual effectiveness in practice.

TODO: Sample images

## Package Dependencies

Tested with

- `python >= 3.8`.
- `cuda >= 11.2`.
- `kornia == 0.6.3`: Using version `>= 0.6.4` will raise an error.
- See `requirements.txt` for all packages' version.

We recommend creating a new python environment because `kornia` and `detectron2` seem to often mess up dependencies and result in a segmentation fault.

```[bash]
# Install from requirements.txt file OR
pip install -r requirements.txt

# Install packages with their latest version manually, e.g.,
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install scipy pandas scikit-learn pip seaborn
pip install timm kornia==0.6.3 opencv-python albumentations

# Detectron2 has to be installed afterward
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

- If there is any problem with `detectron2` installation (e.g., CUDA or `pytorch` version mismatch), see this [documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Dataset Preparation

### MTSD

- [MTSD](https://www.mapillary.com/dataset/trafficsign) is used for training the traffic sign detection models and the classifier used to create REAP.
- Have not found a way to automatically download the dataset.
- `prep_mtsd_for_yolo.py`: Prepare MTSD dataset for YOLOv5.
- YOLO expects samples and labels in `BASE_DIR/images/` and `BASE_DIR/labels/`, respectively. See [link](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#13-organize-directories) for more detail.
- Training set: MTSD training. Symlink to `~/data/yolo_data/(images or labels)/train`.
- Validation set: MTSD validation Symlink to `~/data/yolo_data/(images or labels)/val`.
- If you run into `Argument list too long` error, try to raise limit of argument stack size by `ulimit -S -s 100000000`. [link](https://unix.stackexchange.com/a/401797)
<!-- - Test set: Combine Vistas training and validation. Symlink to `~/data/yolo_data/(images or labels)/test`. -->

```[bash]
# Prepare MTSD dataset
# Dataset should be extracted to ~/data/mtsd_v2_fully_annotated
python prep_mtsd_for_yolo.py
python prep_mtsd_for_detectron.py
# FIXME: change yolo_data
LABEL_NAME=labels_no_color
cd ~/data/ && mkdir yolo_data && mkdir yolo_data/images yolo_data/labels

cd ~/data/yolo_data/images/
ln -s ~/data/mtsd_v2_fully_annotated/images/train train
ln -s ~/data/mtsd_v2_fully_annotated/images/val val
cd ~/data/yolo_data/labels/
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/train train
ln -s ~/data/mtsd_v2_fully_annotated/$LABEL_NAME/val val
```

### Mapillary Vistas

- `prep_mapillary.py`: Prepare Vistas dataset for YOLOv5 using a pre-trained classifier to determine classes of the signs. May require substantial memory to run. Insufficient memory can lead to the script getting killed with no error message.

```[bash]
# Dataset should be extracted to ~/data/mapillary_vistas (use symlink if needed)
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split train --resume PATH_TO_CLASSIFIER
CUDA_VISIBLE_DEVICES=0 python prep_mapillary.py --split val --resume PATH_TO_CLASSIFIER

# Combined train and val partition into "combined"
BASE_DIR=~/data/mapillary_vistas
cd $BASE_DIR
mkdir no_color && cd no_color
mkdir combined && cd combined
mkdir images labels detectron_labels
ln -s $BASE_DIR/training/images/* images/
ln -s $BASE_DIR/validation/images/* images/
ln -s $BASE_DIR/training/labels_no_color/* labels/
ln -s $BASE_DIR/validation/labels_no_color/* labels/
ln -s $BASE_DIR/training/detectron_labels_no_color/* detectron_labels/
ln -s $BASE_DIR/validation/detectron_labels_no_color/* detectron_labels/
```

## Usage

### Use REAP benchmark for evaluation

- `reap_annotations.csv` is the REAP annotation file.
- `configs` contains attack config files and detectron (Faster R-CNN) config files.

<!-- ## Other Tips -->

- To run on annotated signs only (consistent with results in the paper), use flag `--annotated-signs-only`. For Detectron2, the dataset cache has to be deleted before this option to really take effect.

### Recreate REAP from Mapillary Vistas and MTSD

Coming soon!

- `mtsd_label_metadata.csv` is a mapping between the original MTSD classes to classes in REAP. It contains shapes and sizes for each MTSD class.

## TODOs

- `NewDataset`: Changes required to make an addition of new dataset possible.
- `AnnoObj`: Changes from keeping annotation in `pd.DataFrame` to a new object.
- `YOLO`: Implement new changes to YOLO code.
- `enhancement`: Minor documentation or readability improvement.
  - Change interpolation (`interp`) type to `Enum` instead of `str`.
- `feature`: New features that would benefit future attack and defense experiments.

There are signs that may appear in an image but do not have an annotation. There are multiple reasons this happens:

1. The sign do not belong to one of the 11 non-background classes. Most of the signs fall into this category, but there could be some that result from a mistake made by the classifier we trained.
2. The sign is not labeled in the Mapillary Vistas dataset. If the sign is not labeled, our annotation script will not even know it exists.
3. The sign is too small, and so it are filtered out before our annotation process since its transformation parameters would be unreliable. This type of signs likely has to be manually annotated with extra care.

All of these can be fixed by adding or modifying an entry in `reap_annotations.csv`, but case 2 also requires adding the missing segmentation label to Mapillary Vistas labels.

## License

Our benchmark is based on the Mapillary Vistas dataset which uses Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
So it also shares the same license.
Please see the `LICENSE` file or this [link](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact

If you have any question or suggestion, please feel free to open an issue on this repository or directly contact Chawin Sitawarin (chawins@berkeley.edu) or Nabeel Hingun (nabeel126@berkeley.edu).
