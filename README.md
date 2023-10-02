# REAP: A Large-Scale Realistic Adversarial Patch Benchmark

Nabeel Hingun\* (UC Berkeley), Chawin Sitawarin\* (UC Berkeley), Jerry Li (Microsoft), David Wagner (UC Berkeley)

[ICCV'23](https://openaccess.thecvf.com/content/ICCV2023/html/Hingun_REAP_A_Large-Scale_Realistic_Adversarial_Patch_Benchmark_ICCV_2023_paper.html), [ArXiv](https://arxiv.org/abs/2212.05680)

![reap_vs_others](banner.png)

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

## Package Dependencies

Tested with

- `python >= 3.8`.
- `cuda >= 11.2`.
- `kornia >= 0.6.9`.
- See `requirements.txt` for all packages' version.

We recommend creating a new python environment because `kornia` and `detectron2` seem to often mess up dependencies and result in a segmentation fault.

```bash
# Install from requirements.txt file OR
pip install -r requirements.txt

# Install packages with their latest version manually, e.g.,
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install scipy pandas scikit-learn pip seaborn
pip install timm kornia opencv-python albumentations

# Detectron2 has to be installed afterward
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install mish_cuda for DarkNet backbone of YOLO (optional). If you have
# trouble installing mish_cuda, you can rename _mish_cuda.py to mish_cuda.py to
# silence some warning from YOLOv7.
git clone https://github.com/thomasbrandon/mish-cuda
cd mish-cuda
# Uncomment this if see "fatal error: CUDAApplyUtils.cuh: No such file or directory"
# mv external/CUDAApplyUtils.cuh csrc/
python setup.py build install

# Install detrex (required for DINO)
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init
git submodule update
python -m pip install -e detectron2
pip install -e .

# Install yolof (required for YOLOF)
git clone https://github.com/chensnathan/YOLOF.git
cd YOLOF
python setup.py develop

# Install YOLOv7 (required Detectron2)
git clone https://github.com/jinfagang/yolov7_d2
cd yolov7_d2
pip install -e .
pip install alfred-py
```

- If there is any problem with `detectron2` installation (e.g., CUDA or `pytorch` version mismatch), see this [documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Clone this repository and install it. Make sure that [git-lfs](https://git-lfs.com/) is installed.

```bash
git lfs install
git clone https://github.com/wagner-group/reap-benchmark.git
```

## Dataset and Weights

The weights are automatically downloaded when the repo was cloned (using `git lfs`) and are placed in `./weights/` directory.

REAP dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/csitawarin/reap-benchmark/).
This includes REAP, REAP-Shape, and the synthetic datasets.
There is also a CLI tool for downloading the dataset from Kaggle ([link](https://www.kaggle.com/docs/api#interacting-with-datasets), [link](https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/)).

```bash
mv archive data/reap
cd data/reap/no_color/combined && ln -s ../../100/combined/images
```

Scripts for recreating REAP benchmark from Mapillary Vistas and MTSD can be found in `scripts_gen_reap` directory, but they are outdated.

## Usage

### Examples

```bash
bash scripts/example_test_reap.sh
```

Running evaluation on clean data:

- `scripts/example_test_reap.sh`: run evaluation on the REAP benchmark without any adversarial patch on one class only.
- `scripts/example_test_reap_shape.sh`: same as `scripts/example_test_reap.sh` but for REAP-Shape.
- `scripts/example_test_synthetic.sh`: run evaluation on the synthetic benchmark without any adversarial patch on one class only.

Running attack:

- `scripts/example_atk_reap.sh`: generate an adversarial patch for one class, and evaluate it on the REAP benchmark.
- `scripts/example_atk_reap_shape.sh`: same as `scripts/example_atk_reap.sh` but for REAP-Shape.

Training models:

- `scripts/example_train.sh`: (NOT TESTED) example training script including both normal and adversarial training.

Utility:

- `print_results_to_csv.py`: gather results and print them in CSV format.

### Computing Relighting Params

**Realism Test.** Run script to test out different relighting methods on the printed signs and patches.

```bash
python run_realism_test.py
```

After deciding on the relighting method, set variables in `gen_relight_coeffs_main.py` and run the script to generate the relighting transform's parameters for a given relighting method and then write to `reap_annotations.csv`.

```bash
bash scripts/gen_relight_coeffs.sh
```

## File Structure

### Main Python Files

- `test_main.py`: main script for running all experiments; run evaluation either with or without adversarial patch.
- `gen_adv_main.py`: generate adversarial patch.

### Configs

- `configs` contains all config files for both the models and the experiments.
- `configs/cfg_reap_base.yaml`: base config for evaluating on REAP benchmark. Used as base config for running all experiments on `reap`, `reap_shape`, and `synthetic`.
- Config file is specified by `-e` argument in `test_main.py` and `gen_adv_main.py`, e.g., `python test_main.py -e configs/cfg_reap_base.yaml ...`.
- Config parameters are overwritten by command line arguments and custom options, e.g., `python test_main.py -e configs/cfg_reap_base.yaml --options base.dataset=reap_shape ...` See `scripts/example_test_reap.sh` for more examples.

### Others

- `data/reap_annotations.csv` is the REAP annotation file.

## TODOs

- [ ] `NewDataset`: Changes required to make an addition of new dataset possible.
- [ ] `AnnoObj`: Changes from keeping annotation in `pd.DataFrame` to a new object.
- [ ] `YOLO`: Implement new changes to YOLO code.
- [ ] `enhancement`: Minor documentation or readability improvement.
  - [ ] Change interpolation (`interp`) type to `Enum` instead of `str`.
- [ ] `feature`: New features that would benefit future attack and defense experiments.

There are signs that may appear in an image but do not have an annotation. There are multiple reasons this happens:

1. The sign do not belong to one of the 11 non-background classes. Most of the signs fall into this category, but there could be some that result from a mistake made by the classifier we trained.
2. The sign is not labeled in the Mapillary Vistas dataset. If the sign is not labeled, our annotation script will not even know it exists.
3. The sign is too small, and so it are filtered out before our annotation process since its transformation parameters would be unreliable. This type of signs likely has to be manually annotated with extra care.

All of these can be fixed by adding or modifying an entry in `reap_annotations.csv`, but case 2 also requires adding the missing segmentation label to Mapillary Vistas labels.

## License

Our benchmark is based on the Mapillary Vistas dataset which uses Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
So it also shares the same license.
Please see the `LICENSE` file or this [link](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This software and/or data was deposited in the BAIR Open Research Commons repository on Feb 28, 2023.

## Contact

If you have any question or suggestion, please feel free to open an issue on this repository or directly contact Chawin Sitawarin ([chawins@berkeley.edu](chawins@berkeley.edu)) or Nabeel Hingun ([nabeel126@berkeley.edu](nabeel126@berkeley.edu)).
