# ClipHEMIIF-ReID: CLIP-Guided Hard Example Mining and Inter-Image Fusion for Person Re-Identification

**Supplementary material: Code implementation in pytorch for the ITC submission "ClipHEMIIF-ReID: CLIP-Guided Hard Example Mining and Inter-Image Fusion for Person Re-Identification".**

### Installation

```
ftfy==6.1.3
ipdb==0.13.13
matplotlib==3.5.3
numpy==1.21.5
opencv_python==4.5.1.48
Pillow==9.3.0
regex==2022.10.31
scikit_learn==1.0.2
scipy==1.7.3
timm==0.3.2
torch==1.13.1+cu116
torchvision==0.14.1+cu116
tqdm==4.65.0
yacs==0.1.8
```

### Prepare Dataset

Download the datasets ([Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset)), and then unzip them to `your_dataset_dir`.

## Prepare Pre-trained Models

Download pre-trained models from [SOLIDER]([tinyvision/SOLIDER-REID (github.com)](https://github.com/tinyvision/SOLIDER-REID)), Before training, you should convert the models as shown in the repository.

### Training

First, you need to load the pre-trained models. Modify the MODEL.PRETRAIN_PATH in configs/person/vit_clipreid.yml.

If you want to run for the Market-1501, you need to modify the bottom of configs/person/vit_clipreid.yml to

```
DATASETS:
   NAMES: ('market1501')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```

then run

```
python train_clipreid.py --config_file configs/person/vit_clipreid.yml MODEL.SEMANTIC_WEIGHT 0.2
```

### Evaluation

For example, if you want to test for MSMT17

```
python test_clipreid.py --config_file configs/person/vit_clipreid.yml MODEL.SEMANTIC_WEIGHT 0.2 TEST.WEIGHT 'your_trained_checkpoints_path'
```

