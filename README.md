# Deformable Superquadric Network (DSQNet)
The official repository for &lt;DSQNet: A Deformable Model-Based Supervised Learning Algorithm for Grasping Unknown Occluded Objects> (Seungyeon Kim<sup>\*</sup>, Taegyun Ahn<sup>\*</sup>, Yonghyeon Lee, Jihwan Kim, Michael Yu Wang, and Frank C. Park, T-ASE 2022).

<sup>\*</sup> The two lead co-authors contributed equally.

> The paper proposes a recognition-based grasping method that merges a richer set of shape primitives, the deformable superquadrics, with a deep learning network, DSQNet, that is trained to identify complete object shapes from partial point cloud data.

- *[Paper](https://ieeexplore.ieee.org/abstract/document/9802912)* 
- *[Supplementary video](https://ieeexplore.ieee.org/abstract/document/9802912/media#media)* 

## Preview
![pipeline](figures/pipeline.png)
<I>Figure 1: Pipeline for proposed recognition-based grasping algorithm: (i) a trained segmentation network is used to segment a partially observed point cloud into a set of simpler point clouds; (ii) The trained DSQNet converts each point cloud into a deformable superquadric primitive, with its collective union representing the full object shape; (iii) grasp poses are generated in a gripper-dependent manner from the recognized full shapes. </I>

## Progress
- [x] DSQNet training script (`train.py`)
- [x] Segmentation network training script (`train.py`)
- [x] Dataset upload
- [ ] Pre-trained model upload
- [ ] Evaluation script (`evaluation.py`)
- [x] Requirements update
- [ ] Data generation script (`data_generation.py`)

## Requirements
### Environment
The project codes are tested in the following environment.
- python 3.8.13
- pytorch 1.12.0
- tensorboard 2.9.1
- pandas
- scikit-learn
- Open3D

### Datasets
Datasets should be stored in `datasets/` directory. Datasets can be downloaded through the [Google drive link](https://drive.google.com/drive/folders/1PQ9dSeD0WmdESQemsnM1SPmpPDChQ95s?usp=sharing). After set up, the `datasets/` directory should be as follows.
```
datasets
├── primitive_dataset
│   ├── box
│   ├── ... (4 more folders)
│   ├── truncated_torus
│   ├── train_datalist.csv
│   ├── validation_datalist.csv
│   └── test_datalist.csv
└── object_dataset
    ├── bottle_cone
    ├── ... (10 more folders)
    ├── truncated_torus
    ├── train_datalist.csv
    ├── validation_datalist.csv
    └── test_datalist.csv

```
- (Optional) If you want to generate your own custom dataset, run the following script:
```
preparing...
```
> **Tips for playing with code:** preparing...

### Pretrained model
Pre-trained models should be stored in `pretrained/`. The pre-trained models are provided through the [Google drive link](https://drive.google.com/drive/folders/1PN7DF0iNL60iOuyA-QS2g7jMzXSOPD6a?usp=sharing). After set up, the `pretrained/` directory should be follows.
```
pretrained
├── segnet
│   ├── segnet_config.yml
│   └── model_best.pkl
├── sqnet
│   ├── sqnet_config.yml
│   └── model_best.pkl
└── dsqnet
    ├── dsqnet_config.yml
    └── model_best.pkl
```

## Running
### Training
The training script is `train.py`. 
- `--config` specifies a path to a configuration yml file.
- `--logdir` specifies a directory where the results will be saved.
- `--run` specifies a name for an experiment.
- `--device` specifies an GPU number to use.

Training DSQNet (or SQNet) and segmentation network:
```
python train.py --config configs/{X}_config.yml
```
- `X` is either `sqnet`, `dsqnet` or `segnet`.

### Evaluation
preparing...

## Citation
```
@article{kim2022dsqnet,
  title={DSQNet: A Deformable Model-Based Supervised Learning Algorithm for Grasping Unknown Occluded Objects},
  author={Kim, Seungyeon and Ahn, Taegyun and Lee, Yonghyeon and Kim, Jihwan and Wang, Michael Yu and Park, Frank C},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2022},
  publisher={IEEE}
}
```


