# SleepSatelightFTC
A. Ito and T. Tanaka, "SleepSatelightFTC: A Lightweight and Interpretable Deep Learning Model for Single-Channel EEG-Based Sleep Stage Classification," in IEEE Access, vol. 13, pp. 46263-46272, 2025, doi: 10.1109/ACCESS.2025.3549436. [[paper](https://ieeexplore.ieee.org/document/10918694)]

## Dataset
For downloading PSG data and preprocessing, please use the publicly available repository [TinySleepNet](https://github.com/akaraspt/tinysleepnet).

Downsample the signals by 2 and place the resulting NPZ files under:  
(You can modify the directory path in `config.yaml` if needed.)

sleepedf_fs50/  
├── SC4001E0_fs50.npz  
├── SC4002E0_fs50.npz  
└── ...
 
## Training

**1. Train epoch-wise models (train_1)**

```bash
python train_1.py
```

**2. Train context model (train_2)**

```bash
python train_2.py
```
