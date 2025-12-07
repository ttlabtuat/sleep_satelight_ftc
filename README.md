# SleepSatelightFTC
SleepSatelightFTC: A Lightweight and Interpretable Deep Learning Model for Single-Channel EEG-Based Sleep Stage Classification by Aozora Ito and Toshihisa Tanaka from Department of Electrical Engineering and Computer Science, Tokyo University of Agriculture and Technology


## Dataset
For downloading PSG data and preprocessing, please use the publicly available repository TinySleepNet:

https://github.com/akaraspt/tinysleepnet

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
