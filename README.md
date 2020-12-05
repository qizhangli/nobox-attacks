# Nobox-Attacks
This repository contains a PyTorch implementation for our paper ***Practical No-box Adversarial Attacks against DNNs***. 

## Environments
* Python 3.7.0
* PyTorch 1.3.0
* torchvision 0.4.1
* Pillow 7.0.0

## Datasets
Prepare ImageNet into the following structure:
```
Nobox-attack
└───data
    ├── selected_data.csv
    └── ILSVRC2012_img_val
        ├── n01440764
        ├── n01443537
        └── ...
    
```
Select images from ImageNet validation set, and write ```data/selected_data.csv``` as following:
```
n01440764,ILSVRC2012_val_00002138.JPEG,ILSVRC2012_val_00006697.JPEG,ILSVRC2012_val_00009111.JPEG,...
n01484850,ILSVRC2012_val_00002752.JPEG,ILSVRC2012_val_00016988.JPEG,ILSVRC2012_val_00004329.JPEG,...
...
```
## Usage
To train substitute models on Imagenet, run:
```
python3 train.py --n_imgs 20 --n_iters 15000 --n_decoders 20 --lr 0.001 --mode prototypical --save_dir ./trained_models
or
python3 train.py --n_imgs 20 --n_iters 1000 --n_decoders 20 --lr 0.001 --mode unsup_naive --save_dir ./trained_models
or
python3 train.py --n_imgs 20 --n_iters 2000 --n_decoders 20 --lr 0.001 --mode rotate --save_dir ./trained_models
or
python3 train.py --n_imgs 20 --n_iters 5000 --n_decoders 20 --lr 0.001 --mode jigsaw --save_dir ./trained_models
```


To mount an untargeted L-inf attack on ImageNet using substitute models, run:
```
python3 attack.py --epsilon 0.1 --ce_niters 200 --ila_niters 100 --n_imgs 20 --n_decoders 20 --ae_dir ./trained_models --ce_method ifgsm --mode prototypical
```
The ```--mode``` can be set as ```prototypical/rotate/jigsaw/unsup_naive```.


## Citation
```
@inproceedings{li2020practical,
    title={Practical No-box Adversarial Attacks against DNNs},
    author={Li, Qizhang and Guo, Yiwen and Chen, Hao},
    booktitle={NeurIPS},
    year={2020}
}
```
