# Seed

This is the implementation for the paper: “Seed: Bridging Sequence and Diffusion Models for Road Trajectory Generation.” 

## Preliminaries

### Conda Environment

```bash
  conda env create -f environment.yml
  ```

### Requirements
* Python 3.11.4 
* pytorch 2.1.1 
* pandas 2.0.3 
* numpy 1.24.3

<!-- ## Datasets

We use [SIN, NYC](https://sites.google.com/site/yangdingqi/home) and [Gowalla](http://snap.stanford.edu/data/loc-gowalla.html) datasets. The processed SIN and Gowalla datasets are from [ARGAN](https://github.com/wangzb11/AGRAN), and we preprocess NYC dataset by **data_process.py**. For more details of data preprocessing, please refer to our paper or **data_process.py**:

```
python data_process.py
``` -->




## Model Training

To train our model on the Porto dataset:

```
python train.py --dataset porto --use_pre --use_emb --pre_epochs 50 --diff_inc 3 --pretrained_emb node2vec/emb/porto_weighted.emb --device cuda:0 --channel_size 256 --batch_size 4096
```


<!-- ## Acknowledgement

The code is implemented based on [ARGAN](https://github.com/wangzb11/AGRAN). -->
