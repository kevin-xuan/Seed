# Seed [WWW 2025]

This is the implementation for the paper: “Seed: Bridging Sequence and Diffusion Models for Road Trajectory Generation.” 

## Preliminaries

### Data Preparation
```bash
  unzip data.zip 
  unzip emb.zip
  ```

### Conda Environment

```bash
  conda env create -f environment.yml
  ```

## Model Training

To train our model on the Porto dataset (See scripts/run.sh):

```
python train.py --dataset porto --use_pre --use_emb --pre_epochs 50 --diff_inc 3 --pretrained_emb emb/porto_weighted.emb --filename ./node2vec/graph/porto.edgelist --device cuda:0 --channel_size 256 --batch_size 4096
```


## Acknowledgement

The code is implemented based on [DiffTraj](https://github.com/Yasoz/DiffTraj).

## Citing

If you use Seed in your research, please cite the following [paper](https://dl.acm.org/doi/10.1145/3696410.3714951):
```
@inproceedings{DBLP:conf/www/RaoSJ0025,
  author       = {Xuan Rao and
                  Shuo Shang and
                  Renhe Jiang and
                  Peng Han and
                  Lisi Chen},
  title        = {Seed: Bridging Sequence and Diffusion Models for Road Trajectory Generation},
  booktitle    = {Proceedings of the {ACM} on Web Conference 2025, {WWW} 2025, Sydney,
                  NSW, Australia, 28 April 2025- 2 May 2025},
  pages        = {2007--2017},
  year         = {2025}
}
```