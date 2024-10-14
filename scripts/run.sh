python train.py --dataset porto --use_pre --use_emb --pre_epochs 50 --diff_inc 3 --pretrained_emb emb/porto_weighted.emb --device cuda:0 --channel_size 256 --batch_size 4096

python train.py --dataset sz --use_pre --use_emb --pre_epochs 60 --diff_inc 5 --pretrained_emb emb/sz_weighted.emb --batch_size 2048 --channel_size 128 --device cuda:1 

python train.py --dataset cd --use_pre --use_emb --pre_epochs 3 --diff_inc 5 --pretrained_emb emb/cd_weighted.emb --device cuda:2 --batch_size 1024 --channel_size 256 --test_epoch 1