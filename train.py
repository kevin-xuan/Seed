import os
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import geopandas as gpd
from config import args
import datetime
from models.DDIM import DDIM
import pandas as pd
import shutil
from logger import Logger, log_info
from utils import EMA, dataloader, set_seed, construct_segment_graph
import time
from torch.utils.data import DataLoader, RandomSampler
from models.dataset import PretrainDataset
from tqdm import tqdm

def print_model(model, logger):
    weight_decay_list = []
    no_decay_list = []
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            logger.info(name + ',   ' + str(param.numel()))
            param_count += param.numel()
        if 'bias' in name or 'bn' in name:
            no_decay_list.append(param) 
        else:
            weight_decay_list.append(param)
    
    parameters = [{'params': weight_decay_list}, {'params': no_decay_list, 'weight_decay': 0.}]
    logger.info(f'In total: {param_count} trainable parameters.')
    return parameters

if __name__ == '__main__':
    set_seed()
    n_steps = args.n_steps  
    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)
    device = args.device
    batch_size = args.batch_size  
    in_channel = args.channel_size
    if dataset == 'porto':
        boundary = {'min_lat': 41.147, 'max_lat': 41.178, 'min_lng': -8.65, 'max_lng': -8.53}
    elif dataset == 'sz':
        boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    elif dataset == 'cd':
        boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lng': 104, 'max_lng': 104.16}
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataset))
    grid_num = args.grid_num 
    grid_size = args.grid_size
    edges = pd.read_csv(os.path.join(data_path, 'edges.csv'), header=0, usecols=['u', 'v', 'length', 'fid', 'geometry'])
    args.segment_num = len(edges)
    
    save_dir = f'./save/{dataset}'
    if not os.path.exists(f'./save/{dataset}'):
        os.makedirs(f'./save/{dataset}', exist_ok=True)
    
    # Logger
    result_name = '{}_steps={}_len={}_channel={}_bs={}'.format(
        dataset, n_steps,
        args.max_len, 
        in_channel,
        batch_size)
    exp_dir = os.path.join(save_dir, "Seed", result_name)
    log_dir = os.path.join(exp_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    if not os.path.exists(os.path.join(log_dir, timestamp)):
        os.makedirs(os.path.join(log_dir, timestamp), exist_ok=True)
    logger = Logger(
        __name__,
        log_path=os.path.join(log_dir, timestamp, timestamp + '.log'),
        colorize=True,
    )
    model_path = os.path.join(os.path.join(log_dir, timestamp), 'model_ddim.pth')
    shutil.copy2('models/DDIM.py', os.path.join(log_dir, timestamp))
    shutil.copy2('models/Traj_Transformer.py', os.path.join(log_dir, timestamp))
    shutil.copy2('config.py', os.path.join(log_dir, timestamp))
    log_info(args, logger)
    
    segment_graph = construct_segment_graph(edges)
    ddim = DDIM(args, segment_graph)
    
    params = print_model(ddim.model, logger)

    # diffusion config
    alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt = ddim.diffusion_config(n_steps)
    
    trainData, train_trajs, testData, test_trajs = dataloader(data_path, batch_size, device)

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(ddim.model)

    optimizer = optim.Adam(ddim.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    args.mask_id = args.segment_num
    supervised_dataset = PretrainDataset(args, train_trajs)
    supervised_sampler = RandomSampler(supervised_dataset)
    pretrain_dataloader = DataLoader(supervised_dataset, sampler=supervised_sampler, batch_size=args.batch_size)
            
    #* curriculum learning
    if args.use_pre:
        for epoch in range(args.pre_epochs): 
            pretrain_data_iter = tqdm(enumerate(pretrain_dataloader),
                                        desc=f"Seed-{args.dataset} Epoch:{epoch}",
                                        total=len(pretrain_dataloader),
                                        bar_format="{l_bar}{r_bar}")
            ddim.model.train()
            train_start = time.time()
            epoch_loss = 0.0
            epoch_nmp_loss = 0.0
            epoch_cross_loss = 0.0
            epoch_mse_loss = 0.0
            epoch_mse2_loss = 0.0
            if epoch == 0:
                level = 1
            else:
                level = min(epoch * args.diff_inc, args.n_steps-1)
                
            for i, batch in pretrain_data_iter:
                batch = batch.to(device)
                nmp_loss, cross_loss_nmp, mse_loss_nmp, mse_loss2_nmp = ddim.pretrain_mask2(ddim.model, batch, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, level)

                loss = nmp_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_nmp_loss += nmp_loss.item()
                epoch_cross_loss += cross_loss_nmp.item()
                epoch_mse_loss += mse_loss_nmp.item()
                epoch_mse2_loss += mse_loss2_nmp.item()
                
            epoch_loss /= len(pretrain_data_iter)
            epoch_nmp_loss /= len(pretrain_data_iter)
            epoch_cross_loss /= len(pretrain_data_iter)
            epoch_mse_loss /= len(pretrain_data_iter)
            epoch_mse2_loss /= len(pretrain_data_iter)
        
            train_end = time.time()
            logger.info(f"Pre-training Epoch: {epoch + 1}, Level: {level}, Time: {'%.2fs' % (train_end-train_start)}, pre-training loss: {'%.4f' % epoch_loss}, nmp loss: {'%.4f' % epoch_nmp_loss}, cross loss: {'%.4f' % epoch_cross_loss}, mse loss: {'%.4f' % epoch_mse_loss}, mse loss2: {'%.4f' % epoch_mse2_loss}")
        
        logger.info(f'Curriculum learning finished...')
    
    # train
    for epoch in range(args.epochs): 
        train_start = time.time()
        epoch_loss, epoch_cross_loss, epoch_mse_loss, epoch_mse2_loss = 0, 0, 0, 0
        ddim.model.train()
        for index, trajs in trainData:
            trajs = trajs.long().to(device)
            loss, cross_loss, mse_loss, mse_loss2 = ddim.noise_estimation_loss(ddim.model, trajs, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cross_loss += cross_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_mse2_loss += mse_loss2.item()
            
        epoch_loss /= len(trainData)
        epoch_cross_loss /= len(trainData)
        epoch_mse_loss /= len(trainData)
        epoch_mse2_loss /= len(trainData)
        train_end = time.time()
        
        logger.info(f"Epoch: {epoch + 1}, Time: {'%.2fs' % (train_end-train_start)}, total loss: {'%.8f' % epoch_loss}, cross loss: {'%.8f' % epoch_cross_loss}, mse loss: {'%.8f' % epoch_mse_loss}, mse loss2: {'%.8f' % epoch_mse2_loss}")
        
        if (epoch + 1) % args.test_epoch == 0:
            eval_start = time.time()
            torch.save(ddim.model.state_dict(), model_path)
            ddim.model.load_state_dict(torch.load(model_path, map_location=device))
            
            logger.info(f'Epoch {epoch + 1} finished, generating trajectories...')
            gen_trajs = []

            ddim.model.eval()
            gene_s = time.time()
            with torch.no_grad():
                for index, trajs in testData:
                    x_seq = ddim.reverse_process(ddim.model, trajs.shape, n_steps, (1 - alphas).to(device))
                    gen_traj = x_seq.cpu().numpy()
                    gen_trajs.append(gen_traj) 
            
            gen_trajs = np.concatenate(gen_trajs, axis=0)
            with open(os.path.join(log_dir, timestamp, 'gene_epoch_{}.data'.format(epoch+1)), 'w') as fout:
                for sample in gen_trajs:
                    string = ' '.join([str(s) for s in sample.tolist()])
                    fout.write('%s\n' % string)   
               
            logger.info('generating {} trajectories but only {} unique trajectories, needs {}s'.format(len(gen_trajs), len(np.unique(gen_trajs, axis=0)), time.time() - gene_s))
                   
