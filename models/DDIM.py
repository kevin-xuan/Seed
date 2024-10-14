import torch
import torch.nn.functional as F
import torch.nn as nn
from .Traj_Transformer import Model

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape).to(x.device)

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas


class DDIM:
    def __init__(self, args, segment_graph):
        self.device = args.device
        # self.timesteps = 100
        self.args = args
        self.skip = args.skip
        self.schedule = args.schedule
        self.mask = torch.from_numpy(segment_graph).to(self.device)
        self.model = Model(args).to(self.device)
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.criterion = nn.BCELoss(reduction='none')
        self.use_attr = args.use_attr
        
    def diffusion_config(self, n_steps):
        # betas is the schedule of the diffusion process
        betas = make_beta_schedule(schedule=self.schedule, n_timesteps=n_steps, start=0.0001, end=0.05)
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        return alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt

    def noise_estimation_loss(self, model, x_0, n_steps, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
        label = x_0
        batch_size, seq_len = x_0.shape[0], x_0.shape[1]

        # Select a random step for each example
        t = torch.randint(low=0, high=n_steps, size=(batch_size // 2 + 1,), device=self.device)  # [low, high)
        t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size]
        
        # prepend start token
        start = torch.zeros((batch_size, 1), dtype=torch.int64).to(self.device)
        log_seqs = torch.cat([start, x_0[:, :-1]+1], dim=-1)                                       # [B, T]
        
        x_0 = model.segment_embedding(x_0+1)                                                # (B, T) -> (B, T, C)
        log_seqs_emb = model.segment_embedding(log_seqs) 
        
        mean = extract(alphas_bar_sqrt, t, x_0).to(self.device)                             # [B, 1, 1]
        mean_ = mean.repeat(1, x_0.shape[1], 1)                                             # [B, T, 1]
        
        mean = mean_ * x_0
        var_sqrt = extract(one_minus_alphas_bar_sqrt, t, x_0).to(self.device)               # [B, 1, 1]
        var_sqrt_ = var_sqrt.repeat(1, x_0.shape[1], 1)
        var_sqrt = var_sqrt_

        e = torch.randn_like(x_0)
        x = mean + var_sqrt * e                 # x_t forward process

        log_feats, output = model(log_seqs_emb, x, t)                    
        pred_x_0 = (x - var_sqrt*output) / mean_ 
        
        pred = model.sample(pred_x_0, log_feats, log_seqs, mask=self.mask, train=True).squeeze()     
        mse_loss = F.mse_loss(output, e)    
        cross_loss = F.cross_entropy(pred.reshape(-1, self.mask.shape[0]-1), label.reshape(-1).long())
        
        loss = cross_loss
        mse_loss2 = F.mse_loss(pred_x_0, x_0)
        if self.schedule == 'linear':
            loss = loss + mse_loss2 * 0.001 
        elif self.schedule == 'quad':
            loss = loss + mse_loss2 * 0.01 
        loss = loss + mse_loss

        
        return loss, cross_loss, mse_loss, mse_loss2

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(self.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
        return a

    def p_xt(self, xt, noise, t, next_t, beta, eta=0):
        at = self.compute_alpha(beta, t.long())
        at_next = self.compute_alpha(beta, next_t.long())
        x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
        c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        eps = torch.randn(xt.shape, device=self.device)
        xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
        return xt_next

    def reverse_process(self, model, shape, n_steps, beta):
        #* for segment
        batch, seq_len = shape
        new_shape = [batch, seq_len, model.ch]
        # skip = n_steps // self.timesteps  # 5
        skip = self.skip
        eta = 0.0

        seq = range(0, n_steps, skip)
        seq_next = [-1] + list(seq[:-1])
        
        # prepend start token
        start = torch.zeros((batch, 1), dtype=torch.int64).to(self.device)                # [B, 1]
        
        x = torch.randn(new_shape).to(self.device)                                        # (B, T, 128)
        n = x.shape[0]
        result = [start]
        segment_embs = model.segment_embedding.weight    
        
        for l in range(seq_len):
            ims = []
            log_x = start  #* 已经加1
            log_x = log_x.long().to(self.device)                                   # (B, T) -> (B, T, C)
            log_seqs_emb = segment_embs[log_x, :]
            state = model.encode(log_seqs_emb)[:, -1]
            x_ = x[:, l]
        
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).long().to(self.device)
                next_t = (torch.ones(n) * j).long().to(self.device)
                with torch.no_grad():
                    pred_noise = model.step(state, x_, t)
                    x_ = self.p_xt(x_, pred_noise, t, next_t, beta, eta)
                    ims.append(x_)
            pred = model.sample(ims[-1], state, log_x, self.mask)                      # [B, N]
            _, indices = torch.topk(pred, 1, dim=-1)                                   # [B, 1]
            result.append(indices+1)
            start = torch.cat(result, dim=-1)
            
        return torch.cat(result, dim=-1)[:, 1:]
    
    def pretrain_mask2(self, model, pos_items, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, level):
        label = pos_items - 1       #* 已经加1
        batch_size = pos_items.shape[0]
        seq_len = pos_items.shape[1]
        t = torch.ones((batch_size,), device=self.device, dtype=torch.int64) * level
    
        # prepend start token
        start = torch.zeros((batch_size, 1), dtype=torch.int64).to(self.device)
        log_seqs = torch.cat([start, pos_items[:, :-1]], dim=-1)   
        pos_items_emb = model.segment_embedding(pos_items)
        log_seqs_emb = model.segment_embedding(log_seqs)
        
        mean = extract(alphas_bar_sqrt, t, pos_items_emb).to(self.device)                             # [B, 1, 1]
        mean_ = mean.repeat(1, pos_items_emb.shape[1], 1)                                             # [B, T, 1]
        mean = mean_ * pos_items_emb
        var_sqrt = extract(one_minus_alphas_bar_sqrt, t, pos_items_emb).to(self.device)               # [B, 1, 1]
        var_sqrt_ = var_sqrt.repeat(1, pos_items_emb.shape[1], 1)
        var_sqrt = var_sqrt_
        
        e = torch.randn_like(pos_items_emb)
        x = mean + var_sqrt * e                 # x_t forward process

        # Next Movement Prediction
        log_feats, output = model(log_seqs_emb, x, t)                    
        pred_x_0 = (x - var_sqrt*output) / mean_
        
        pred = model.sample(pred_x_0, log_feats, log_seqs, mask=self.mask, train=True).squeeze()     
        mse_loss_nmp = F.mse_loss(output, e)    
        cross_loss_nmp = F.cross_entropy(pred.reshape(-1, self.mask.shape[0]-1), label.reshape(-1).long())
        mse_loss2_nmp = F.mse_loss(pred_x_0, pos_items_emb)
        
        nmp_loss = cross_loss_nmp + mse_loss_nmp + mse_loss2_nmp
        
        return nmp_loss, cross_loss_nmp, mse_loss_nmp, mse_loss2_nmp
    