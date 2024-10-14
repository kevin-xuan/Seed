import numpy as np
import torch
import torch.nn as nn
from math import radians, cos, sin, asin, sqrt, pi
import os

def readEmbedFile(embedFile):
    input = open(embedFile, 'r')
    lines = []
    for line in input:
        lines.append(line) 

    embeddings_dict = {}
    embeddings = []
    for lineId in range(1, len(lines)): 
        splits = lines[lineId].split(' ')
        # embedId赋值
        embedId = int(splits[0])
        embedValue = splits[1:]
        new_embedValue = [float(x) for x in embedValue]
        embeddings_dict[embedId] = new_embedValue

    for i in sorted(embeddings_dict): 
        embeddings.append(embeddings_dict[i])
    
    embeddings = torch.from_numpy(np.array(embeddings)).float()
        
    return embeddings

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32,
                              num_channels=in_channels,
                              eps=1e-6,
                              affine=True)
    
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate, dev, use_rope=False):
        super(MultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)
        self.use_rope = use_rope

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, attn_mask, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)
        if not self.use_rope:
            Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)  
            K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
            V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

            abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
            abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

            attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
            attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))

            attn_weights = attn_weights / (K_.shape[-1] ** 0.5)
            attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
            paddings = torch.ones(attn_weights.shape) *  (-2**32+1)
            paddings = paddings.to(self.dev)
            attn_weights = torch.where(attn_mask, paddings, attn_weights)

            attn_weights = self.softmax(attn_weights)
            attn_weights = self.dropout(attn_weights)

            outputs = attn_weights.matmul(V_)
            outputs += attn_weights.matmul(abs_pos_V_)

            outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)
        else:
            Q_ = torch.stack(torch.split(Q, self.head_size, dim=2), dim=1)  
            K_ = torch.stack(torch.split(K, self.head_size, dim=2), dim=1)
            V_ = torch.stack(torch.split(V, self.head_size, dim=2), dim=1)
            
            Q_, K_ = RoPE(Q_, K_)
            attn_weights = Q_.matmul(torch.transpose(K_, -1, -2))                       # [B, H, L, L]
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).expand(attn_weights.shape[0], self.head_num, -1, -1)
            paddings = torch.ones(attn_weights.shape) *  (-2**32+1)
            paddings = paddings.to(self.dev)
            attn_weights = torch.where(attn_mask, paddings, attn_weights)
            
            attn_weights = self.softmax(attn_weights)
            attn_weights = self.dropout(attn_weights)
            
            outputs = attn_weights.matmul(V_) 
            
            outputs = outputs.view(Q.shape[0], Q.shape[1], -1)                          # [B, H, L, D] -> [B, L, H*D]
        
        return outputs


class Model(nn.Module):

    def __init__(self, args, rid_gps=None):
        super().__init__()
        ch = args.channel_size  # 128
        self.dev = args.device
        
        # attention
        self.dropout = 0.0
        self.num_blocks = args.num_blocks               # default: 2
        self.num_heads = args.num_heads                 # default: 2

        self.ch = ch  # 128
        self.use_attr = args.use_attr
        self.condition = args.use_cond
        self.pred = args.pred
        self.input_dim = self.ch
        self.use_rope = False
        
        # segment embedding
        self.segment_num = args.segment_num
        file_dir, file_name = os.path.split(args.pretrained_emb)
        prefix, suffix = file_name.split('.')
        new_file_name = os.path.join(file_dir, prefix + '_' + str(self.ch) + '.' + suffix)
        self.pretreained_emb = readEmbedFile(new_file_name)
        if args.use_pre:
            self.segment_embedding = nn.Embedding(self.segment_num+2, self.ch, padding_idx=self.segment_num+1)
            if args.use_emb:
                self.segment_embedding.weight.data[:-1, :] = self.pretreained_emb
                self.segment_embedding.weight.requires_grad = False
        else:
            self.segment_embedding = nn.Embedding(self.segment_num+1, self.ch)
        self.ch = self.input_dim
        self.temb_ch = self.ch * 4  # 512
    
        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.ch)
        ])  # (128, 512) & (512, 512) & (512, 128)
        if not self.use_rope:
            self.abs_pos_K_emb = torch.nn.Embedding(args.max_len, self.ch)
            self.abs_pos_V_emb = torch.nn.Embedding(args.max_len, self.ch)
        
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(self.ch, eps=1e-8)

        for _ in range(self.num_blocks):  # 2
            new_attn_layernorm = nn.LayerNorm(self.ch, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = MultiHeadAttention(self.ch, self.num_heads, self.dropout, self.dev, self.use_rope)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(self.ch, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.ch, self.dropout)
            self.forward_layers.append(new_fwd_layer)
            
        if self.condition:
            self.decoder = nn.Linear(self.ch * 3, self.ch)  # default: start, end, feats, noise_label
        else:
            self.decoder = nn.Linear(self.ch * 2, self.ch)
        if self.pred == 'proj':
            self.proj = nn.Linear(self.ch, self.segment_num)   # for vector score
        elif self.pred == 'score':
            self.proj = nn.Linear(self.ch * 2, 1)   # for vector score

    def gcn_embeddings(self, tra_matrix, anchor_idx=None):
        item_embs, support = self.gcn(self.segment_embedding, tra_matrix, anchor_idx) 
        return item_embs, support
    
    def seq2feats(self, log_seqs, mode, gps):
        seqs = log_seqs
        if not self.use_rope:
            positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) 
            positions = torch.LongTensor(positions).to(self.dev)
            abs_pos_K = self.abs_pos_K_emb(positions)
            abs_pos_V = self.abs_pos_V_emb(positions)
        else:
            abs_pos_K = None
            abs_pos_V = None

        tl = seqs.shape[1]
        if mode == 'pre-training':
            attention_mask = torch.zeros((tl, tl), dtype=torch.bool, device=self.dev)
        else:
            attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev)) 
        
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs, attention_mask, abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)

        return log_feats
    
    def get_diffusion_input(self, seqs_nxt, t, extra_embed=None):

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)           # (B, 128)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)                     # (B, 512)
        if extra_embed is not None:
            temb = temb + extra_embed
        temb = nonlinearity(temb)
        temb = self.temb.dense[2](temb)                     # (B, 128)
        if seqs_nxt.ndim >= 3:
            temb = temb.unsqueeze(1) 
        seqs_nxt = seqs_nxt + temb
        
        
        return seqs_nxt
    
    def encode(self, log_seqs, mode='training', gps=None):
        
        log_feats = self.seq2feats(log_seqs, mode, gps)  
        
        return log_feats
    
    def forward(self, log_seqs, seqs_nxt, t, extra_embed=None, mode='training', gps=None):
        
        log_feats = self.encode(log_seqs, mode, gps)           # regarded as condition for diffusion model
        
        return log_feats, self.step(log_feats, seqs_nxt, t, extra_embed)
    
    def step(self, log_feats, seqs_nxt, t, extra_embed=None):
        
        x_t = self.get_diffusion_input(seqs_nxt, t)                            # (B, T, 128) or (B, 128)
        
        if self.condition:
            if x_t.ndim >=3:
                extra_embed = extra_embed.unsqueeze(1)                         # (B, 128*2) -> (B, 1, 128*2)
                extra_embed = extra_embed.expand(-1, x_t.shape[1], -1)         # (B, T, 128*2)
            fin_logits = self.decoder(torch.cat([extra_embed, log_feats, x_t], dim=-1))     # (B, 128*4)
        else:
            fin_logits = self.decoder(torch.cat([log_feats, x_t], dim=-1))     # (B, 128)
            
        return fin_logits
    
    def sample(self, x_recover, log_feats, log_seqs=None, mask=None, train=False):
        if self.pred == 'mse':
            batch = x_recover.shape[0]
            x_recover = x_recover.reshape(-1, 1, self.ch)                    # [B*T, 1, D] or [B, 1, D] 
            logits = -torch.mean(torch.sqrt(torch.square(x_recover - self.segment_embedding.weight[1:-1].unsqueeze(0))), dim=-1)                                                          # [B, N, D] -> [B, N]
            logits = logits.reshape(batch, -1, self.segment_num).squeeze()   
            logits = torch.sigmoid(logits)          
        elif self.pred == 'cosine':    
            batch = x_recover.shape[0]
            x_recover = x_recover.reshape(-1, 1, self.ch)                    # [B*T, 1, D] or [B, 1, D]        
            logits = torch.cosine_similarity(x_recover, self.segment_embedding.weight[1:-1].unsqueeze(0), dim=-1)                                                          # [B, N]  
            logits = logits.reshape(batch, -1, self.segment_num).squeeze()     
            logits = torch.sigmoid(logits)             
        elif self.pred == 'mul':
            logits = torch.sigmoid(x_recover.matmul(self.segment_embedding.weight[1:-1, :].transpose(0,1)))
        elif self.pred == 'score':
            batch = x_recover.shape[0]
            x_recover = x_recover.reshape(-1, 1, self.ch).repeat(1, self.segment_num, 1)                                         # [B*T, N, D] or [B, N, D]
            item_emb = self.segment_embedding.weight[1:-1, :].unsqueeze(0).repeat(x_recover.shape[0], 1, 1)                          # [B, N, D]
            logits = torch.sigmoid(self.proj(torch.cat([x_recover, item_emb], dim=-1)))                    # [B, N, 1]
            logits = logits.reshape(batch, -1, self.segment_num).squeeze()
        else:
            # vector score
            logits =torch.sigmoid(self.proj(x_recover))  
        
        if mask is not None:
            bias = mask[log_seqs, :].to(self.dev)                                                   # [B, T, N]     
            if not train:
                logits = logits + bias[:, -1, :][:,1:]                                                                                  # [B, N]
            else:
                logits = logits + bias[..., 1:]                                                                                         # [B, T, N]
        
        return logits        
