import torch
import torch.nn as nn
import numpy as np
from utils.stft import STFT

class CalculateSpatialFeatures(nn.Module):
    def __init__(self, win_len, win_shift_ratio, nfft, ch_mode='MM'):
        super().__init__()
        self.stft = STFT(win_len, win_shift_ratio, nfft)
        self.ch_mode = ch_mode
        self.epsilon = 1e-8

    def forward(self, signal):
        # 1. STFT: (B, F, T, C) -> permute to (B, C, F, T)
        stft_complex = self.stft(signal)
        stft_data = stft_complex.permute(0, 3, 1, 2) 
        nb, nch, nf, nt = stft_data.shape
        
        # 2. 生成麦克风对
        pairs_p1, pairs_p2 = [], []
        if self.ch_mode == 'M': # (0,1), (0,2)...
            for i in range(1, nch):
                pairs_p1.append(stft_data[:, 0])
                pairs_p2.append(stft_data[:, i])
        elif self.ch_mode == 'MM': # All pairs
            for i in range(nch - 1):
                for j in range(i + 1, nch):
                    pairs_p1.append(stft_data[:, i])
                    pairs_p2.append(stft_data[:, j])
        
        # (B * NumPairs, F, T)
        Y_p1 = torch.cat(pairs_p1, dim=0)
        Y_p2 = torch.cat(pairs_p2, dim=0)
        
        # 3. IPD
        angle_p1 = torch.angle(Y_p1)
        angle_p2 = torch.angle(Y_p2)
        ipd = torch.atan2(torch.sin(angle_p1 - angle_p2), torch.cos(angle_p1 - angle_p2))
        
        # 4. ILD
        mag_p1 = torch.abs(Y_p1)
        mag_p2 = torch.abs(Y_p2)
        ild = 20 * torch.log10(mag_p1 + self.epsilon) - 20 * torch.log10(mag_p2 + self.epsilon)
        
        return ipd, ild, stft_complex

class CalculateDirectionFeature(nn.Module):
    """
    计算方向特征 V(theta, phi)
    """
    def __init__(self, mic_locations, nfft, fs=16000, v=343.0, ch_mode='MM'):
        super().__init__()
        self.v = v
        # 注册频率仓，不作为参数更新
        self.register_buffer('freq_bins', torch.linspace(0, fs/2, nfft//2 + 1))
        
        # 预计算麦克风对向量
        nch = mic_locations.shape[0]
        pair_vecs = []
        if ch_mode == 'M':
            for i in range(1, nch):
                pair_vecs.append(mic_locations[i] - mic_locations[0])
        elif ch_mode == 'MM':
            for i in range(nch - 1):
                for j in range(i + 1, nch):
                    pair_vecs.append(mic_locations[j] - mic_locations[i])
        self.register_buffer('pair_vectors', torch.from_numpy(np.array(pair_vecs)).float())

    def forward(self, observed_ipd, query_azi, query_ele):
        """
        observed_ipd: (B*P, F, T)
        query_azi/ele: (N_views,)
        """
        num_pairs = self.pair_vectors.shape[0]
        total_batch, nf, nt = observed_ipd.shape
        nbatch = total_batch // num_pairs
        
        # (B, P, F, T)
        obs_ipd = observed_ipd.view(nbatch, num_pairs, nf, nt)
        
        # 计算 TDOA: (P, 3) @ (3, N) -> (P, N)
        r_x = torch.sin(query_ele) * torch.cos(query_azi)
        r_y = torch.sin(query_ele) * torch.sin(query_azi)
        r_z = torch.cos(query_ele)
        r = torch.stack([r_x, r_y, r_z], dim=0).to(self.pair_vectors.device)
        
        tdoa = torch.matmul(self.pair_vectors, r) / self.v
        
        # 计算 TPD: (P, N, F)
        # 2*pi * tdoa(P,N,1) * f(1,1,F)
        tpd = 2 * torch.pi * tdoa.unsqueeze(-1) * self.freq_bins.view(1, 1, -1)
        
        # 计算相似度 V = sum_p cos(IPD - TPD)
        # Expand dims for broadcasting:
        # IPD: (B, P, 1, F, T)
        # TPD: (1, P, N, F, 1)
        similarity = torch.cos(obs_ipd.unsqueeze(2) - tpd.unsqueeze(0).unsqueeze(-1))
        
        # Sum over pairs -> (B, N, F, T)
        V = torch.sum(similarity, dim=1)
        return V

class DistanceEmbeddingGenerator(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embedding_dim), nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim), nn.Tanh()
        )
    def forward(self, d):
        return self.net(d)


class SubbandDEG(nn.Module):
    """
    论文 IV-C & Fig. 7: Subband-specific DEG
    为每个子带生成独立的距离嵌入
    """
    def __init__(self, num_subbands, embedding_dim=16):
        super().__init__()
        self.degs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim), nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim) # 论文提及 D-ReZero 中移除了 BN
            ) for _ in range(num_subbands)
        ])

    def forward(self, d):
        # d: (Batch, 1) or (Batch,)
        if d.dim() == 1: d = d.unsqueeze(-1)
        
        # 对每个子带计算嵌入
        embeddings = []
        for deg in self.degs:
            # out: (B, P)
            e = deg(d)
            embeddings.append(e.unsqueeze(1)) # (B, 1, P)
            
        # 拼接为 (B, K, P)
        return torch.cat(embeddings, dim=1)