import torch
import torch.nn as nn
from .bsrnn_layers import get_band_specs

class RegionFeatureAggregator(nn.Module):
    """
    [论文对应] Fig. 5 (e): Subband-wise RNN-Loop Aggregation
    对每个子带分别执行 RNN-Loop 聚合。
    """
    def __init__(self, input_dim_nf, hidden_dim_P, fs=16000, nfft=512):
        super().__init__()
        self.P = hidden_dim_P
        self.band_specs = get_band_specs(fs, nfft)
        
        # 共享的 LSTM，用于处理所有子带的特征序列
        # 输入维度: 子带带宽(Bin数) -> 这一步需要投影到统一维度吗？
        # 论文中 Fig 5 "BW_k -> LSTM"。
        # 由于不同子带带宽不同 (BW_k)，我们不能直接用同一个LSTM处理原始 Bin。
        # 方案：先将每个子带特征线性映射到统一维度 (如 P)，再进 LSTM。
        
        self.subband_projs = nn.ModuleList([
            nn.Linear(int(bw), hidden_dim_P) for bw in self.band_specs
        ])
        
        # RNN Loop LSTM (Shared weights across bands is a common practice for parameter efficiency)
        self.lstm = nn.LSTM(input_size=hidden_dim_P, hidden_size=hidden_dim_P, batch_first=True)

    def forward(self, V):
        """
        Input: V (Batch, N_views, Freq, Time)
        Output: V_agg (Batch, K_subbands, Time, 2*P) -> 对齐 BSRNN 输入
        """
        nb, N, nf, nt = V.shape
        K = len(self.band_specs)
        
        # 1. Split Frequency into Subbands
        # V: (B, N, F, T) -> (B, T, N, F)
        V = V.permute(0, 3, 1, 2) 
        
        curr_freq = 0
        agg_features = []
        
        for k, bw in enumerate(self.band_specs):
            bw = int(bw)
            # Slice subband: (B, T, N, BW_k)
            v_band = V[..., curr_freq : curr_freq + bw]
            curr_freq += bw
            
            # Project to hidden dim: (B, T, N, P)
            v_proj = self.subband_projs[k](v_band)
            
            # Merge Batch and Time for efficient RNN processing
            # (B*T, N, P)
            v_flat = v_proj.reshape(nb * nt, N, -1)
            
            # RNN Loop Logic
            # Append first view: (B*T, N+1, P)
            v_first = v_flat[:, 0:1, :]
            v_loop = torch.cat([v_flat, v_first], dim=1)
            
            # LSTM Forward
            out, _ = self.lstm(v_loop) # (B*T, N+1, P)
            
            # Take last two steps: (B*T, 2, P) -> (B*T, 2P)
            v_last_two = out[:, -2:, :].reshape(nb * nt, -1)
            
            # Reshape back: (B, T, 2P)
            v_agg_k = v_last_two.reshape(nb, nt, -1)
            agg_features.append(v_agg_k.unsqueeze(1)) # (B, 1, T, 2P)
            
        # Concat all subbands: (B, K, T, 2P)
        V_agg = torch.cat(agg_features, dim=1)
        
        return V_agg