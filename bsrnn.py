import torch
import torch.nn as nn
from .bsrnn_layers import BandSplit, MaskDecoder, get_band_specs

class SubbandDEG(nn.Module):
    """
    [论文对应] Fig. 7: Subband-specific Distance Embedding Generator
    为每个子带生成独立的距离嵌入，无 Batch Normalization。
    """
    def __init__(self, num_subbands, embedding_dim=128):
        super().__init__()
        # 为每个子带实例化独立的 MLP
        self.degs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, embedding_dim)
            ) for _ in range(num_subbands)
        ])

    def forward(self, d):
        """
        Args:
            d: (Batch, 1) or (Batch,)
        Returns:
            out: (Batch, K, Embedding_Dim)
        """
        if d.dim() == 1: d = d.unsqueeze(-1)
        
        embeddings = []
        for deg in self.degs:
            e = deg(d) # (B, Embd_Dim)
            embeddings.append(e.unsqueeze(1))
            
        return torch.cat(embeddings, dim=1)

class ReZeroBSRNN(nn.Module):
    def __init__(self, 
                 num_channel=128, 
                 num_layer=6, 
                 condition_type='angle', # 'angle' 或 'distance'
                 spatial_dim=6,          # 麦克风对数 (Pairs)
                 condition_dim=32,       # 角度特征维度 (2*P) 或 距离特征维度
                 fs=16000, 
                 nfft=512):
        super(ReZeroBSRNN, self).__init__()
        
        self.num_layer = num_layer
        self.condition_type = condition_type
        
        # 1. 获取频带配置
        self.band_specs = get_band_specs(fs, nfft)
        num_subbands = len(self.band_specs)
        
        # 2. 多流特征 BandSplit
        # Stream 1: 频谱 (Real+Imag)
        self.bs_spec = BandSplit(channels=num_channel, fs=fs, nfft=nfft, input_dim=2)
        
        # Stream 2: 空间特征 (IPD 或 ILD) -> 通道数为麦克风对数
        self.bs_spatial = BandSplit(channels=num_channel, fs=fs, nfft=nfft, input_dim=spatial_dim)
        
        # Stream 3: 条件分支
        if condition_type == 'distance':
            # D-ReZero: 子带 DEG
            self.subband_deg = SubbandDEG(num_subbands, embedding_dim=num_channel)
        elif condition_type == 'angle':
            # A-ReZero: 接收子带聚合后的特征 (B, K, T, Cond_Dim)
            # 映射到 num_channel 以便融合
            self.cond_proj = nn.Linear(condition_dim, num_channel)

        # 3. 骨干网络 (Backbone)
        for i in range(self.num_layer):
            # Time-LSTM: 输入已融合，维度为 num_channel
            setattr(self, f'norm_t{i+1}', nn.GroupNorm(1, num_channel))
            setattr(self, f'lstm_t{i+1}', nn.LSTM(num_channel, 2*num_channel, batch_first=True))
            setattr(self, f'fc_t{i+1}', nn.Linear(2*num_channel, num_channel))

            # Band-LSTM
            setattr(self, f'norm_k{i+1}', nn.GroupNorm(1, num_channel))
            setattr(self, f'lstm_k{i+1}', nn.LSTM(num_channel, 2*num_channel, batch_first=True, bidirectional=True))
            setattr(self, f'fc_k{i+1}', nn.Linear(4*num_channel, num_channel))

        # 4. 解码器
        self.mask_decoder = MaskDecoder(channels=num_channel, fs=fs, nfft=nfft)

    def forward(self, x_audio_complex, spatial_feature, condition_input):
        """
        x_audio_complex: (B, F, T)
        spatial_feature: (B, Pairs, F, T) -> IPD 或 ILD
        condition_input: 
            - distance: (B, 1)
            - angle: (B, K, T, Cond_Dim) [由 RegionFeatureAggregator 生成]
        """
        # (B, F, T) -> (B, 2, F, T)
        x_real = torch.view_as_real(x_audio_complex).permute(0, 3, 1, 2)
        nb, _, nf, nt = x_real.shape
        
        # --- Stream 1: Spectrum ---
        z_spec = self.bs_spec(x_real) # (B, K, T, C)
        B, K, T, C = z_spec.shape
        
        # --- Stream 2: Spatial ---
        # spatial_feature: (B, P, F, T)
        z_spatial = self.bs_spatial(spatial_feature) # (B, K, T, C)
        
        # --- Stream 3: Condition ---
        z_cond = 0
        if self.condition_type == 'distance':
            # deg_out: (B, K, C)
            deg_out = self.subband_deg(condition_input)
            # 扩展到时间维度: (B, K, 1, C) -> (B, K, T, C)
            z_cond = deg_out.unsqueeze(2).expand(-1, -1, T, -1)
            
        elif self.condition_type == 'angle':
            # condition_input: (B, K, T, Cond_Dim)
            # 投影: (B, K, T, C)
            z_cond = self.cond_proj(condition_input)
            
        # --- 特征融合 (Fusion) ---
        z = z_spec + z_spatial + z_cond
        skip = z

        # --- BSRNN Processing ---
        for i in range(self.num_layer):
            # Time Block
            out_t = getattr(self, f'norm_t{i+1}')(skip.permute(0,3,1,2).reshape(B,C,K,T)).reshape(B,C,K,T).permute(0,2,3,1)
            out_t = out_t.reshape(B*K, T, C)
            lstm_out, _ = getattr(self, f'lstm_t{i+1}')(out_t)
            out_t = getattr(self, f'fc_t{i+1}')(lstm_out)
            out_t = out_t.reshape(B, K, T, C)
            skip = skip + out_t
            
            # Frequency Block
            out_k = getattr(self, f'norm_k{i+1}')(skip.permute(0,3,1,2).reshape(B,C,K,T)).reshape(B,C,K,T).permute(0,2,3,1)
            out_k = out_k.permute(0, 2, 1, 3).contiguous().reshape(B*T, K, C)
            lstm_out, _ = getattr(self, f'lstm_k{i+1}')(out_k)
            out_k = getattr(self, f'fc_k{i+1}')(lstm_out)
            out_k = out_k.reshape(B, T, K, C).permute(0, 2, 1, 3)
            skip = skip + out_k
            
        # --- Mask ---
        m = self.mask_decoder(skip)
        m_complex = torch.view_as_complex(m.contiguous())
        
        return m_complex * x_audio_complex