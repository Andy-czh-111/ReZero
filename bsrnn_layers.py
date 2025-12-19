import torch
import torch.nn as nn

def get_band_specs(fs=16000, nfft=512):
    """
    根据论文 IV-B 章节实现频带划分:
    - 10个 100 Hz
    - 12个 200 Hz
    - 8个 500 Hz
    - 剩余部分作为一个子带
    """
    freq_resolution = fs / nfft
    bands_hz = [100]*10 + [200]*12 + [500]*8
    
    band_width_bins = []
    current_freq = 0
    total_bins = nfft // 2 + 1
    
    for bh in bands_hz:
        # 计算该带宽对应的 bin 数量
        bw_bin = int(bh / freq_resolution)
        band_width_bins.append(bw_bin)
        current_freq += bh
        
    # 处理剩余频带
    used_bins = sum(band_width_bins)
    if used_bins < total_bins:
        band_width_bins.append(total_bins - used_bins)
        
    return torch.tensor(band_width_bins, dtype=torch.long)

class BandSplit(nn.Module):
    def __init__(self, channels=128, fs=16000, nfft=512, input_dim=2):
        """
        Args:
            channels: BSRNN 内部特征维度
            fs: 采样率
            nfft: FFT 点数
            input_dim: 输入特征的维度 (复数谱为2, 空间特征可能为1或麦克风对数)
        """
        super(BandSplit, self).__init__()
        # 动态获取频带划分
        self.band = get_band_specs(fs, nfft)
        
        for i in range(len(self.band)):
            # Input size = bandwidth * input_dim
            # e.g., complex spectrum: bw * 2 (Real+Imag)
            # e.g., IPD/ILD: bw * num_pairs (Passed as input_dim)
            input_size = int(self.band[i] * input_dim)
            setattr(self, f'norm{i+1}', nn.GroupNorm(1, input_size))
            setattr(self, f'fc{i+1}', nn.Linear(input_size, channels))

    def forward(self, x):
        # x: (B, input_dim, F, T) -> need permutation
        nb, input_dim, _, nt = x.shape
        hz_band = 0
        z_list = []
        
        # Permute to (B, T, F, input_dim) to slice frequency easily
        x = x.permute(0, 3, 2, 1) # (B, T, F, input_dim)
        
        for i in range(len(self.band)):
            bw = int(self.band[i])
            # Slice frequency band
            x_band = x[:, :, hz_band:hz_band+bw, :] # (B, T, bw, input_dim)
            # Flatten last two dims: bw * input_dim
            x_band = x_band.reshape(nb, nt, -1) # (B, T, bw*input_dim)
            
            norm = getattr(self, f'norm{i+1}')
            fc = getattr(self, f'fc{i+1}')
            
            # Norm requires (B, C, T) -> permute
            out = norm(x_band.permute(0, 2, 1)).permute(0, 2, 1)
            out = fc(out) # (B, T, C)
            z_list.append(out.unsqueeze(1))
            hz_band += bw
            
        z = torch.cat(z_list, dim=1) # (B, K, T, C)
        return z

class MaskDecoder(nn.Module):
    def __init__(self, channels=128, fs=16000, nfft=512):
        super(MaskDecoder, self).__init__()
        self.band = get_band_specs(fs, nfft)
        
        for i in range(len(self.band)):
            setattr(self, f'norm{i+1}', nn.GroupNorm(1, channels))
            setattr(self, f'fc1{i+1}', nn.Linear(channels, 4*channels))
            # Output is complex mask (Real+Imag) -> bw * 2
            # GLU input needs to be double -> bw * 4
            out_dim = int(self.band[i] * 4) 
            setattr(self, f'fc2{i+1}', nn.Linear(4*channels, out_dim))
            setattr(self, f'glu{i+1}', nn.GLU(dim=-1))

    def forward(self, x):
        # x: (B, K, T, C)
        nb, _, nt, _ = x.shape
        m_list = []
        for i in range(len(self.band)):
            x_band = x[:, i, :, :] # (B, T, C)
            norm = getattr(self, f'norm{i+1}')
            fc1 = getattr(self, f'fc1{i+1}')
            fc2 = getattr(self, f'fc2{i+1}')
            glu = getattr(self, f'glu{i+1}')
            
            out = norm(x_band.permute(0, 2, 1)).permute(0, 2, 1)
            out = torch.tanh(fc1(out))
            out = glu(fc2(out)) # (B, T, bw*2)
            m_list.append(out)
            
        m = torch.cat(m_list, dim=-1) # (B, T, F*2)
        m = m.reshape(nb, nt, -1, 2).permute(0, 2, 1, 3) # (B, F, T, 2)
        return m