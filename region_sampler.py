import torch
import torch.nn as nn
import numpy as np

class RegionSampler(nn.Module):
    """
    实现论文 D. Region feature sampling
    """
    def __init__(self, sampling_mode='fixed_number', num_views_N=8, interval_delta_theta_deg=None):
        super().__init__()
        self.sampling_mode = sampling_mode
        self.N = num_views_N
        if interval_delta_theta_deg is not None:
            self.delta_theta = np.deg2rad(interval_delta_theta_deg)

    def forward(self, azi_low, azi_high, ele_low, ele_high, dist_low, dist_high):
        """
        输入: 标量或单元素的 tensor，表示区域边界 (单位: 弧度/米)
        输出: 
            sampled_azimuths: (N,)
            sampled_elevations: (N,)
            sampled_distances: (2,)
        """
        device = azi_low.device if isinstance(azi_low, torch.Tensor) else 'cpu'
        
        # 1. 角度采样 (Azimuth)
        # 简化: Elevation 取中心
        center_elevation = (ele_low + ele_high) / 2.0
        
        if self.sampling_mode == 'fixed_number':
            if self.N == 1:
                vals = [(azi_low + azi_high) / 2.0]
            else:
                vals = np.linspace(float(azi_low), float(azi_high), self.N)
            sampled_azimuths = torch.tensor(vals, dtype=torch.float32, device=device)
            
        elif self.sampling_mode == 'fixed_interval':
            vals = np.arange(float(azi_low), float(azi_high) + 1e-6, self.delta_theta)
            if len(vals) == 0:
                vals = [(azi_low + azi_high) / 2.0]
            sampled_azimuths = torch.tensor(vals, dtype=torch.float32, device=device)

        sampled_elevations = center_elevation.expand_as(sampled_azimuths)
        
        # 2. 距离采样 (仅取边界)
        sampled_distances = torch.tensor([dist_low, dist_high], dtype=torch.float32, device=device)
        
        return sampled_azimuths, sampled_elevations, sampled_distances