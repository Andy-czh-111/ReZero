import torch
import torch.nn as nn
import numpy as np
from .feature_extraction import CalculateSpatialFeatures, CalculateDirectionFeature
from .feature_aggregation import RegionFeatureAggregator
from .region_sampler import RegionSampler
from .bsrnn import ReZeroBSRNN

class ReZeroSystem(nn.Module):
    def __init__(self, mic_locations, win_len=512, nfft=512, bsrnn_channels=64, P_dim=16, task_type='cascade'):
        super().__init__()

        self.task_type = task_type.lower()
        num_mics = mic_locations.shape[0]
        # Calculate number of pairs for 'MM' mode (all pairs)
        self.num_pairs = num_mics * (num_mics - 1) // 2

        # 1. Feature Extractors
        self.spatial_extractor = CalculateSpatialFeatures(win_len, 0.25, nfft, ch_mode='MM')

        allowed_tasks = ['angle', 'distance', 'cascade']
        if self.task_type not in allowed_tasks:
            raise ValueError(f"task_type must be one of {allowed_tasks}, but got {self.task_type}")

        if self.task_type in ['angle', 'cascade']:
            self.direction_extractor = CalculateDirectionFeature(mic_locations, nfft, ch_mode='MM')
            self.sampler = RegionSampler(sampling_mode='fixed_number', num_views_N=8)
            nf = nfft // 2 + 1
            # 修正：显式传入 nfft，确保与系统配置一致
            self.aggregator = RegionFeatureAggregator(input_dim_nf=nf, hidden_dim_P=P_dim, nfft=nfft)
        
        if self.task_type in ['distance', 'cascade']:
            if not hasattr(self, 'sampler'):
                self.sampler = RegionSampler(sampling_mode='fixed_number', num_views_N=8)

        # --- Stage 1: Angle Separation (A-ReZero) ---
        if self.task_type in ['angle', 'cascade']:
            self.separator_angle = ReZeroBSRNN(
                num_channel=bsrnn_channels, 
                num_layer=6, 
                condition_type='angle',
                spatial_dim=self.num_pairs,    
                condition_dim=2 * P_dim,       
                nfft=nfft # 传递 nfft
            )

        # --- Stage 2: Distance Separation (D-ReZero) ---
        if self.task_type in ['distance', 'cascade']:
            self.separator_dist = ReZeroBSRNN(
                num_channel=bsrnn_channels, 
                num_layer=6, 
                condition_type='distance',
                spatial_dim=self.num_pairs,   
                condition_dim=0,             
                nfft=nfft # 传递 nfft
            )

    def forward(self, audio_multichannel, region_params):
        """
        audio_multichannel: (B, Samples, Channels)
        region_params: dict containing tensors 'azi_low', 'azi_high', etc.
        """
        # A. 提取 STFT 和 空间特征
        ipd, ild, stft_complex = self.spatial_extractor(audio_multichannel)
        ref_stft = stft_complex[..., 0] 

        nb = audio_multichannel.shape[0]
        nf = ref_stft.shape[1]
        nt = ref_stft.shape[2]
        
        # Reshape IPD/ILD (B*Pairs -> B, Pairs)
        ipd = ipd.reshape(nb, -1, nf, nt)
        ild = ild.reshape(nb, -1, nf, nt)

        # B. 区域采样
        s_azi, s_ele, s_dist = self.sampler(
            region_params['azi_low'], region_params['azi_high'],
            region_params['ele_low'], region_params['ele_high'],
            region_params['dist_low'], region_params['dist_high']
        )

        est_angle = None
        
        # === 分支 A: 角度分离 ===
        if self.task_type in ['angle', 'cascade']:
            # flattened IPD for direction extractor if needed, or pass reshaped depending on implementation
            # CalculateDirectionFeature expects (B*P, F, T) usually, let's verify
            # Looking at previous code, it reshapes inside. Passing (B*P, F, T) is safer if it expects flat batch.
            # But here ipd is (B, P, F, T). 
            # Check CalculateDirectionFeature.forward: 
            #   observed_ipd: (B*P, F, T) -> obs_ipd = observed_ipd.view(nbatch, num_pairs, nf, nt)
            # So passing flat IPD is correct.
            
            V = self.direction_extractor(ipd.reshape(-1, nf, nt), s_azi, s_ele) 
            V_agg = self.aggregator(V)
            est_angle = self.separator_angle(ref_stft, spatial_feature=ipd, condition_input=V_agg)

            if self.task_type == 'angle':
                return est_angle

        # === 分支 B: 距离分离 ===
        if self.task_type in ['distance', 'cascade']:
            # D-ReZero 使用 ILD 和 距离标量
            input_stft = est_angle if self.task_type == 'cascade' else ref_stft
            
            # --- 修复部分 START ---
            # s_dist 是 (2,) 张量 [low, high]。
            # 我们需要提取距离上限(high)作为查询条件，并将其扩展到 Batch 大小。
            dist_high = s_dist[1]       # Scalar tensor
            d_input = dist_high.expand(nb) # (B,)
            # --- 修复部分 END ---
            
            est_final = self.separator_dist(input_stft, spatial_feature=ild, condition_input=d_input)
            return est_final
        
        return None