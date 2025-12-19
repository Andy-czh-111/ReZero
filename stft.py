import torch
import torch.nn as nn
import numpy as np

class STFT(nn.Module):
    def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
        """
        初始化短时傅里叶变换(STFT)的核心参数配置
        
        参数:
            win_len: int, 窗口长度，表示每次FFT计算的样本点数，影响频率分辨率
            win_shift_ratio: float, 帧移比率，表示相邻窗口之间的重叠比例，通常为0.5(50%重叠)
            nfft: int, FFT点数，决定频域分辨率，通常大于或等于win_len
            win: str, 窗函数类型，默认为'hann'(汉宁窗)，用于减少频谱泄漏
        """
        # 调用父类初始化方法，确保继承的属性和方法被正确初始化
        super(STFT, self).__init__()
        
        # 保存窗口长度参数，这是STFT中每次FFT处理的信号段长度
        self.win_len = win_len
        
        # 保存帧移比率参数，用于计算相邻窗口之间的实际样本点数偏移
        self.win_shift_ratio = win_shift_ratio
        
        # 保存FFT点数参数，决定频域变换后的点数
        self.nfft = nfft
        
        # 保存窗函数类型参数，指定将使用的窗函数种类
        self.win = win

    def forward(self, signal):
        # 获取信号维度信息  输入：signal - 形状为[batch_size, n_samples, n_channels]
        nsample = signal.shape[-2]  # 每个通道的采样点数
        nch = signal.shape[-1]      # 通道数量
        win_shift = int(self.win_len * self.win_shift_ratio)  # 窗口移动步长
        # nf = int(self.nfft / 2) + 1  # 频点数量（仅保留正频率部分）
        # nb = signal.shape[0]        # 批次大小
        
        # # 计算时间帧数
        # nt = np.floor((nsample - self.win_len) / win_shift + 1).astype(int)
        
        # # 初始化复数类型的输出张量，形状为[batch_size, n_freq, n_time, n_channel]
        # stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)
        
        # 根据配置选择窗函数
        if self.win == 'hann':
            # 使用汉宁窗，减少频谱泄漏
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        else:
            # 默认使用矩形窗
            window = torch.ones(self.win_len, device=signal.device)
        
        # # 逐通道计算短时傅里叶变换
        # for ch_idx in range(0, nch, 1):
        #     # 对每个通道调用PyTorch内置的STFT函数
        #     # center=False表示不进行中心填充，保持信号时间对齐
        #     # return_complex=True表示直接返回复数结果
        #     stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, 
        #                                    win_length=self.win_len, window=window, center=False, 
        #                                    normalized=False, return_complex=True)
        
        stft_list = []
        for ch_idx in range(nch):
            # return_complex=True 在较新版本的 PyTorch 中是推荐的
            s = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift, 
                           win_length=self.win_len, window=window, center=False, 
                           return_complex=True)
            stft_list.append(s.unsqueeze(-1))
        
        # Concat along channel dim
        stft = torch.cat(stft_list, dim=-1)

        # 返回时频域表示结果
        return stft