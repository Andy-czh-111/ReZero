import torch
import numpy as np
from model.system import ReZeroSystem

def create_circular_mic_array(radius, num_mics):
    """
    创建一个位于 XY 平面的圆形麦克风阵列 (模拟论文配置)
    参数:
        radius: 半径 (米)
        num_mics: 麦克风数量
    返回:
        mic_locs: (N, 3) 的 numpy 数组
    """
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    mic_locs = []
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0
        mic_locs.append([x, y, z])
    return np.array(mic_locs)

def run_example():
    # ---------------------------------------------------------
    # 1. 全局配置 (Configuration)
    # ---------------------------------------------------------
    print("=== ReZero System Implementation Example ===")
    
    # 音频参数
    batch_size = 2
    fs = 16000
    duration_sec = 2.0
    samples = int(fs * duration_sec)
    
    # 麦克风阵列配置 (参考论文: 8麦圆阵, 直径5cm -> 半径0.025m )
    n_mic = 8
    mic_radius = 0.025 
    mic_locs = create_circular_mic_array(mic_radius, n_mic)
    print(f"[*] Microphone Array: {n_mic} mics, Radius {mic_radius}m")

    # 模拟输入音频 (Batch, Samples, Channels)
    # 假设输入是归一化的波形
    dummy_audio = torch.randn(batch_size, samples, n_mic)
    print(f"[*] Input Audio Shape: {dummy_audio.shape}")

    # ---------------------------------------------------------
    # 2. 场景 A: A-ReZero (仅角度提取)
    # ---------------------------------------------------------
    print("\n--- [Scenario 1] A-ReZero: Angle Extraction ---")
    
    # 初始化系统 (Angle 模式)
    # bsrnn_channels 设为 32 仅为了演示速度，实际论文可能更大
    system_angle = ReZeroSystem(
        mic_locations=mic_locs, 
        task_type='angle',
        bsrnn_channels=32, 
        P_dim=16  # 区域特征维度
    )
    
    # 定义查询区域: 目标在 0度 到 90度 之间
    # 注意: 当前实现通常对整个 Batch 广播同一个查询区域，或支持 Batch 维度的标量
    region_params_angle = {
        'azi_low': torch.tensor(0.0),          # 0 rad
        'azi_high': torch.tensor(np.pi / 2),   # 90 deg -> rad
        'ele_low': torch.tensor(np.pi / 2),    # 90 deg (水平面)
        'ele_high': torch.tensor(np.pi / 2),
        'dist_low': torch.tensor(0.0),         # 角度模式下忽略距离
        'dist_high': torch.tensor(5.0)
    }
    
    # 前向传播
    with torch.no_grad():
        # 输出通常是复数谱 (Batch, F, T)
        est_stft_angle = system_angle(dummy_audio, region_params_angle)
    
    print(f"Input: {dummy_audio.shape} -> Output STFT: {est_stft_angle.shape}")
    print("Success for A-ReZero!")


    # ---------------------------------------------------------
    # 3. 场景 B: D-ReZero (仅距离提取)
    # ---------------------------------------------------------
    print("\n--- [Scenario 2] D-ReZero: Distance Extraction ---")
    
    # 初始化系统 (Distance 模式)
    system_dist = ReZeroSystem(
        mic_locations=mic_locs, 
        task_type='distance',
        bsrnn_channels=32
    )
    
    # 定义查询区域: 距离 0.5m 到 1.0m 之间 (球形区域 [cite: 198])
    region_params_dist = {
        'azi_low': torch.tensor(0.0),
        'azi_high': torch.tensor(2*np.pi),
        'ele_low': torch.tensor(0.0),
        'ele_high': torch.tensor(np.pi),
        'dist_low': torch.tensor(0.5),   # 0.5m
        'dist_high': torch.tensor(1.0)   # 1.0m
    }
    
    with torch.no_grad():
        est_stft_dist = system_dist(dummy_audio, region_params_dist)
        
    print(f"Input: {dummy_audio.shape} -> Output STFT: {est_stft_dist.shape}")
    print("Success for D-ReZero!")

    # ---------------------------------------------------------
    # 4. 场景 C: Cascade (圆锥区域提取 A -> D)
    # ---------------------------------------------------------
    print("\n--- [Scenario 3] Cascade: Conical Region (Angle -> Distance) ---")
    
    # 初始化系统 (Cascade 模式: 内部包含 A-ReZero 和 D-ReZero 两个子网)
    system_cascade = ReZeroSystem(
        mic_locations=mic_locs, 
        task_type='cascade',
        bsrnn_channels=32
    )
    
    # 定义查询区域: 特定角度范围 + 特定距离范围 (圆锥 [cite: 198])
    region_params_cone = {
        'azi_low': torch.tensor(-np.pi/4),     # -45 deg
        'azi_high': torch.tensor(np.pi/4),     # +45 deg
        'ele_low': torch.tensor(np.pi/2),
        'ele_high': torch.tensor(np.pi/2),
        'dist_low': torch.tensor(0.0),
        'dist_high': torch.tensor(1.5)         # 1.5m
    }
    
    with torch.no_grad():
        est_stft_cone = system_cascade(dummy_audio, region_params_cone)
        
    print(f"Input: {dummy_audio.shape} -> Output STFT: {est_stft_cone.shape}")
    print("Success for Cascade ReZero!")
    
    # ---------------------------------------------------------
    # 5. ISTFT (可选: 转换回波形)
    # ---------------------------------------------------------
    print("\n[*] Converting STFT back to Waveform (Example)...")
    # 简单的 ISTFT 演示 (需要与 STFT 参数匹配)
    win_len = 512
    hop_len = 128 # 0.25 shift ratio
    window = torch.hann_window(win_len)
    
    # 取 Cascade 的输出进行逆变换
    # Output STFT: (B, F, T) complex
    # torch.istft 需要 (B, F, T) complex 或 (B, F, T, 2) real
    # 注意: torch.istft 的输入维度通常需要 Batch, Freq, Time
    try:
        est_wav = torch.istft(
            est_stft_cone, 
            n_fft=512, 
            hop_length=hop_len, 
            win_length=win_len, 
            window=window,
            length=samples # 确保长度对齐
        )
        print(f"Recovered Waveform Shape: {est_wav.shape}") # 应该是 (B, Samples)
    except Exception as e:
        print(f"ISTFT Skipped or Failed: {e}")
        print("Note: Ensure PyTorch version supports complex ISTFT directly.")

if __name__ == "__main__":
    run_example()