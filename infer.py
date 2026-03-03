import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
from turn import FNO2d  # 导入模型类

# ----------------------------
# 配置参数
# ----------------------------
vel_path = "CurveVel_A_03.T.bin"  # 波速场文件路径
model_path = "best_model_epoch_999.pth"  # 模型权重路径
save_dir = "prediction_visualization"  # 可视化结果保存目录
src_x = 31  # 震源水平位置（可选：21,31,41,51）
time_steps_to_show = [0, 20, 40, 60, 80]  # 要展示的时间步
os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ----------------------------
# 1. 定义设备
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")


# ----------------------------
# 2. 波速场处理与模型输入准备
# ----------------------------
def process_velocity(vel_path):
    """处理波速场并生成所有输入特征"""
    # 读取并重塑波速场
    vel_data = np.fromfile(vel_path, dtype=np.float32).reshape(70, 70).T  # 转置匹配训练格式
    
    # 归一化（1500-4500m/s → [0,1]）
    vel_min, vel_max = 1500.0, 4500.0
    vel_norm = (vel_data - vel_min) / (vel_max - vel_min)
    vel_real = vel_data  # 保留真实波速值（用于可视化）
    
    # 计算频域特征
    vel_fft = fft2(vel_norm)
    vel_fft_shifted = fftshift(vel_fft)
    vel_fft_real = np.real(vel_fft_shifted)
    vel_fft_imag = np.imag(vel_fft_shifted)
    
    # 生成震源掩码
    src_mask = np.zeros((70, 70), dtype=np.float32)
    src_mask[0, src_x] = 1.0  # 震源深度固定在第0行
    
    # 生成坐标特征
    dx, dz = 25.0, 25.0
    nx, nz = 70, 70
    x_coord = np.linspace(0, (nx-1)*dx, nx) / ((nx-1)*dx)
    z_coord = np.linspace(0, (nz-1)*dz, nz) / ((nz-1)*dz)
    x_grid = np.tile(x_coord, (nz, 1))
    z_grid = np.tile(z_coord[:, None], (1, nx))
    
    # 拼接特征并转换为张量
    vel_norm = vel_norm[..., None]
    vel_fft_real = vel_fft_real[..., None]
    vel_fft_imag = vel_fft_imag[..., None]
    src_mask = src_mask[..., None]
    x_grid = x_grid[..., None]
    z_grid = z_grid[..., None]
    
    input_features = np.concatenate(
        [vel_norm, vel_fft_real, vel_fft_imag, src_mask, x_grid, z_grid],
        axis=-1
    )
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)  # (1,70,70,6)
    
    return input_tensor, vel_real, src_mask[..., 0]  # 返回输入张量、真实波速场、震源掩码


# ----------------------------
# 3. 模型预测
# ----------------------------
def predict_wavefield(input_tensor, model_path):
    """加载模型并预测波场"""
    # 初始化模型
    model = FNO2d(modes1=12, modes2=12, width=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 预测
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        wavefield_pred = model(input_tensor).cpu().numpy()[0]  # (98,70,70)
    
    return wavefield_pred


# ----------------------------
# 4. 可视化函数
# ----------------------------
def plot_velocity_field(vel_real, src_mask, save_path):
    """可视化波速场并标记震源位置"""
    plt.figure(figsize=(8, 7))
    # 波速场（灰度图）
    plt.imshow(vel_real, cmap='gray', origin='lower', vmin=1500, vmax=4500)
    # 标记震源（黄色星标）
    src_pos = np.where(src_mask == 1)
    plt.scatter(src_pos[1], src_pos[0], marker='*', c='yellow', s=200, edgecolors='black', label='震源')
    plt.colorbar(label='波速 (m/s)')
    plt.title('地下介质波速场分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存波速场图：{save_path}")


def plot_predicted_wavefields(wavefield_pred, src_x, time_steps, save_path):
    """可视化多个时间步的预测波场"""
    n_timesteps = len(time_steps)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(5*n_timesteps, 6))
    if n_timesteps == 1:
        axes = [axes]  # 处理单图情况
    
    # 统一颜色范围（避免不同时间步颜色标尺不一致）
    all_vals = wavefield_pred.flatten()
    vmax = np.percentile(np.abs(all_vals), 99.9)
    vmin = -vmax
    
    for i, t in enumerate(time_steps):
        ax = axes[i]
        # 绘制波场（地震波色标）
        im = ax.imshow(wavefield_pred[t], cmap='seismic', origin='lower', vmin=vmin, vmax=vmax)
        # 标记震源
        ax.scatter(src_x, 0, marker='*', c='yellow', s=150, edgecolors='black', label='震源')
        ax.set_title(f'预测波场（时间步 t={t}）')
        ax.axis('off')
        if i == 0:
            ax.legend()
    
    # 共享颜色条
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('波场振幅')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存多时间步波场图：{save_path}")


def plot_vel_overlay_wavefield(vel_real, wavefield_pred, src_x, time_steps, save_path):
    """波速场与波场叠加可视化"""
    n_timesteps = len(time_steps)
    fig, axes = plt.subplots(1, n_timesteps, figsize=(5*n_timesteps, 6))
    if n_timesteps == 1:
        axes = [axes]
    
    # 波速场范围与波场颜色范围
    vel_vmin, vel_vmax = 1500, 4500
    all_vals = wavefield_pred.flatten()
    wave_vmax = np.percentile(np.abs(all_vals), 99.9)
    wave_vmin = -wave_vmax
    
    for i, t in enumerate(time_steps):
        ax = axes[i]
        # 底图：波速场（灰度）
        ax.imshow(vel_real, cmap='gray', origin='lower', vmin=vel_vmin, vmax=vel_vmax, alpha=0.7)
        # 叠加波场（彩色透明）
        im = ax.imshow(wavefield_pred[t], cmap='seismic', origin='lower', 
                      vmin=wave_vmin, vmax=wave_vmax, alpha=0.5)
        # 标记震源
        ax.scatter(0, src_x, marker='*', c='yellow', s=150, edgecolors='black')
        ax.set_title(f'波速场+波场（t={t}）')
        ax.axis('off')
    
    # 颜色条
    cbar_vel = fig.colorbar(plt.cm.ScalarMappable(cmap='gray'), ax=axes, 
                           orientation='horizontal', pad=0.05, aspect=50)
    cbar_vel.set_label('波速 (m/s)')
    cbar_wave = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.01, aspect=50)
    cbar_wave.set_label('波场振幅')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存波速叠加波场图：{save_path}")


# ----------------------------
# 主流程：处理→预测→可视化
# ----------------------------
if __name__ == "__main__":
    # 1. 处理波速场，生成模型输入
    input_tensor, vel_real, src_mask = process_velocity(vel_path)
    
    # 2. 预测波场
    wavefield_pred = predict_wavefield(input_tensor, model_path)
    print(f"预测完成，波场形状：{wavefield_pred.shape}（时间步×高度×宽度）")
    
    # 3. 可视化并保存结果
    # 3.1 波速场展示
    plot_velocity_field(
        vel_real, 
        src_mask, 
        os.path.join(save_dir, "velocity_field.png")
    )
    
    # 3.2 多时间步预测波场展示
    plot_predicted_wavefields(
        wavefield_pred, 
        src_x, 
        time_steps_to_show, 
        os.path.join(save_dir, "predicted_wavefields.png")
    )
    
    # 3.3 波速场与波场叠加展示
    plot_vel_overlay_wavefield(
        vel_real, 
        wavefield_pred, 
        src_x, 
        time_steps_to_show, 
        os.path.join(save_dir, "vel_overlay_wavefield.png")
    )
    
    print(f"所有可视化结果已保存至：{save_dir}")