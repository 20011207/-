import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import label

# ----------------------------
# 1. 数据加载类
# ----------------------------
class SeismicDataset(Dataset):
    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            velocity = f['inputs/velocity'][:].astype(np.float32)
            vel_fft_real = f['inputs/velocity_fft_real'][:].astype(np.float32)
            vel_fft_imag = f['inputs/velocity_fft_imag'][:].astype(np.float32)
            source_mask = f['inputs/source_mask'][:].astype(np.float32)
            x_coord = f['coords/x'][:].astype(np.float32)
            z_coord = f['coords/z'][:].astype(np.float32)
            wavefield = f['labels/wavefield'][:].astype(np.float32)
            
            if indices is not None:
                velocity = velocity[indices]
                vel_fft_real = vel_fft_real[indices]
                vel_fft_imag = vel_fft_imag[indices]
                source_mask = source_mask[indices]
                wavefield = wavefield[indices]
            
            # 维度转置，确保与模型输入匹配
            velocity = np.transpose(velocity, (0, 2, 1))
            vel_fft_real = np.transpose(vel_fft_real, (0, 2, 1))
            vel_fft_imag = np.transpose(vel_fft_imag, (0, 2, 1))
            source_mask = np.transpose(source_mask, (0, 2, 1))
            x_coord = np.transpose(x_coord)
            z_coord = np.transpose(z_coord)
            
            # 扩展坐标维度以匹配样本数量
            N = velocity.shape[0]
            x_coord = np.tile(x_coord[None, ..., None], (N, 1, 1, 1))
            z_coord = np.tile(z_coord[None, ..., None], (N, 1, 1, 1))
            
            # 增加通道维度
            velocity = velocity[..., None]
            vel_fft_real = vel_fft_real[..., None]
            vel_fft_imag = vel_fft_imag[..., None]
            source_mask = source_mask[..., None]
            
            # 拼接所有输入特征
            self.x = torch.tensor(
                np.concatenate([velocity, vel_fft_real, vel_fft_imag, source_mask, x_coord, z_coord], axis=-1),
                dtype=torch.float32
            )
            self.y = torch.tensor(wavefield, dtype=torch.float32)
            self.total_samples = self.x.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ----------------------------
# 2. FNO模型定义
# ----------------------------
class SpectralConv2d(nn.Module):
    """傅里叶卷积层"""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, modes1, modes2, dtype=torch.float32)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, modes1, modes2, dtype=torch.float32)
        )

    def forward(self, x):
        batch_size, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # 转为(batch, channels, h, w)
        
        # 傅里叶变换
        x_ft = torch.fft.rfft2(x, dim=[2, 3])
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # 分离实部虚部
        
        # 傅里叶域卷积
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes1, :self.modes2, :] = self.conv_complex(
            x_ft[:, :, :self.modes1, :self.modes2, :], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2, :] = self.conv_complex(
            x_ft[:, :, -self.modes1:, :self.modes2, :], self.weights2
        )
        
        # 逆傅里叶变换
        out_ft_complex = torch.complex(out_ft[..., 0], out_ft[..., 1])
        x = torch.fft.irfft2(out_ft_complex, s=[h, w], dim=[2, 3])
        return x.permute(0, 2, 3, 1)  # 转回(batch, h, w, channels)

    def conv_complex(self, x_ft, weights):
        x_real = x_ft[..., 0]
        x_imag = x_ft[..., 1]
        
        out_real = torch.einsum('bihw,oihw->bohw', x_real, weights)
        out_imag = torch.einsum('bihw,oihw->bohw', x_imag, weights)
        return torch.stack([out_real, out_imag], dim=-1)


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(6, self.width)  # 输入6通道

        # 傅里叶卷积层
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        # 普通卷积层（残差连接用）
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # 输出层
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 98)  # 输出98个时间步
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 输入映射
        x = self.fc0(x)

        # 傅里叶卷积+残差连接
        x1 = self.conv0(x)
        x2 = self.w0(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = nn.functional.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = nn.functional.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = nn.functional.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = nn.functional.relu(x1 + x2)

        # 输出映射
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # 输出形状(batch, T, Z, X)


# ----------------------------
# 3. 损失函数与工具函数
# ----------------------------
def generate_time_weights(num_time_steps=98, decay_type="linear", min_weight=0.1):
    """生成时间衰减权重"""
    assert min_weight > 0, "最小权重必须大于0"
    assert decay_type in ["linear", "exponential"], "仅支持linear或exponential"
    
    if decay_type == "linear":
        weights = np.linspace(min_weight, 1.0, num_time_steps)
    else:
        base = (1 / min_weight) ** (1 / (num_time_steps - 1))
        weights = np.array([base ** i for i in range(num_time_steps)])
    
    return torch.tensor(weights / weights.sum(), dtype=torch.float32)


def compute_huygens_loss(y_pred, dx=25.0, dz=25.0):
    """
    惠更斯约束损失：确保波前扩散符合子波源规律
    兼容所有PyTorch版本（分两步计算多维度最大值）
    """
    batch, T, Z, X = y_pred.shape
    huygens_loss = 0.0

    # 每10帧取一个关键时间步验证波前扩散
    for t in range(5, T-5, 10):
        # 当前时刻波场（取绝对值）
        wave_t = torch.abs(y_pred[:, t, :, :])  # shape: (batch, Z, X)
        
        # 分两步计算最大值（兼容低版本PyTorch）
        max_z, _ = torch.max(wave_t, dim=1, keepdim=True)  # 先对Z轴求max
        max_val, _ = torch.max(max_z, dim=2, keepdim=True)  # 再对X轴求max
        
        # 波前阈值（取当前帧振幅的80%作为波前点）
        wave_threshold = 0.8 * max_val
        wavefront_mask = (wave_t >= wave_threshold).float()  # 波前掩码

        # 下一时刻波场
        wave_t1 = torch.abs(y_pred[:, t+1, :, :])
        # 同样分两步计算下一时刻最大值
        max_z1, _ = torch.max(wave_t1, dim=1, keepdim=True)
        max_val1, _ = torch.max(max_z1, dim=2, keepdim=True)
        wavefront_t1 = (wave_t1 >= 0.8 * max_val1).float()  # 下一时刻波前掩码

        # 计算理论扩散范围（3x3邻域，当前波前点的子波源应扩散到的区域）
        kernel = torch.ones(1, 1, 3, 3, device=y_pred.device)
        wavefront_diffuse = torch.nn.functional.conv2d(
            wavefront_mask.unsqueeze(1), kernel, padding=1
        ).squeeze(1) > 0  # 扩散范围内为True

        # 无效波前点：下一时刻波前不在理论扩散范围内的点
        invalid_wavefront = wavefront_t1 * (~wavefront_diffuse).float()
        huygens_loss += torch.mean(invalid_wavefront, dim=(1,2)).mean()  # 平均到batch

    return huygens_loss / (T//10)  # 归一化到每个时间步


def compute_physical_loss(y_pred, x, device, time_weights):
    """波动方程物理损失（基于波动方程残差）"""
    batch, T, Z, X = y_pred.shape
    
    velocity = x[..., 0].squeeze(-1)
    c_sq = velocity **2  # 波速平方
    
    # 计算拉普拉斯算子（空间二阶导数）
    u_xx = y_pred[..., 1:-1, 2:] - 2 * y_pred[..., 1:-1, 1:-1] + y_pred[..., 1:-1, :-2]
    u_zz = y_pred[..., 2:, 1:-1] - 2 * y_pred[..., 1:-1, 1:-1] + y_pred[..., :-2, 1:-1]
    laplacian = u_xx + u_zz
    
    # 波速平方乘以拉普拉斯
    c_sq_cropped = c_sq[..., 1:-1, 1:-1]
    c_sq_laplacian = c_sq_cropped.unsqueeze(1) * laplacian
    
    # 计算时间二阶导数
    u_tt = y_pred[:, 2:, 1:-1, 1:-1] - 2 * y_pred[:, 1:-1, 1:-1, 1:-1] + y_pred[:, :-2, 1:-1, 1:-1]
    c_sq_laplacian_cropped = c_sq_laplacian[:, 1:-1, :, :]
    
    # 波动方程残差（u_tt = c²∇²u + f，这里f=0）
    residual = u_tt - (c_sq_laplacian_cropped)
    cropped_weights = time_weights[1:-1].to(device)
    cropped_weights = cropped_weights / cropped_weights.sum()  # 归一化权重
    
    # 加权残差损失
    residual_sq_per_time = (residual** 2).mean(dim=[2, 3])
    weighted_phys_loss = (residual_sq_per_time * cropped_weights).sum(dim=1).mean()
    
    return weighted_phys_loss


def compute_total_physical_loss(y_pred, x, device, time_weights, lambda_huygens=0.05):
    """总物理损失 = 波动方程损失 + 惠更斯约束损失"""
    phys_loss = compute_physical_loss(y_pred, x, device, time_weights)
    huygens_loss = compute_huygens_loss(y_pred)
    return phys_loss + lambda_huygens * huygens_loss


def plot_training_curves(train_losses, val_losses, train_pred, val_pred, train_phys, val_phys, save_path):
    """绘制训练损失曲线"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='训练总损失')
    plt.plot(val_losses, label='验证总损失')
    plt.xlabel('Epoch')
    plt.ylabel('总损失')
    plt.title('总损失曲线（预测损失+物理损失）')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(train_pred, label='训练加权预测损失')
    plt.plot(val_pred, label='验证加权预测损失')
    plt.plot(train_phys, label='训练物理损失（含惠更斯）')
    plt.plot(val_phys, label='验证物理损失（含惠更斯）')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('损失分量对比')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curves.png'))
    plt.close()


def plot_wavefield_comparison(model, val_loader, device, save_path, epoch, vel_min=1500.0, vel_max=4500.0):
    """波场对比可视化（含波速场、真实/预测波场、误差）"""
    model.eval()
    with torch.no_grad():
        x, y_true = next(iter(val_loader))
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        
        # 随机选一个样本
        sample_idx = np.random.randint(0, x.size(0))
        # 提取并反归一化波速场
        vel_norm = x[sample_idx, ..., 0].cpu().numpy()
        vel_real = vel_norm * (vel_max - vel_min) + vel_min
        # 提取波场和震源信息
        y_true_sample = y_true[sample_idx].cpu().numpy()
        y_pred_sample = y_pred[sample_idx].cpu().numpy()
        source_mask = x[sample_idx, ..., 3].cpu().numpy()
        
        # 提取震源位置
        labeled_regions, num_regions = label(source_mask > 0)
        assert num_regions == 1, f"样本应含1个震源，实际{num_regions}个"
        src_coords = np.argwhere(labeled_regions == 1)
        src_z, src_x = int(np.mean(src_coords[:, 0])), int(np.mean(src_coords[:, 1]))
        
        # 统一颜色范围
        vel_vmin, vel_vmax = 1500.0, 4500.0
        all_wave_values = np.concatenate([y_true_sample[:90].flatten(), y_pred_sample[:90].flatten()])
        wave_vmax = np.percentile(np.abs(all_wave_values), 99.9)
        wave_vmin = -wave_vmax
        error_vmax = wave_vmax / 5
        error_vmin = -error_vmax
        
        # 绘制前90帧
        time_steps = range(90)
        rows = 9
        cols_per_group = 4
        groups_per_row = 10
        fig, axes = plt.subplots(
            rows, groups_per_row * cols_per_group,
            figsize=(groups_per_row * 4, rows * 3)
        )
        fig.suptitle(f'样本{sample_idx} - 波速场+波场对比 (Epoch {epoch})', fontsize=20, y=0.98)
        
        for i, t in enumerate(time_steps):
            row = i // groups_per_row
            group_idx = i % groups_per_row
            col = group_idx * cols_per_group
            
            # 波速场子图
            ax_vel = axes[row, col]
            im_vel = ax_vel.imshow(vel_real, cmap='viridis', vmin=vel_vmin, vmax=vel_vmax, origin='lower')
            ax_vel.scatter(src_x, src_z, marker='*', c='red', s=20, label='震源')
            if i == 0:
                ax_vel.set_title(f'波速场\n(t={t})', fontsize=8, pad=8)
                ax_vel.legend(fontsize=6, loc='upper right')
            else:
                ax_vel.set_title(f't={t}', fontsize=8)
            ax_vel.axis('off')
            
            # 真实波场子图
            ax_true = axes[row, col+1]
            im_true = ax_true.imshow(y_true_sample[t], cmap='seismic', vmin=wave_vmin, vmax=wave_vmax, origin='lower')
            ax_true.scatter(src_x, src_z, marker='*', c='red', s=20)
            ax_true.set_title('真实波场', fontsize=8)
            ax_true.axis('off')
            
            # 预测波场子图
            ax_pred = axes[row, col+2]
            im_pred = ax_pred.imshow(y_pred_sample[t], cmap='seismic', vmin=wave_vmin, vmax=wave_vmax, origin='lower')
            ax_pred.scatter(src_x, src_z, marker='*', c='red', s=20)
            ax_pred.set_title('预测波场', fontsize=8)
            ax_pred.axis('off')
            
            # 误差子图
            ax_err = axes[row, col+3]
            im_err = ax_err.imshow(y_true_sample[t]-y_pred_sample[t], cmap='coolwarm', vmin=error_vmin, vmax=error_vmax, origin='lower')
            ax_err.set_title('误差', fontsize=8)
            ax_err.axis('off')
        
        # 添加颜色条
        cbar_vel = fig.colorbar(im_vel, cax=fig.add_axes([0.12, 0.02, 0.2, 0.015]), orientation='horizontal')
        cbar_vel.set_label('波速（m/s）', labelpad=5)
        
        cbar_wave = fig.colorbar(im_true, cax=fig.add_axes([0.4, 0.02, 0.2, 0.015]), orientation='horizontal')
        cbar_wave.set_label('波场振幅', labelpad=5)
        
        cbar_err = fig.colorbar(im_err, cax=fig.add_axes([0.68, 0.02, 0.2, 0.015]), orientation='horizontal')
        cbar_err.set_label('误差振幅', labelpad=5)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(save_path, f'wavefield_vel_comparison_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 已保存含波速场的对比图：wavefield_vel_comparison_epoch_{epoch}.png")


# ----------------------------
# 4. 主训练函数
# ----------------------------
def train_model(
    h5_path, save_dir, epochs=200, batch_size=16, lr=1e-4,
    resume_path=None, decay_type="linear", min_time_weight=0.1, lambda_phys=0.1,
    vel_min=1500.0, vel_max=4500.0
):
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据并划分训练/验证集
    with h5py.File(h5_path, 'r') as f:
        total_samples = f.attrs['total_samples']
    indices = np.arange(total_samples)
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_dataset = SeismicDataset(h5_path, train_indices)
    val_dataset = SeismicDataset(h5_path, val_indices)
    
    # 数据加载器（Windows系统num_workers=0）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 4,
        pin_memory=True,
        drop_last=True
    )
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = FNO2d(modes1=12, modes2=12, width=32).to(device)
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 损失记录
    train_losses, val_losses = [], []
    train_pred_losses, val_pred_losses = [], []
    train_phys_losses, val_phys_losses = [], []
    
    # 加载预训练模型（如果有）
    if resume_path and os.path.exists(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"✅ 加载预训练模型：{resume_path}")
        
        # 解析起始epoch
        if "epoch" in resume_path:
            start_epoch = int(resume_path.split("epoch_")[-1].split(".")[0])
        
        # 加载历史损失
        loss_log_path = os.path.join(save_dir, "loss_log.npz")
        if os.path.exists(loss_log_path):
            loss_log = np.load(loss_log_path)
            train_losses = loss_log["train_losses"].tolist()
            val_losses = loss_log["val_losses"].tolist()
            train_pred_losses = loss_log["train_pred"].tolist()
            val_pred_losses = loss_log["val_pred"].tolist()
            train_phys_losses = loss_log["train_phys"].tolist()
            val_phys_losses = loss_log["val_phys"].tolist()
            best_val_loss = min(val_losses) if val_losses else float('inf')
            print(f"✅ 加载历史损失，最佳验证损失：{best_val_loss:.6f}")
    
    # 优化器与 scheduler
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 加载优化器状态（如果有）
    optimizer_path = os.path.join(save_dir, "optimizer.pth")
    if resume_path and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        print(f"✅ 加载优化器状态")
    
    # 生成时间权重
    time_weights = generate_time_weights(
        num_time_steps=98,
        decay_type=decay_type,
        min_weight=min_time_weight
    )
    print(f"✅ 时间权重：{decay_type}衰减，最小权重={min_time_weight}")
    print(f"✅ 波速范围：{vel_min}~{vel_max} m/s（用于可视化反归一化）")
    
    # 开始训练
    for epoch in range(start_epoch, epochs):
        model.train()
        train_total = 0.0
        train_pred = 0.0
        train_phys = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 模型预测
            y_pred = model(x)
            
            # 计算损失（含惠更斯约束）
            mse_per_time = criterion(y_pred, y).mean(dim=[2, 3])
            weighted_pred_loss = (mse_per_time * time_weights.to(device)).sum(dim=1).mean()
            phys_loss = compute_total_physical_loss(y_pred, x, device, time_weights, lambda_huygens=0.05)
            total_loss = weighted_pred_loss + lambda_phys * phys_loss
            
            # 反向传播与优化
            total_loss.backward()
            optimizer.step()
            
            # 累加损失
            batch_size = x.size(0)
            train_total += total_loss.item() * batch_size
            train_pred += weighted_pred_loss.item() * batch_size
            train_phys += phys_loss.item() * batch_size
        
        # 计算训练集平均损失
        avg_train_total = train_total / len(train_loader.dataset)
        avg_train_pred = train_pred / len(train_loader.dataset)
        avg_train_phys = train_phys / len(train_loader.dataset)
        train_losses.append(avg_train_total)
        train_pred_losses.append(avg_train_pred)
        train_phys_losses.append(avg_train_phys)
        
        # 验证阶段
        model.eval()
        val_total = 0.0
        val_pred = 0.0
        val_phys = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                
                # 计算验证损失（与训练一致，含惠更斯约束）
                mse_per_time = criterion(y_pred, y).mean(dim=[2, 3])
                weighted_pred_loss = (mse_per_time * time_weights.to(device)).sum(dim=1).mean()
                phys_loss = compute_total_physical_loss(y_pred, x, device, time_weights, lambda_huygens=0.05)
                total_loss = weighted_pred_loss + lambda_phys * phys_loss
                
                # 累加验证损失
                batch_size = x.size(0)
                val_total += total_loss.item() * batch_size
                val_pred += weighted_pred_loss.item() * batch_size
                val_phys += phys_loss.item() * batch_size
        
        # 计算验证集平均损失
        avg_val_total = val_total / len(val_loader.dataset)
        avg_val_pred = val_pred / len(val_loader.dataset)
        avg_val_phys = val_phys / len(val_loader.dataset)
        val_losses.append(avg_val_total)
        val_pred_losses.append(avg_val_pred)
        val_phys_losses.append(avg_val_phys)
        
        # 打印损失信息
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"训练总损失: {avg_train_total:.6f} (预测: {avg_train_pred:.6f}, 物理: {avg_train_phys:.6f})")
        print(f"验证总损失: {avg_val_total:.6f} (预测: {avg_val_pred:.6f}, 物理: {avg_val_phys:.6f})")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 保存最佳模型
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            
            np.savez(
                os.path.join(save_dir, "loss_log.npz"),
                train_losses=train_losses,
                val_losses=val_losses,
                train_pred=train_pred_losses,
                val_pred=val_pred_losses,
                train_phys=train_phys_losses,
                val_phys=val_phys_losses
            )
            print(f"✅ 保存最佳模型：{model_path}")
        
        # 每20轮可视化
        if (epoch + 1) % 20 == 0:
            plot_wavefield_comparison(
                model=model,
                val_loader=val_loader,
                device=device,
                save_path=save_dir,
                epoch=epoch+1,
                vel_min=vel_min,
                vel_max=vel_max
            )
        
        # 调整学习率
        scheduler.step(avg_val_total)
    
    # 训练结束后绘制损失曲线
    plot_training_curves(
        train_losses, val_losses,
        train_pred_losses, val_pred_losses,
        train_phys_losses, val_phys_losses,
        save_path=save_dir
    )
    
    print(f"\n训练完成！最佳验证总损失: {best_val_loss:.6f}")


# ----------------------------
# 5. 启动训练
# ----------------------------
if __name__ == "__main__":
    # 配置参数
    h5_file_path = "out"
    save_directory = "infer-results"
    num_epochs = 10000
    batch_size = 128  # 根据GPU显存调整
    learning_rate = 1e-3
    resume_model_path = 'best_model_epoch_999.pth'  # 增量训练路径

    # 损失与波速参数
    time_decay_type = "linear"
    min_time_weight = 0.1
    lambda_phys = 0.5  # 物理损失权重
    vel_min = 1500.0
    vel_max = 4500.0
    
    # 打印配置信息
    print("="*60)
    print("===== FNO地震波场预测模型（含惠更斯约束） =====")
    print(f"数据文件: {h5_file_path}")
    print(f"保存目录: {save_directory}")
    print(f"训练配置: 轮数={num_epochs}, 批次={batch_size}, 学习率={learning_rate}")
    print(f"损失配置: 时间衰减={time_decay_type}, 物理权重={lambda_phys}, 惠更斯权重=0.05")
    print(f"波速配置: {vel_min}~{vel_max} m/s")
    print(f"增量训练: {'启用' if resume_model_path else '禁用'}")
    if resume_model_path:
        print(f"预训练模型: {resume_model_path}")
    print("="*60)
    
    # 启动训练
    train_model(
        h5_path=h5_file_path,
        save_dir=save_directory,
        epochs=num_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        resume_path=resume_model_path,
        decay_type=time_decay_type,
        min_time_weight=min_time_weight,
        lambda_phys=lambda_phys,
        vel_min=vel_min,
        vel_max=vel_max
    )