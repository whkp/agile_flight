import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .VAD import LightweightViT

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class TemporalAttention(nn.Module):
    """时序注意力机制，用于处理轨迹序列内的长程依赖"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class ConditionalLayerNorm(nn.Module):
    """条件层归一化，根据时间步调整归一化参数"""
    def __init__(self, hidden_size, embedding_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale_shift = nn.Linear(embedding_size, hidden_size * 2)
        self.hidden_size = hidden_size
        
    def forward(self, x, cond):
        x = self.layer_norm(x)
        scale_shift = self.scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return x * (1 + scale) + shift

class DiffusionBlock(nn.Module):
    """扩散模型核心块，结合时间条件、图像条件和状态序列"""
    def __init__(self, state_dim, hidden_dim, time_embed_dim, img_cond_dim, dropout=0.1):
        super().__init__()
        
        # 状态编码器
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        
        # 条件层归一化
        self.norm1 = ConditionalLayerNorm(hidden_dim, time_embed_dim)
        self.norm2 = ConditionalLayerNorm(hidden_dim, time_embed_dim)
        self.norm3 = ConditionalLayerNorm(hidden_dim, time_embed_dim)
        
        # 图像条件嵌入
        self.img_proj = nn.Sequential(
            nn.Linear(img_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 时序注意力
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 残差连接中的dropout
        self.dropout = nn.Dropout(dropout)
        
        # 最终状态预测
        self.out = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, x, time_embed, img_cond):
        # x: [B, T, state_dim]
        # time_embed: [B, time_embed_dim]
        # img_cond: [B, img_cond_dim]
        
        # 编码状态
        h = self.state_encoder(x)
        
        # 图像条件
        img_features = self.img_proj(img_cond).unsqueeze(1).expand(-1, x.size(1), -1)
        h = h + img_features
        
        # 时序注意力 + 残差
        h_norm = self.norm1(h, time_embed)
        h = h + self.dropout(self.temporal_attn(h_norm))
        
        # 前馈网络 + 残差
        h_norm = self.norm2(h, time_embed)
        h = h + self.dropout(self.ff(h_norm))
        
        # 最终归一化
        h_norm = self.norm3(h, time_embed)
        
        # 输出预测噪声/状态
        return self.out(h_norm)

class DroneTrajectoryDiffusion(nn.Module):
    """无人机轨迹扩散模型 - 对轨迹生成进行降噪预测"""
    def __init__(
        self,
        state_dim=12,  # 位置(3), 速度(3), 姿态四元数(4), 角速度(2)
        sequence_length=64,
        hidden_dim=256,
        time_embed_dim=128,
        img_encoder_dim=256,
        depth=4,
        dropout=0.1,
        img_size=224,
        patch_size=16,
        in_channels=1
    ):
        super().__init__()
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 图像编码器 (复用现有ViT)
        self.img_encoder = LightweightViT(
            image_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=img_encoder_dim,
            depth=4,
            num_heads=4
        )
        
        # 扩散块
        self.diffusion_blocks = nn.ModuleList([
            DiffusionBlock(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                time_embed_dim=time_embed_dim,
                img_cond_dim=img_encoder_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        
    def forward(self, x, time_steps, depth_images):
        """
        参数:
            x: 带噪声的轨迹状态序列 [B, T, state_dim]
            time_steps: 时间步 [B]
            depth_images: 深度图像 [B, C, H, W]
        返回:
            预测的噪声 [B, T, state_dim]
        """
        # 时间嵌入
        time_embed = self.time_embed(time_steps)
        
        # 图像编码
        img_features = self.img_encoder(depth_images)
        
        # 应用扩散块
        for block in self.diffusion_blocks:
            x = block(x, time_embed, img_features)
            
        return x

class DiffusionTrainer:
    """扩散模型训练器"""
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, device="cuda"):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 预计算扩散过程的常数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """向干净轨迹添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def extract(self, a, t, x_shape):
        """提取适当的t索引，用于批量数据"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def p_losses(self, x_start, t, depth_images, noise=None):
        """计算扩散损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, depth_images)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @torch.no_grad()
    def p_sample(self, x, t, t_index, depth_images):
        """从噪声中采样一步"""
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        
        # 模型预测噪声
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, depth_images) / sqrt_one_minus_alphas_cumprod_t
        )
        
        # 只有t>0时才添加噪声
        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
        
        if t_index == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, depth_images):
        """完整的采样循环，从纯噪声生成轨迹"""
        device = self.device
        b = shape[0]
        
        # 从噪声开始
        img = torch.randn(shape, device=device)
        
        # 逐步降噪
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i, depth_images)
            
        return img
    
    @torch.no_grad()
    def sample(self, batch_size, sequence_length, depth_images):
        """生成样本轨迹"""
        return self.p_sample_loop(
            shape=(batch_size, sequence_length, self.model.state_dim),
            depth_images=depth_images
        )

class TrajectoryDataset(torch.utils.data.Dataset):
    """无人机轨迹数据集"""
    def __init__(self, trajectories, depth_images, sequence_length=64, transforms=None):
        """
        参数:
            trajectories: 轨迹状态序列 [num_trajectories, max_length, state_dim]
            depth_images: 对应的深度图像 [num_trajectories, channels, height, width]
            sequence_length: 使用的序列长度
            transforms: 图像变换
        """
        self.trajectories = trajectories
        self.depth_images = depth_images
        self.sequence_length = sequence_length
        self.transforms = transforms
        
        # 确保轨迹至少有sequence_length长度
        valid_indices = []
        for i, traj in enumerate(trajectories):
            # 假设轨迹是一个嵌套列表或numpy数组，需要检查其长度
            if len(traj) >= sequence_length:
                valid_indices.append(i)
        
        self.valid_indices = valid_indices
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 获取有效轨迹索引
        traj_idx = self.valid_indices[idx]
        
        # 获取轨迹和对应深度图像
        trajectory = self.trajectories[traj_idx]
        depth_image = self.depth_images[traj_idx]
        
        # 如果轨迹长度大于sequence_length，随机选择一个起点
        if len(trajectory) > self.sequence_length:
            start_idx = np.random.randint(0, len(trajectory) - self.sequence_length)
            trajectory = trajectory[start_idx:start_idx + self.sequence_length]
        else:
            # 否则从头开始
            trajectory = trajectory[:self.sequence_length]
        
        # 应用图像变换
        if self.transforms:
            depth_image = self.transforms(depth_image)
        
        # 转换为torch张量
        trajectory = torch.FloatTensor(trajectory)
        depth_image = torch.FloatTensor(depth_image)
        
        return trajectory, depth_image 