import torch
import torch.nn as nn
import torch.nn.functional as F


# 轻量级ViT的组件实现
class PatchEmbedding(nn.Module):
    """将图像分割为patches并进行线性投影"""
    def __init__(self, image_size=224, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # 使用卷积层进行patch嵌入
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, f"输入图像大小({H}*{W})与模型期望大小({self.image_size}*{self.image_size})不匹配"
        
        # 应用卷积提取patch特征: (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        # 重塑为序列: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # 转置为 (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # 为每个patch和class token创建可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
    
    def forward(self, x):
        # 添加位置编码到输入
        return x + self.pos_embedding[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成QKV并重塑为多头形式
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力并重塑回原始维度
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )
    
    def forward(self, x):
        # 自注意力块 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP块 + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x

class LightweightViT(nn.Module):
    """轻量级Vision Transformer"""
    def __init__(self, 
                 image_size=224, 
                 patch_size=16, 
                 in_channels=1, 
                 embed_dim=256, 
                 depth=4, 
                 num_heads=4, 
                 mlp_ratio=4.0, 
                 dropout=0.1):
        super().__init__()
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_embed = LearnablePositionalEncoding(num_patches, embed_dim)
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # 最终层归一化
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        
        # 应用patch嵌入
        x = self.patch_embed(x)
        
        # 添加分类token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        # 应用Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 应用归一化
        x = self.norm(x)
        
        # 返回分类token作为图像特征
        return x[:, 0]

# 无人机状态编码器
class StateEncoder(nn.Module):
    """无人机状态编码器"""
    def __init__(self, state_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# 轨迹嵌入层
class TrajectoryEmbedding(nn.Module):
    """轨迹嵌入层 - 处理(512, 50, 4)形状的轨迹数据"""
    def __init__(self, trajectories, embed_dim):
        super().__init__()
        # 轨迹数据形状为(num_trajectories, time_steps, 4)
        self.num_trajectories = trajectories.shape[0]
        self.time_steps = trajectories.shape[1]
        self.traj_dim = trajectories.shape[2]
        
        # 将轨迹注册为缓冲区(不参与训练)
        self.register_buffer('trajectories', torch.tensor(trajectories, dtype=torch.float32))
        
        # 轨迹嵌入网络
        self.encoder = nn.Sequential(
            nn.Linear(self.time_steps * self.traj_dim, embed_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, indices=None):
        if indices is None:
            # 嵌入所有轨迹
            # 先将轨迹展平为(num_trajectories, time_steps*traj_dim)
            traj_flat = self.trajectories.view(self.num_trajectories, -1)
            # 应用嵌入网络
            return self.encoder(traj_flat)
        else:
            # 嵌入选定的轨迹
            selected_trajs = self.trajectories[indices]
            # 展平
            traj_flat = selected_trajs.view(len(indices), -1)
            # 应用嵌入网络
            return self.encoder(traj_flat)

# 特征融合模块
class FeatureFusion(nn.Module):
    """图像特征和状态特征的融合"""
    def __init__(self, visual_dim, state_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + state_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, visual_features, state_features):
        # 拼接特征
        combined = torch.cat([visual_features, state_features], dim=1)
        return self.fusion(combined)

# 完整的无人机导航模型
class DroneNavigationModel(nn.Module):
    def __init__(self,
                 trajectories,
                 image_size=224,
                 patch_size=16,
                 in_channels=1,
                 state_dim=11,
                 embed_dim=256,
                 fusion_dim=256,
                 vit_depth=4,
                 vit_heads=4,
                 vit_mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        # 保存轨迹数据尺寸信息
        self.num_trajectories = trajectories.shape[0]
        
        # 视觉编码器 - 轻量级ViT
        self.vision_encoder = LightweightViT(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=dropout
        )
        
        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            hidden_dim=embed_dim // 2,
            output_dim=embed_dim,
            dropout=dropout
        )
        
        # 特征融合
        self.feature_fusion = FeatureFusion(
            visual_dim=embed_dim,
            state_dim=embed_dim,
            output_dim=fusion_dim,
            dropout=dropout
        )
        
        # 轨迹嵌入
        self.trajectory_embedding = TrajectoryEmbedding(
            trajectories=trajectories,
            embed_dim=fusion_dim
        )
        
        # 输出层 - 计算轨迹的兼容性分数
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.num_trajectories)
        )
    
    def forward(self, image, state):
        # 提取视觉特征
        visual_features = self.vision_encoder(image)
        
        # 编码状态
        state_features = self.state_encoder(state)
        
        # 融合特征
        fused_features = self.feature_fusion(visual_features, state_features)
        
        # 计算轨迹相容性分数
        trajectory_scores = self.output_layer(fused_features)
        
        # 应用softmax获取概率
        probabilities = F.softmax(trajectory_scores, dim=1)
        
        return probabilities
