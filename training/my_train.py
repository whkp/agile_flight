import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from einops import rearrange, repeat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 数据集类 - 用于加载深度图像和无人机状态
class DroneNavigationDataset(Dataset):
    def __init__(self, data_dir, trajectory_indices_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 加载轨迹数据 - 形状为(512, 50, 4)
        self.trajectories = np.load(trajectory_indices_path)
        print(f"Trajectory indices shape: {self.trajectories.shape}")
        self.num_trajectories = self.trajectories.shape[0]
        print(f"Number of trajectories: {self.num_trajectories}")
        
        # 打印第一个轨迹的形状以便调试
        print(f"First trajectory shape: {self.trajectories[0].shape}")
        
        # 创建样本列表
        self.samples = []
        
        # 获取所有数据文件夹
        self.data_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        print(f"找到 {len(self.data_folders)} 个数据文件夹")
        
        # 处理每个文件夹的数据
        for folder in tqdm(self.data_folders, desc="加载数据"):
            folder_path = os.path.join(data_dir, folder)
            # 加载CSV数据
            csv_path = os.path.join(folder_path, "data.csv")
            if not os.path.exists(csv_path):
                continue
                
            csv_data = pd.read_csv(csv_path)
            
            # 获取所有图像文件
            image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
            
            for img_path in image_files:
                # 从文件名提取时间戳
                timestamp = float('.'.join(os.path.basename(img_path).split('.')[:2]))
                
                # 在CSV中找到最接近该时间戳的行
                csv_idx = (csv_data['timestamp'] - timestamp).abs().idxmin()
                
                # 获取状态数据
                state_data = csv_data.iloc[csv_idx]
                
                # 获取轨迹标签 (将当前速度命令与最接近的代表性轨迹匹配)
                # 这里使用简化方法，实际应用中应基于实际飞行轨迹进行更精确的匹配
                vel_cmd = np.array([
                    state_data['velcmd_x'], 
                    state_data['velcmd_y'], 
                    state_data['velcmd_z']
                ])
                
                # 简化的轨迹匹配，随机选择一个代表性轨迹作为目标
                # 在实际应用中，应该基于完整的轨迹相似度计算
                target_idx = np.random.randint(0, self.num_trajectories)
                
                self.samples.append({
                    'image_path': img_path,
                    'state_data': state_data,
                    'target_idx': target_idx,
                    'episode_idx': folder  # 保存文件夹名作为episode索引
                })
        
        print(f"总共加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载深度图像
        img = Image.open(sample['image_path']).convert('L')  # 转为灰度图
        if self.transform:
            img = self.transform(img)
        
        # 提取状态数据
        state_data = sample['state_data']
        state_tensor = torch.tensor([
            state_data['quat_1'],
            state_data['quat_2'],
            state_data['quat_3'],
            state_data['quat_4'],
            state_data['pos_x'],
            state_data['pos_y'],
            state_data['pos_z'],
            state_data['vel_x'],
            state_data['vel_y'],
            state_data['vel_z'],
            state_data['desired_vel']
        ], dtype=torch.float32)
        
        # 获取目标轨迹索引 (简化，使用随机选择的轨迹)
        target_idx = sample['target_idx']
        
        # 创建目标的one-hot编码
        target_onehot = torch.zeros(self.num_trajectories)
        target_onehot[target_idx] = 1.0
        
        return img, state_tensor, target_onehot

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

# 混合训练函数 - 结合教师强制和强化学习
def train_model(model, train_loader, val_loader, 
                num_epochs=50, 
                lr=1e-4, 
                teacher_forcing_ratio=0.5, 
                rl_weight=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    print(f"使用设备: {device}")
    
    # 将模型移至设备
    model = model.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_start_time = time.time()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, states, targets) in enumerate(progress_bar):
            # 将数据移至设备
            images = images.to(device)
            states = states.to(device)
            targets = targets.to(device)
            
            # 前向传播
            probs = model(images, states)
            
            # 教师强制损失 (交叉熵)
            ce_loss = F.cross_entropy(probs, targets.argmax(dim=1))
            
            # 强化学习组件 - 熵正则化促进探索
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            
            # 合并损失
            loss = (1 - rl_weight) * ce_loss - rl_weight * entropy_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新运行损失
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce_loss': f"{ce_loss.item():.4f}",
                'entropy': f"{entropy_loss.item():.4f}"
            })
        
        # 平均训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, states, targets in tqdm(val_loader, desc="验证"):
                # 将数据移至设备
                images = images.to(device)
                states = states.to(device)
                targets = targets.to(device)
                
                # 前向传播
                probs = model(images, states)
                
                # 计算损失
                loss = F.cross_entropy(probs, targets.argmax(dim=1))
                val_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(probs, 1)
                _, true = torch.max(targets, 1)
                total += targets.size(0)
                correct += (predicted == true).sum().item()
        
        # 平均验证损失和准确率
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 计算每个epoch的耗时
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{num_epochs} 完成 (耗时: {epoch_time:.2f}秒)")
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 准确率: {accuracy:.2f}%")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'accuracy': accuracy
            }, 'best_drone_navigation_model.pth')
            print(f"模型已保存，验证损失: {val_loss:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(100 * (1 - np.array(val_losses) / val_losses[0]), label='相对于初始值的改进百分比')
    plt.xlabel('Epoch')
    plt.ylabel('改进百分比')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()
    
    print("训练完成！")
    return model

# 评估函数
def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, states, targets in tqdm(test_loader, desc="评估"):
            # 将数据移至设备
            images = images.to(device)
            states = states.to(device)
            targets = targets.to(device)
            
            # 前向传播
            probs = model(images, states)
            
            # 保存概率和目标
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # 计算准确率
            _, predicted = torch.max(probs, 1)
            _, true = torch.max(targets, 1)
            total += targets.size(0)
            correct += (predicted == true).sum().item()
    
    # 计算整体准确率
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    
    # 计算Top-5准确率
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    top5_correct = 0
    
    for i in range(len(all_probs)):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if np.argmax(all_targets[i]) in top5_indices:
            top5_correct += 1
    
    top5_accuracy = 100 * top5_correct / total
    print(f'Top-5 准确率: {top5_accuracy:.2f}%')
    
    return accuracy, top5_accuracy

# 主函数 - 运行训练流程
def main():
    # 设置随机种子以提高可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义路径
    data_dir = 'datasets/data'
    trajectory_indices_path = 'selected_indices_yaw.npy'
    
    # 加载轨迹数据
    print("加载轨迹数据...")
    trajectories = np.load(trajectory_indices_path)
    print(f"轨迹数据形状: {trajectories.shape}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集
    print("创建数据集...")
    full_dataset = DroneNavigationDataset(
        data_dir=data_dir,
        trajectory_indices_path=trajectory_indices_path,
        transform=transform
    )
    
    # 分割数据集为训练集、验证集和测试集
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size + test_size]
    )
    
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    print(f"数据集分割: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}, 测试 {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    print("初始化模型...")
    model = DroneNavigationModel(
        trajectories=trajectories,
        image_size=224,
        patch_size=16,
        in_channels=1,
        state_dim=11,
        embed_dim=256,
        fusion_dim=256,
        vit_depth=4,
        vit_heads=4,
        vit_mlp_ratio=4.0,
        dropout=0.1
    )
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 训练模型
    print("开始训练...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        lr=3e-4,
        teacher_forcing_ratio=0.5,
        rl_weight=0.2
    )
    
    # 评估模型
    print("开始评估...")
    accuracy, top5_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader
    )
    
    print("完成所有操作!")
    print(f"最终准确率: {accuracy:.2f}%, Top-5准确率: {top5_accuracy:.2f}%")

if __name__ == '__main__':
    main()