import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from os.path import join as opj

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
from drone_diffusion import DroneTrajectoryDiffusion, DiffusionTrainer
from diffusion_dataloading import create_diffusion_dataloaders, normalize_trajectories, denormalize_trajectories

def parse_args():
    parser = argparse.ArgumentParser(description='训练无人机轨迹扩散模型')
    parser.add_argument('--data_dir', type=str, required=True, help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--sequence_length', type=int, default=64, help='轨迹序列长度')
    parser.add_argument('--state_dim', type=int, default=12, help='状态维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--save_interval', type=int, default=10, help='保存间隔')
    parser.add_argument('--val_interval', type=int, default=5, help='验证间隔')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散模型时间步')
    parser.add_argument('--depth', type=int, default=4, help='扩散模型网络深度')
    parser.add_argument('--img_size', type=int, default=60, help='输入深度图大小')
    parser.add_argument('--patch_size', type=int, default=4, help='ViT的patch大小')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    
    return parser.parse_args()

def setup_experiment(args):
    """设置实验目录和日志"""
    # 创建唯一的实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = opj(args.output_dir, f"drone_diffusion_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 创建子目录
    ckpt_dir = opj(exp_dir, "checkpoints")
    samples_dir = opj(exp_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    # 设置TensorBoard
    tb_dir = opj(exp_dir, "logs")
    writer = SummaryWriter(tb_dir)
    
    # 保存参数
    with open(opj(exp_dir, "args.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    return exp_dir, ckpt_dir, samples_dir, writer

def save_checkpoint(model, trainer, optimizer, epoch, loss, ckpt_dir, stats=None):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'stats': stats  # 保存归一化统计信息
    }
    
    torch.save(checkpoint, opj(ckpt_dir, f"ckpt_epoch_{epoch}.pt"))
    # 保存最新的检查点
    torch.save(checkpoint, opj(ckpt_dir, "ckpt_latest.pt"))

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    stats = checkpoint.get('stats', None)
    
    return model, optimizer, epoch, loss, stats

def visualize_trajectory(trajectory, filename):
    """可视化轨迹并保存为图像"""
    fig = plt.figure(figsize=(10, 8))
    
    # 3D轨迹图
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', s=50, label='起点')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', s=50, label='终点')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('无人机飞行轨迹')
    ax.legend()
    
    plt.savefig(filename)
    plt.close(fig)

def visualize_sample_batch(batch_trajectories, epoch, samples_dir, stats=None):
    """可视化一批生成的样本轨迹"""
    os.makedirs(opj(samples_dir, f"epoch_{epoch}"), exist_ok=True)
    
    # 如果提供了统计信息，反归一化轨迹
    if stats is not None:
        batch_trajectories = denormalize_trajectories(batch_trajectories, stats)
    
    for i, traj in enumerate(batch_trajectories):
        filename = opj(samples_dir, f"epoch_{epoch}", f"traj_{i}.png")
        visualize_trajectory(traj.cpu().numpy(), filename)

def evaluate(model, val_loader, trainer, device, epoch, writer, samples_dir, stats=None):
    """评估模型并生成样本"""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    # 生成和可视化样本
    with torch.no_grad():
        # 从验证集获取一批图像用于条件生成
        for trajectories, depth_images in val_loader:
            depth_images = depth_images.to(device)
            
            # 生成样本轨迹
            generated_trajectories = trainer.sample(
                batch_size=depth_images.size(0), 
                sequence_length=model.sequence_length, 
                depth_images=depth_images
            )
            
            # 可视化生成的轨迹
            visualize_sample_batch(generated_trajectories, epoch, samples_dir, stats)
            
            # 跳出循环，只生成一批样本
            break
        
        # 计算验证损失
        for trajectories, depth_images in val_loader:
            trajectories = trajectories.to(device)
            depth_images = depth_images.to(device)
            batch_size = trajectories.size(0)
            
            # 随机采样时间步
            t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = trainer.p_losses(trajectories, t, depth_images)
            val_loss += loss.item()
            num_batches += 1
    
    val_loss /= num_batches
    writer.add_scalar('Loss/validation', val_loss, epoch)
    
    return val_loss

def train_epoch(model, train_loader, trainer, optimizer, device, epoch, writer):
    """训练一个轮次"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (trajectories, depth_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        trajectories = trajectories.to(device)
        depth_images = depth_images.to(device)
        batch_size = trajectories.size(0)
        
        # 随机采样时间步
        t = torch.randint(0, trainer.timesteps, (batch_size,), device=device).long()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 计算损失
        loss = trainer.p_losses(trajectories, t, depth_images)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    
    return avg_loss

def main():
    args = parse_args()
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 设置实验目录和日志
    exp_dir, ckpt_dir, samples_dir, writer = setup_experiment(args)
    
    # 创建数据加载器
    train_loader, val_loader = create_diffusion_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        val_split=0.1,
        num_workers=4
    )
    
    # 获取归一化统计信息
    # 假设我们可以访问第一个批次的数据
    sample_trajectories = []
    for traj, _ in train_loader:
        for t in traj:
            sample_trajectories.append(t.numpy())
        break
    
    normalized_trajectories, stats = normalize_trajectories(sample_trajectories)
    
    # 创建模型和训练器
    model = DroneTrajectoryDiffusion(
        state_dim=args.state_dim,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=1  # 深度图为单通道
    ).to(device)
    
    trainer = DiffusionTrainer(
        model=model,
        timesteps=args.timesteps,
        device=device
    )
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 恢复训练（如果提供了检查点）
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        model, optimizer, start_epoch, best_val_loss, loaded_stats = load_checkpoint(
            model, optimizer, args.resume, device)
        if loaded_stats is not None:
            stats = loaded_stats
        print(f"从检查点恢复训练: {args.resume}，当前轮次: {start_epoch}")
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 训练一个轮次
        train_loss = train_epoch(model, train_loader, trainer, optimizer, device, epoch, writer)
        print(f"轮次 {epoch}, 训练损失: {train_loss:.6f}")
        
        # 定期验证
        if (epoch + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_loader, trainer, device, epoch, writer, samples_dir, stats)
            print(f"轮次 {epoch}, 验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, trainer, optimizer, epoch, val_loss, ckpt_dir, stats)
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, trainer, optimizer, epoch, train_loss, ckpt_dir, stats)
            print(f"保存检查点，轮次: {epoch}")
    
    # 保存最终模型
    save_checkpoint(model, trainer, optimizer, args.epochs - 1, train_loss, ckpt_dir, stats)
    print("训练完成！")
    
    # 关闭TensorBoard写入器
    writer.close()

if __name__ == "__main__":
    main() 