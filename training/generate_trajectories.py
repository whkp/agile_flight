import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join as opj
import cv2
import json
import glob

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
from drone_diffusion import DroneTrajectoryDiffusion, DiffusionTrainer
from diffusion_dataloading import denormalize_trajectories, load_preprocessed_data

def parse_args():
    parser = argparse.ArgumentParser(description='生成无人机避障轨迹')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='./generated_trajectories', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10, help='为每个条件生成的样本数')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据文件/目录')
    parser.add_argument('--state_dim', type=int, default=12, help='状态维度')
    parser.add_argument('--sequence_length', type=int, default=64, help='轨迹序列长度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--timesteps', type=int, default=1000, help='扩散模型时间步')
    parser.add_argument('--depth', type=int, default=4, help='扩散模型层数')
    parser.add_argument('--img_size', type=int, default=60, help='输入深度图大小')
    parser.add_argument('--patch_size', type=int, default=4, help='ViT的patch大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--no_cuda', action='store_true', help='不使用CUDA')
    
    return parser.parse_args()

def load_model(checkpoint_path, args, device):
    """加载训练好的模型"""
    # 创建模型
    model = DroneTrajectoryDiffusion(
        state_dim=args.state_dim,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=1  # 深度图为单通道
    ).to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    stats = checkpoint.get('stats', None)
    
    # 创建扩散训练器
    trainer = DiffusionTrainer(
        model=model,
        timesteps=args.timesteps,
        device=device
    )
    
    return model, trainer, stats

def load_test_depth_images(test_data_path, args):
    """加载测试用的深度图像"""
    depth_images = []
    
    # 检查是文件还是目录
    if os.path.isfile(test_data_path):
        # 如果是HDF5文件，使用load_preprocessed_data加载
        if test_data_path.endswith('.h5'):
            _, depth_images = load_preprocessed_data(test_data_path)
        else:
            print(f"不支持的文件格式: {test_data_path}")
            return []
    else:
        # 如果是目录，寻找所有png文件
        depth_files = sorted(glob.glob(os.path.join(test_data_path, '**/*.png'), recursive=True))
        
        for img_file in tqdm(depth_files, desc="加载测试深度图"):
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            # 归一化深度值到 [0, 1]
            img = img.astype(np.float32) / 65535.0  # 假设是16位深度图
            
            # 调整大小以适应模型输入
            if img.shape[0] != args.img_size or img.shape[1] != args.img_size:
                img = cv2.resize(img, (args.img_size, args.img_size))
            
            # 添加通道维度
            img = img[np.newaxis, ...]
            depth_images.append(img)
    
    return depth_images

def visualize_trajectory(trajectory, filename, step_markers=False):
    """可视化轨迹并保存为图像，带有步骤标记选项"""
    fig = plt.figure(figsize=(10, 8))
    
    # 3D轨迹图
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-')
    
    # 起点和终点标记
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', s=50, label='起点')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', s=50, label='终点')
    
    # 可选：添加轨迹点标记
    if step_markers:
        # 每隔几步显示一个标记
        step_size = max(1, len(trajectory) // 10)
        for i in range(0, len(trajectory), step_size):
            ax.text(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2], f"{i}", fontsize=8)
            ax.scatter(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2], c='blue', s=20, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('无人机飞行轨迹')
    ax.legend()
    
    plt.savefig(filename)
    plt.close(fig)

def save_trajectory_data(trajectory, filename):
    """将轨迹数据保存为CSV和JSON格式"""
    # 保存为CSV
    csv_filename = f"{filename}.csv"
    np.savetxt(csv_filename, trajectory, delimiter=',', 
               header='pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,quat_x,quat_y,quat_z,quat_w,ang_vel_x,ang_vel_y,ang_vel_z', 
               comments='')
    
    # 保存为JSON
    json_filename = f"{filename}.json"
    data = {
        "trajectory": trajectory.tolist(),
        "metadata": {
            "sequence_length": len(trajectory),
            "state_dim": trajectory.shape[1],
            "dims": ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", 
                    "quat_x", "quat_y", "quat_z", "quat_w", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
        }
    }
    
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)

def generate_trajectories(model, trainer, depth_images, num_samples, output_dir, stats=None):
    """为每个深度图生成多个轨迹"""
    model.eval()
    
    for i, depth_image in enumerate(tqdm(depth_images, desc="生成轨迹")):
        # 创建样本目录
        sample_dir = opj(output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # 转换深度图为张量并添加批量维度
        depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).to(next(model.parameters()).device)
        
        # 生成多个样本
        for j in range(num_samples):
            # 生成轨迹
            generated_trajectory = trainer.sample(
                batch_size=1,
                sequence_length=model.sequence_length,
                depth_images=depth_tensor
            )[0]  # 取批次中的第一个样本
            
            # 如果有统计信息，反归一化
            if stats is not None:
                generated_trajectory = torch.FloatTensor(
                    denormalize_trajectories([generated_trajectory.cpu().numpy()], stats)[0]
                ).to(generated_trajectory.device)
            
            # 保存轨迹数据
            traj_path = opj(sample_dir, f"traj_{j:02d}")
            save_trajectory_data(generated_trajectory.cpu().numpy(), traj_path)
            
            # 可视化轨迹
            visualize_trajectory(
                generated_trajectory.cpu().numpy(), 
                f"{traj_path}.png",
                step_markers=(j == 0)  # 只在第一个样本上显示步骤标记
            )
            
            # 如果是第一个样本，保存深度图
            if j == 0:
                depth_img = depth_image[0]  # 取第一个通道
                plt.figure(figsize=(5, 5))
                plt.imshow(depth_img, cmap='viridis')
                plt.colorbar(label='深度')
                plt.title('条件深度图')
                plt.savefig(opj(sample_dir, "depth_image.png"))
                plt.close()

def main():
    args = parse_args()
    
    # 设置设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, trainer, stats = load_model(args.checkpoint, args, device)
    print(f"成功加载模型从: {args.checkpoint}")
    
    # 加载测试深度图像
    if args.test_data:
        depth_images = load_test_depth_images(args.test_data, args)
        if not depth_images:
            print("找不到有效的测试深度图像，将使用随机噪声图像")
            # 生成随机噪声图像
            depth_images = [np.random.rand(1, args.img_size, args.img_size).astype(np.float32) for _ in range(5)]
    else:
        # 生成随机噪声图像
        print("未提供测试数据，使用随机噪声图像")
        depth_images = [np.random.rand(1, args.img_size, args.img_size).astype(np.float32) for _ in range(5)]
    
    print(f"准备使用 {len(depth_images)} 个深度图像生成轨迹")
    
    # 生成轨迹
    generate_trajectories(model, trainer, depth_images, args.num_samples, args.output_dir, stats)
    
    print(f"生成完成！轨迹已保存到: {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main()
