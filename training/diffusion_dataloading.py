import os
import numpy as np
import torch
import glob
import h5py
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from os.path import join as opj
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
from drone_diffusion import TrajectoryDataset

def load_trajectory_data(data_dir, max_trajectories=None):
    """
    加载无人机飞行轨迹数据
    
    参数:
        data_dir: 数据目录路径
        max_trajectories: 最大轨迹数量，None表示加载所有
    
    返回:
        trajectories: 轨迹状态数组 [num_trajectories, max_seq_len, state_dim]
        depth_images: 对应的深度图像 [num_trajectories, channels, height, width]
        traj_lengths: 每个轨迹的实际长度
    """
    print(f"[DATA] 从 {data_dir} 加载轨迹数据")
    
    # 找到所有轨迹子目录
    traj_dirs = sorted(glob.glob(os.path.join(data_dir, "traj_*")))
    if max_trajectories is not None:
        traj_dirs = traj_dirs[:max_trajectories]
    
    all_trajectories = []
    all_depth_images = []
    all_traj_lengths = []
    
    for traj_dir in traj_dirs:
        # 加载状态数据
        state_file = os.path.join(traj_dir, "states.npy")
        if os.path.exists(state_file):
            states = np.load(state_file)
            
            # 加载对应的深度图像
            depth_dir = os.path.join(traj_dir, "depth")
            depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
            
            if len(depth_files) > 0:
                # 读取第一张图像以获取尺寸
                sample_img = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
                img_height, img_width = sample_img.shape[:2]
                
                # 为这个轨迹创建深度图像数组
                depth_images = np.zeros((len(depth_files), 1, img_height, img_width), dtype=np.float32)
                
                # 读取所有深度图像
                for i, img_file in enumerate(depth_files):
                    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                    # 归一化深度值到 [0, 1]
                    img = img.astype(np.float32) / 65535.0  # 假设是16位深度图
                    depth_images[i, 0] = img
                
                # 确保状态和图像数量匹配
                min_len = min(states.shape[0], depth_images.shape[0])
                states = states[:min_len]
                depth_images = depth_images[:min_len]
                
                all_trajectories.append(states)
                all_depth_images.append(depth_images)
                all_traj_lengths.append(min_len)
                
                print(f"  加载轨迹 {os.path.basename(traj_dir)}: {min_len} 帧")
    
    print(f"[DATA] 成功加载 {len(all_trajectories)} 个轨迹")
    return all_trajectories, all_depth_images, all_traj_lengths

def preprocess_trajectory_data(trajectories, depth_images, traj_lengths, sequence_length=64):
    """
    预处理轨迹数据，确保所有轨迹都有相同的长度
    
    参数:
        trajectories: 轨迹列表，每个轨迹是一个状态序列
        depth_images: 深度图像列表，与轨迹对应
        traj_lengths: 每个轨迹的实际长度
        sequence_length: 期望的序列长度
    
    返回:
        processed_trajectories: 预处理后的轨迹
        processed_images: 预处理后的图像
    """
    processed_trajectories = []
    processed_images = []
    
    for traj, imgs, length in zip(trajectories, depth_images, traj_lengths):
        # 只处理长度足够的轨迹
        if length >= sequence_length:
            # 处理每个完整的序列
            for start_idx in range(0, length - sequence_length + 1, sequence_length // 2):  # 50% 重叠
                seq_traj = traj[start_idx:start_idx + sequence_length]
                seq_imgs = imgs[start_idx:start_idx + sequence_length]
                
                # 使用第一帧的深度图作为条件
                cond_img = seq_imgs[0]
                
                processed_trajectories.append(seq_traj)
                processed_images.append(cond_img)
    
    return processed_trajectories, processed_images

def create_diffusion_dataloaders(data_dir, batch_size=32, sequence_length=64, val_split=0.1, 
                                max_trajectories=None, num_workers=4):
    """
    创建用于扩散模型训练的数据加载器
    
    参数:
        data_dir: 数据目录
        batch_size: 批处理大小
        sequence_length: 序列长度
        val_split: 验证集比例
        max_trajectories: 最大轨迹数量
        num_workers: 数据加载器工作线程数
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 加载原始数据
    trajectories, depth_images, traj_lengths = load_trajectory_data(data_dir, max_trajectories)
    
    # 预处理数据
    processed_trajectories, processed_images = preprocess_trajectory_data(
        trajectories, depth_images, traj_lengths, sequence_length)
    
    # 分割训练集和验证集
    num_samples = len(processed_trajectories)
    indices = np.random.permutation(num_samples)
    split_idx = int(np.floor(val_split * num_samples))
    train_indices = indices[split_idx:]
    val_indices = indices[:split_idx]
    
    train_trajectories = [processed_trajectories[i] for i in train_indices]
    train_images = [processed_images[i] for i in train_indices]
    
    val_trajectories = [processed_trajectories[i] for i in val_indices]
    val_images = [processed_images[i] for i in val_indices]
    
    # 定义图像变换
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集
    train_dataset = TrajectoryDataset(train_trajectories, train_images, sequence_length, transforms=None)
    val_dataset = TrajectoryDataset(val_trajectories, val_images, sequence_length, transforms=None)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"[DATA] 创建了训练数据加载器: {len(train_dataset)} 样本")
    print(f"[DATA] 创建了验证数据加载器: {len(val_dataset)} 样本")
    
    return train_loader, val_loader

def extract_data_from_ros_bags(bag_dir, output_dir):
    """
    从ROS包中提取飞行数据
    
    参数:
        bag_dir: ROS包目录
        output_dir: 输出目录
    """
    try:
        import rosbag
        import tf
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError:
        print("[ERROR] 无法导入ROS相关包，请确保已安装rosbag、tf、sensor_msgs和cv_bridge")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有ROS包
    bag_files = glob.glob(os.path.join(bag_dir, "*.bag"))
    bridge = CvBridge()
    
    for bag_idx, bag_file in enumerate(bag_files):
        print(f"[EXTRACT] 处理ROS包 {bag_file}")
        
        traj_dir = os.path.join(output_dir, f"traj_{bag_idx:04d}")
        os.makedirs(traj_dir, exist_ok=True)
        os.makedirs(os.path.join(traj_dir, "depth"), exist_ok=True)
        
        # 打开ROS包
        bag = rosbag.Bag(bag_file)
        
        # 收集状态和图像数据
        states = []
        timestamps = []
        
        # 首先获取所有状态和时间戳
        for topic, msg, t in bag.read_messages(topics=['/kingfisher/unity/state']):
            # 示例状态提取，需根据实际消息类型调整
            position = [msg.position.x, msg.position.y, msg.position.z]
            velocity = [msg.velocity.x, msg.velocity.y, msg.velocity.z]
            quaternion = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
            
            state = position + velocity + quaternion + angular_velocity
            states.append(state)
            timestamps.append(t.to_sec())
        
        # 将状态保存为numpy数组
        states = np.array(states, dtype=np.float32)
        np.save(os.path.join(traj_dir, "states.npy"), states)
        
        # 处理深度图像
        image_count = 0
        for topic, msg, t in bag.read_messages(topics=['/kingfisher/unity/depth']):
            # 将ROS图像消息转换为OpenCV图像
            try:
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                
                # 保存图像
                img_filename = os.path.join(traj_dir, "depth", f"{t.to_sec():.3f}.png")
                cv2.imwrite(img_filename, cv_img)
                image_count += 1
            except Exception as e:
                print(f"[ERROR] 无法转换图像: {e}")
        
        print(f"[EXTRACT] 轨迹 {bag_idx}: 保存了 {len(states)} 个状态和 {image_count} 个深度图像")
        
        bag.close()
    
    print(f"[EXTRACT] 完成从 {len(bag_files)} 个ROS包中提取数据")

def normalize_trajectories(trajectories):
    """
    归一化轨迹数据
    
    参数:
        trajectories: 轨迹列表
    
    返回:
        normalized_trajectories: 归一化后的轨迹
        stats: 归一化统计信息 (均值, 标准差)
    """
    # 将所有轨迹合并为一个大数组
    all_states = np.vstack([traj for traj in trajectories])
    
    # 计算均值和标准差
    mean = np.mean(all_states, axis=0)
    std = np.std(all_states, axis=0)
    
    # 确保标准差不为0
    std = np.where(std < 1e-6, 1.0, std)
    
    # 归一化每个轨迹
    normalized_trajectories = []
    for traj in trajectories:
        normalized_traj = (traj - mean) / std
        normalized_trajectories.append(normalized_traj)
    
    return normalized_trajectories, (mean, std)

def denormalize_trajectories(normalized_trajectories, stats):
    """
    反归一化轨迹数据
    
    参数:
        normalized_trajectories: 归一化后的轨迹
        stats: 归一化统计信息 (均值, 标准差)
    
    返回:
        trajectories: 原始轨迹
    """
    mean, std = stats
    trajectories = []
    
    for norm_traj in normalized_trajectories:
        traj = norm_traj * std + mean
        trajectories.append(traj)
    
    return trajectories

def augment_trajectory_data(trajectories, depth_images, factor=2):
    """
    通过简单的数据增强扩充轨迹数据
    
    参数:
        trajectories: 轨迹列表
        depth_images: 深度图像列表
        factor: 扩充因子
    
    返回:
        augmented_trajectories: 增强后的轨迹
        augmented_images: 增强后的图像
    """
    augmented_trajectories = []
    augmented_images = []
    
    for traj, img in zip(trajectories, depth_images):
        # 添加原始数据
        augmented_trajectories.append(traj)
        augmented_images.append(img)
        
        # 添加简单的噪声
        for _ in range(factor - 1):
            # 对状态添加小随机扰动
            noise = np.random.normal(0, 0.02, traj.shape)
            noisy_traj = traj + noise
            
            # 对图像添加小随机扰动
            img_noise = np.random.normal(0, 0.02, img.shape)
            noisy_img = np.clip(img + img_noise, 0, 1)
            
            augmented_trajectories.append(noisy_traj)
            augmented_images.append(noisy_img)
    
    return augmented_trajectories, augmented_images

def save_preprocessed_data(trajectories, depth_images, output_file):
    """
    保存预处理的数据
    
    参数:
        trajectories: 轨迹列表
        depth_images: 深度图像列表
        output_file: 输出文件路径
    """
    with h5py.File(output_file, 'w') as f:
        # 创建轨迹数据集
        for i, (traj, img) in enumerate(zip(trajectories, depth_images)):
            traj_group = f.create_group(f'trajectory_{i}')
            traj_group.create_dataset('states', data=traj)
            traj_group.create_dataset('depth_image', data=img)
    
    print(f"[DATA] 已保存预处理数据到 {output_file}, 共 {len(trajectories)} 个轨迹")

def load_preprocessed_data(input_file):
    """
    加载预处理的数据
    
    参数:
        input_file: 输入文件路径
    
    返回:
        trajectories: 轨迹列表
        depth_images: 深度图像列表
    """
    trajectories = []
    depth_images = []
    
    with h5py.File(input_file, 'r') as f:
        for traj_name in f.keys():
            traj_group = f[traj_name]
            traj = traj_group['states'][:]
            img = traj_group['depth_image'][:]
            
            trajectories.append(traj)
            depth_images.append(img)
    
    print(f"[DATA] 从 {input_file} 加载了 {len(trajectories)} 个轨迹")
    return trajectories, depth_images 