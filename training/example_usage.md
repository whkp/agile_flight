# 扩散模型生成无人机飞行轨迹示例

本文档介绍如何使用扩散模型生成无人机的飞行避障轨迹数据。

## 准备环境

确保已安装所需的依赖：

```bash
pip install torch torchvision matplotlib numpy opencv-python tqdm h5py
```

## 模型训练

1. 准备训练数据，包括无人机飞行轨迹和对应的深度图像：

```bash
# 从ROS包中提取数据（如果有）
python -c "from training.diffusion_dataloading import extract_data_from_ros_bags; extract_data_from_ros_bags('/path/to/rosbags', './training/datasets/extracted')"
```

2. 训练扩散模型：

```bash
python training/train_diffusion.py --data_dir ./training/datasets/extracted --output_dir ./results --batch_size 16 --epochs 100 --lr 2e-4 --sequence_length 64 --state_dim 12 --img_size 60 --patch_size 4
```

## 生成轨迹

使用训练好的模型生成新的飞行轨迹：

```bash
# 使用随机噪声作为条件
python training/generate_trajectories.py --checkpoint ./results/drone_diffusion_YYYYMMDD_HHMMSS/checkpoints/ckpt_latest.pt --output_dir ./generated_trajectories --num_samples 10

# 使用现有深度图像作为条件
python training/generate_trajectories.py --checkpoint ./results/drone_diffusion_YYYYMMDD_HHMMSS/checkpoints/ckpt_latest.pt --output_dir ./generated_trajectories --test_data ./path/to/depth/images --num_samples 10
```

## 使用生成的轨迹

生成的轨迹将保存在指定的输出目录中，每个深度图条件下会生成多个轨迹：

- JSON格式：包含完整的轨迹数据和元数据
- CSV格式：轨迹数据，便于导入其他工具分析
- PNG格式：3D轨迹可视化

这些轨迹可以用于：

1. 训练避障控制器的数据增强
2. 无人机仿真环境中的参考轨迹
3. 分析不同环境下的最优避障策略

## 自定义生成过程

生成脚本支持多种参数定制：

```bash
python training/generate_trajectories.py --help
```

主要参数：
- `--state_dim`：状态维度（默认12，包括位置、速度、姿态四元数和角速度）
- `--sequence_length`：轨迹序列长度
- `--timesteps`：扩散过程的时间步数
- `--num_samples`：为每个条件生成的样本数量
- `--img_size`：输入深度图像的大小 