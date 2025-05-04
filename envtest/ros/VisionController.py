import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from copy import deepcopy
import os
import time
import rospy
from threading import Lock
from utils import AgileCommand

# 导入模型架构定义
from VAD import DroneNavigationModel, LightweightViT, StateEncoder, FeatureFusion, TrajectoryEmbedding

class VisionBasedController:
    def __init__(self, model_path, trajectory_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.trajectory_path = trajectory_path
        self.lock = Lock()  # 用于线程安全
        
        # 加载轨迹数据
        self.trajectories = np.load(trajectory_path)
        rospy.loginfo(f"Loaded trajectories with shape: {self.trajectories.shape}")
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # 加载模型
        self.load_model()
        
        # 保存最新的预测结果
        self.latest_prediction = None
        self.latest_control = None
        self.prediction_time = None
        
        rospy.loginfo("Vision-based controller initialized successfully.")
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            # 创建模型实例
            self.model = DroneNavigationModel(
                trajectories=self.trajectories,
                image_size=224,
                patch_size=16,
                in_channels=1,  # 深度图是单通道
                state_dim=11,
                embed_dim=256,
                fusion_dim=256,
                vit_depth=4,
                vit_heads=4,
                vit_mlp_ratio=4.0,
                dropout=0.0  # 推理时关闭dropout
            )
            
            # 加载训练好的权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 将模型设为评估模式并移至指定设备
            self.model.eval()
            self.model.to(self.device)
            
            rospy.loginfo(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_image(self, cv_image):
        """将OpenCV图像转换为适合模型输入的张量"""
        # 确保图像是灰度的
        if len(cv_image.shape) > 2 and cv_image.shape[2] > 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(cv_image)
        
        # 应用变换
        tensor_image = self.transform(pil_image)
        
        # 添加批次维度
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        return tensor_image
    
    def extract_state_features(self, state):
        """从无人机状态提取模型所需的特征"""
        # 提取四元数、位置、速度和期望速度
        quat = [state.att[0], state.att[1], state.att[2], state.att[3]]
        pos = [state.pos[0], state.pos[1], state.pos[2]]
        vel = [state.vel[0], state.vel[1], state.vel[2]]
        desired_vel = state.desired_velocity
        
        # 组合特征
        state_features = quat + pos + vel + [desired_vel]
        state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return state_tensor
    
    def predict(self, state, cv_image):
        """使用模型预测轨迹概率"""
        with self.lock:
            try:
                # 预处理图像
                tensor_image = self.preprocess_image(cv_image)
                
                # 提取状态特征
                state_tensor = self.extract_state_features(state)
                
                # 使用模型进行推理
                with torch.no_grad():
                    start_time = time.time()
                    probabilities = self.model(tensor_image, state_tensor)
                    inference_time = time.time() - start_time
                
                # 转换为numpy数组
                probs_np = probabilities.cpu().numpy()[0]
                
                # 获取最高概率的轨迹索引
                top_k_indices = np.argsort(probs_np)[-5:][::-1]  # 获取前5个
                top_probs = probs_np[top_k_indices]
                
                # 获取最高概率的轨迹
                best_traj_idx = top_k_indices[0]
                best_traj = self.trajectories[best_traj_idx]
                
                # 更新最新预测
                self.latest_prediction = {
                    'best_traj_idx': best_traj_idx,
                    'best_traj': best_traj,
                    'top_indices': top_k_indices,
                    'top_probs': top_probs,
                    'inference_time': inference_time
                }
                self.prediction_time = time.time()
                
                rospy.logdebug(f"Predicted best trajectory {best_traj_idx} with probability {top_probs[0]:.4f}")
                rospy.logdebug(f"Inference time: {inference_time*1000:.2f} ms")
                
                return self.latest_prediction
                
            except Exception as e:
                rospy.logerr(f"Prediction error: {str(e)}")
                return None
    
    def trajectory_to_command(self, prediction, command_mode=2, target_vel=5.0):
        """将轨迹转换为无人机控制命令"""
        if prediction is None:
            rospy.logwarn("No valid prediction available for command generation")
            return None
        
        best_traj = prediction['best_traj']
        
        # 提取轨迹的第一步作为即时控制指令
        # 假设轨迹格式为[t, [x, y, z, yaw]]
        first_step = best_traj[0]  # 轨迹的第一步
        
        if command_mode == 0:
            # SRT命令 - 转子推力
            # 注意：此处需要根据具体的无人机动力学模型转换
            # 简化示例
            command = AgileCommand(command_mode)
            command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]  # 简单默认值
            
        elif command_mode == 1:
            # CTBR命令 - 集体推力和体速率
            # 简化实现
            command = AgileCommand(command_mode)
            command.collective_thrust = 15.0  # 默认推力
            
            # 从轨迹生成体速率（这需要知道无人机的具体动力学模型）
            # 简化示例
            command.bodyrates = [0.0, 0.0, 0.0]
            
        elif command_mode == 2:
            # LINVEL命令 - 线速度（世界坐标系）
            command = AgileCommand(command_mode)
            
            # 假设轨迹中的前三个值是位置[x,y,z]，我们计算速度方向
            if best_traj.shape[0] > 1:  # 确保有足够的点计算速度
                # 计算前两个点之间的速度向量
                pos_current = best_traj[0, :3]  # 当前位置
                pos_next = best_traj[1, :3]    # 下一位置
                
                # 计算速度向量，可以根据需要缩放
                velocity_vec = pos_next - pos_current
                velocity_norm = np.linalg.norm(velocity_vec)
                
                # 避免除以零
                if velocity_norm > 1e-6:
                    # 计算期望速度方向和大小
                    desired_direction = velocity_vec / velocity_norm
                    
                    #使用目标速度
                    command.velocity = (desired_direction * target_vel).tolist()
                else:
                    #默认前向低速行驶
                    command.velocity = [target_vel * 0.2, 0.0, 0.0]
                
                # 提取偏航速率（如果轨迹中有）
                # 假设第四个值是偏航角
                if best_traj.shape[1] > 3:
                    yaw_current = best_traj[0, 3]
                    yaw_next = best_traj[1, 3]
                    # 计算偏航速率（注意处理角度换算）
                    yaw_diff = yaw_next - yaw_current
                    # 归一化到[-pi, pi]
                    if yaw_diff > np.pi:
                        yaw_diff -= 2 * np.pi
                    elif yaw_diff < -np.pi:
                        yaw_diff += 2 * np.pi
                    
                    command.yawrate = yaw_diff  # 可以根据需要调整比例
                else:
                    command.yawrate = 0.0
            else:
                # 如果轨迹点不足，使用默认值
                command.velocity = [target_vel * 0.2, 0.0, 0.0]
                command.yawrate = 0.0
        
        # 保存最新的控制命令
        self.latest_control = command
        
        return command

# 在全局作用域创建控制器实例
controller = None

def init_controller():
    """初始化控制器，只需调用一次"""
    if controller is None:
        #当前脚本的绝对路径
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_drone_navigation_model.pth')
        trajectory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selected_indices_yaw.npy')
        
        controller = VisionBasedController(model_path, trajectory_path)
        rospy.loginfo("Controller initialized")

def img_callback(self, img_data):
    """图像回调函数"""
    global controller
    
    # 初始化控制器（如果还未初始化）
    if controller is None:
        try:
            init_controller()
        except Exception as e:
            rospy.logerr(f"Failed to initialize controller: {str(e)}")
            return
    
    # 转换图像
    cv_image = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')
    
    # 更新最后有效图像
    if self.last_valid_img is None:
        self.last_valid_img = deepcopy(cv_image)
    
    # 检查是否启用了基于视觉的控制和是否有有效的状态
    if not self.vision_based or self.state is None:
        return
    
    # 计算命令
    command = compute_command_vision_based(self.state, cv_image)
    
    # 发布命令
    self.publish_command(command)

def compute_command_vision_based(state, img):
    """基于视觉计算控制命令"""
    global controller
    
    # 确保控制器已初始化
    if controller is None:
        try:
            init_controller()
        except Exception as e:
            rospy.logerr(f"Failed to initialize controller: {str(e)}")
            # 返回安全的默认命令
            command_mode = 2  # LINVEL模式
            command = AgileCommand(command_mode)
            command.t = state.t
            command.velocity = [0.0, 0.0, 0.0]  # 悬停
            command.yawrate = 0.0
            return command
    
    # 使用模型进行预测
    prediction = controller.predict(state, img)
    
    # 选择命令模式（以LINVEL为例）
    command_mode = 2
    
    # 将预测转换为命令
    command = controller.trajectory_to_command(prediction, command_mode)
    
    # 如果转换失败，返回安全的默认命令
    if command is None:
        command = AgileCommand(command_mode)
        command.t = state.t
        command.velocity = [0.0, 0.0, 0.0]  # 悬停
        command.yawrate = 0.0
    else:
        # 设置时间戳
        command.t = state.t
    
    # 记录日志
    rospy.loginfo(f"Command: mode={command_mode}, vel=[{command.velocity[0]:.2f}, {command.velocity[1]:.2f}, {command.velocity[2]:.2f}], yawrate={command.yawrate:.2f}")
    
    return command