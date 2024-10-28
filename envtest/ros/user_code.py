#!/usr/bin/python3


from pickle import NONE
from utils import AgileCommandMode, AgileCommand
from rl_example import rl_example
import numpy as np

def check_collision(line, obstacle):
    # 获取线段的起点和终点坐标
    (x1, y1, z1), (x2, y2, z2) = line
    # 获取障碍物的中心坐标和半径
    (x3, y3, z3), r = obstacle
    # 计算b的值
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    # 计算a的值
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    # 计算c的值
    c = (
        x3**2
        + y3**2
        + z3**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r**2
    )
    # 判断是否有碰撞
    return b**2 - 4 * a * c >= 0

def compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command vision-based!")
    # print(state)
    # print("Image shape: ", img.shape)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    ################################################
    # !!! End of user code !!!
    ################################################

    return command

def find_closest_zero_index(arr):
    center = np.array(arr.shape) // 2  # find the center point of the array
    dist_to_center = np.abs(np.indices(arr.shape) - center.reshape(-1, 1, 1)).sum(0)  # calculate distance to center for each element
    zero_indices = np.argwhere(arr == 0)  # find indices of all zero elements
    if len(zero_indices) == 0:
        return None  # if no zero elements, return None
    dist_to_zeros = dist_to_center[tuple(zero_indices.T)]  # get distances to center for zero elements
    min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()  # find indices of zero elements with minimum distance to center
    chosen_index = np.random.choice(min_dist_indices)  # randomly choose one of the zero elements with minimum distance to center
    return tuple(zero_indices[chosen_index])  # return index tuple


def compute_command_state_based(state, obstacles, desiredVel, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0

    obst_dist_threshold = 8
    obst_inflate_factor = 0.6 #0.4#0.6

    #x偏移量保证无人机往前飞
    x_displacement = 8
    #网格遍历，寻找无碰撞路径点
    grid_center_offset = 8
    grid_displacement = 0.25
    y_values = np.arange(-grid_center_offset, grid_center_offset+grid_displacement, grid_displacement)
    num_wpts = y_values.size

    wpts_2d = np.zeros((num_wpts, num_wpts, 3))
    collisions = np.zeros((num_wpts, num_wpts))
    for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
        for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            wpts_2d[yi, xi] = [x_displacement, x, y]
            for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                # print(f'wpt: {wpts_2d[yi, xi]} \t obst: {obst.position.x, obst.position.y, obst.position.z, obst.scale+obst_inflate_factor}')
                if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                    collisions[yi, xi] = 1
                    break


    if collisions.sum() == collisions.size:
        print(f'[EXPERT] No collision-free path found')
        xvel = 0.1
        yvel = 0
        zvel = 0            
    else:
        # 如果找到了没有碰撞的路径点，使用 find_closest_zero_index 函数找到距离中心点最近的未发生碰撞的路径点
        wpt_idx = find_closest_zero_index(collisions)
        wpt = wpts_2d[wpt_idx[0], wpt_idx[1]]

        # make the desired velocity vector of magnitude desiredVel
        wpt = (wpt / np.linalg.norm(wpt)) * desiredVel
        xvel = wpt[0]
        yvel = wpt[1]
        zvel = wpt[2]
    
    scaler = desiredVel/np.linalg.norm([xvel, yvel, zvel])
    xvel, yvel, zvel = (xvel*scaler, yvel*scaler, zvel*scaler)

    command.velocity = [xvel, yvel, zvel]

    # recover altitude if too low
    if state.pos[2] < 2:
        command.velocity[2] = (2 - state.pos[2]) * 2
    
    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)

    # # Example of SRT command
    # command_mode = 0
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]
 
    # # Example of CTBR command
    # command_mode = 1
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.collective_thrust = 10.0
    # command.bodyrates = [0.0, 0.0, 0.0]

    # # Example of LINVEL command (velocity is expressed in world frame)
    # command_mode = 2
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.velocity = [1.0, 0.0, 0.0]
    # command.yawrate = 0.0

    # # If you want to test your RL policy
    # if rl_policy is not None:
    #     command = rl_example(state, obstacles, rl_policy)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command
