import numpy as np

def compute_quintic_coefficients(p0, v0, a0, pf, vf, af, T):
    """
    计算五次多项式轨迹的系数。
    
    :param p0: 初始位置
    :param v0: 初始速度
    :param a0: 初始加速度
    :param pf: 终止位置
    :param vf: 终止速度
    :param af: 终止加速度
    :param T: 轨迹总时间
    :return: 多项式系数 [a0, a1, a2, a3, a4, a5]
    """
    # 构造矩阵方程
    M = np.array([
        [0, 0, 0, 0, 0, 1],
        [T**5, T**4, T**3, T**2, T, 1],
        [0, 0, 0, 0, 1, 0],
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]
    ])
    
    # 构造边界条件向量
    b = np.array([p0, pf, v0, vf, a0, af])
    
    # 求解系数
    coeffs = np.linalg.solve(M, b)
    
    return coeffs

def evaluate_quintic(coeffs, t):
    """
    评估五次多项式轨迹在时间 t 处的位置和速度。
    
    :param coeffs: 多项式系数 [a0, a1, a2, a3, a4, a5]
    :param t: 当前时间
    :return: (position, velocity)
    """
    a0, a1, a2, a3, a4, a5 = coeffs
    position = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    velocity = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    return position, velocity

def compute_quintic_coefficients_3d(p0, v0, a0, pf, vf, af, T):
    """
    计算三维五次多项式轨迹的系数。
    
    :param p0: 初始位置 [x0, y0, z0]
    :param v0: 初始速度 [vx0, vy0, vz0]
    :param a0: 初始加速度 [ax0, ay0, az0]
    :param pf: 终止位置 [xf, yf, zf]
    :param vf: 终止速度 [vxf, vyf, vzf]
    :param af: 终止加速度 [axf, ayf, azf]
    :param T: 轨迹总时间
    :return: 多项式系数 [x_coeffs, y_coeffs, z_coeffs]
    """
    coeffs_x = compute_quintic_coefficients(p0[0], v0[0], a0[0], pf[0], vf[0], af[0], T)
    coeffs_y = compute_quintic_coefficients(p0[1], v0[1], a0[1], pf[1], vf[1], af[1], T)
    coeffs_z = compute_quintic_coefficients(p0[2], v0[2], a0[2], pf[2], vf[2], af[2], T)
    return [coeffs_x, coeffs_y, coeffs_z]

def evaluate_quintic_3d(coeffs, t):
    """
    评估三维五次多项式轨迹在时间 t 处的位置和速度。
    
    :param coeffs: 多项式系数 [x_coeffs, y_coeffs, z_coeffs]
    :param t: 当前时间
    :return: (position [x, y, z], velocity [vx, vy, vz])
    """
    pos = []
    vel = []
    for axis in range(3):
        p, v = evaluate_quintic(coeffs[axis], t)
        pos.append(p)
        vel.append(v)
    return np.array(pos), np.array(vel)