import numpy as np

class ILQR:
    def __init__(self, dynamics, cost_fn, max_iter=100, tol=1e-6, alpha=0.1):
        """
        初始化ILQR优化器。

        参数：
            dynamics (object): 包含系统动力学的对象，需实现：
                - step(state, control) -> next_state
                - derivatives(state, control) -> (A, B)
            cost_fn (object): 包含成本函数的对象，需实现：
                - cost(state, control, next_state) -> cost_value
                - derivatives(state, control, next_state) -> (cx, cu, cxx, cuu, cxu, cx_next, cu_next)
            max_iter (int): 最大迭代次数。
            tol (float): 收敛容忍度。
            alpha (float): 学习率或步长缩放因子。
        """
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha  # 步长缩放因子，用于线搜索

    def optimize(self, initial_state, initial_control_seq, goal_state, horizon):
        """
        执行ILQR优化。

        参数：
            initial_state (np.ndarray): 初始状态向量。
            initial_control_seq (np.ndarray): 初始控制序列，形状为 (horizon, control_dim)。
            goal_state (np.ndarray): 目标状态向量。
            horizon (int): 优化时间步数。

        返回：
            optimized_control_seq (np.ndarray): 优化后的控制序列，形状为 (horizon, control_dim)。
            trajectory (np.ndarray): 优化后的状态轨迹，形状为 (horizon + 1, state_dim)。
        """
        # 初始化变量
        n_states = initial_state.shape[0]
        n_controls = initial_control_seq.shape[1]
        control_seq = initial_control_seq.copy()
        state_seq = self.forward_pass(initial_state, control_seq, horizon)

        for iteration in range(self.max_iter):
            # 初始化价值函数的反馈和前馈项
            V_x = np.zeros(n_states)
            V_xx = np.zeros((n_states, n_states))

            # 初始化成本函数的终止条件
            V_x = self.cost_fn.final_cost_derivatives(state_seq[-1], goal_state)
            V_xx = self.cost_fn.final_cost_second_derivatives(state_seq[-1], goal_state)

            # 初始化为0
            k_seq = np.zeros((horizon, n_controls))
            K_seq = np.zeros((horizon, n_controls, n_states))

            # 反向传播计算反馈和前馈控制增量
            for t in reversed(range(horizon)):
                state = state_seq[t]
                control = control_seq[t]
                next_state = state_seq[t + 1]

                # 计算成本函数的一阶和二阶导数
                cx, cu, cxx, cuu, cxu = self.cost_fn.derivatives(state, control, next_state)

                # 获取系统动力学的一阶导数（雅可比矩阵）
                A, B = self.dynamics.derivatives(state, control)

                # 计算Q函数的一阶和二阶导数
                Q_x = cx + A.T @ V_x
                Q_u = cu + B.T @ V_x
                Q_xx = cxx + A.T @ V_xx @ A
                Q_ux = cxu + B.T @ V_xx @ A
                Q_uu = cuu + B.T @ V_xx @ B

                # 使用牛顿法计算增益
                try:
                    inv_Q_uu = np.linalg.inv(Q_uu)
                except np.linalg.LinAlgError:
                    print(f"Iteration {iteration}: Q_uu is not invertible.")
                    return control_seq, state_seq

                # 计算反馈和前馈增益
                K = -inv_Q_uu @ Q_ux
                k = -inv_Q_uu @ Q_u

                # 存储增益
                K_seq[t] = K
                k_seq[t] = k

                # 更新价值函数的导数
                V_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K

                # 确保对称性
                V_xx = 0.5 * (V_xx + V_xx.T)

            # 前向传播应用控制增量
            new_control_seq, success = self.forward_update(initial_state, control_seq, K_seq, k_seq, horizon)

            if not success:
                print(f"Iteration {iteration}: Line search failed.")
                break

            # 计算新的状态序列
            new_state_seq = self.forward_pass(initial_state, new_control_seq, horizon)

            # 计算总成本
            old_cost = self.total_cost(state_seq, control_seq, goal_state)
            new_cost = self.total_cost(new_state_seq, new_control_seq, goal_state)
            cost_reduction = old_cost - new_cost

            print(f"Iteration {iteration}: Cost reduction = {cost_reduction}")

            if cost_reduction < self.tol:
                print(f"Converged at iteration {iteration}.")
                break

            # 更新控制和状态序列
            control_seq = new_control_seq
            state_seq = new_state_seq

        # 最终优化的控制和状态序列
        optimized_control_seq = control_seq
        trajectory = state_seq

        return optimized_control_seq, trajectory

    def forward_pass(self, initial_state, control_seq, horizon):
        """
        计算给定控制序列下的状态轨迹。

        参数：
            initial_state (np.ndarray): 初始状态向量。
            control_seq (np.ndarray): 控制序列，形状为 (horizon, control_dim)。
            horizon (int): 时间步数。

        返回：
            state_seq (np.ndarray): 状态轨迹，形状为 (horizon + 1, state_dim)。
        """
        state_seq = np.zeros((horizon + 1, initial_state.shape[0]))
        state_seq[0] = initial_state

        for t in range(horizon):
            state_seq[t + 1] = self.dynamics.step(state_seq[t], control_seq[t])

        return state_seq

    def forward_update(self, initial_state, control_seq, K_seq, k_seq, horizon):
        """
        通过线搜索更新控制序列。

        参数：
            initial_state (np.ndarray): 初始状态向量。
            control_seq (np.ndarray): 当前控制序列，形状为 (horizon, control_dim)。
            K_seq (np.ndarray): 反馈增益序列，形状为 (horizon, control_dim, state_dim)。
            k_seq (np.ndarray): 前馈增益序列，形状为 (horizon, control_dim)。
            horizon (int): 时间步数。

        返回：
            new_control_seq (np.ndarray): 更新后的控制序列。
            success (bool): 是否成功找到合适的步长。
        """
        alpha = 1.0
        for step in range(10):
            new_control_seq = control_seq.copy()
            state = initial_state.copy()
            for t in range(horizon):
                delta_u = self.alpha * (k_seq[t] + K_seq[t] @ (state - state))
                new_control_seq[t] += delta_u
                state = self.dynamics.step(state, new_control_seq[t])

            # 评估新控制序列的成本是否减少
            new_cost = self.total_cost(self.forward_pass(initial_state, new_control_seq, horizon), new_control_seq, self.cost_fn.goal_state)
            old_cost = self.total_cost(self.forward_pass(initial_state, control_seq, horizon), control_seq, self.cost_fn.goal_state)

            if new_cost < old_cost:
                return new_control_seq, True
            else:
                self.alpha *= 0.5  # 缩小步长

        return control_seq, False

    def total_cost(self, state_seq, control_seq, goal_state):
        """
        计算总成本。

        参数：
            state_seq (np.ndarray): 状态轨迹，形状为 (horizon + 1, state_dim)。
            control_seq (np.ndarray): 控制序列，形状为 (horizon, control_dim)。
            goal_state (np.ndarray): 目标状态向量。

        返回：
            total_cost (float): 总成本。
        """
        total_cost = 0.0
        horizon = control_seq.shape[0]
        for t in range(horizon):
            total_cost += self.cost_fn.cost(state_seq[t], control_seq[t], state_seq[t + 1])
        # 添加终止成本
        total_cost += self.cost_fn.final_cost(state_seq[-1], goal_state)
        return total_cost

class UAVDynamics:
    def __init__(self, dt=0.1):
        """
        初始化无人机动力学模型。

        参数：
            dt (float): 时间步长。
        """
        self.dt = dt

    def step(self, state, control):
        """
        计算下一个状态。

        参数：
            state (np.ndarray): 当前状态向量。
            control (np.ndarray): 当前控制指令向量。

        返回：
            next_state (np.ndarray): 下一个状态向量。
        """
        # 假设状态向量包含位置 (x, y, z), 速度 (vx, vy, vz), 姿态 (roll, pitch, yaw)
        pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]

        thrust = control[0]
        br_cmd = control[1:4]  # 角速度指令

        # 更新位置和速度（简单积分）
        new_pos = pos + vel * self.dt
        new_vel = vel + np.array([0, 0, thrust]) * self.dt  # 假设推力只影响z轴

        # 更新姿态
        new_euler = euler + br_cmd * self.dt

        next_state = np.concatenate([new_pos, new_vel, new_euler])
        return next_state

    def derivatives(self, state, control):
        """
        计算状态转移方程的雅可比矩阵 A 和 B。

        参数：
            state (np.ndarray): 当前状态向量。
            control (np.ndarray): 当前控制指令向量。

        返回：
            A (np.ndarray): 状态对状态的雅可比矩阵。
            B (np.ndarray): 状态对控制的雅可比矩阵。
        """
        n_states = state.shape[0]
        n_controls = control.shape[0]

        A = np.eye(n_states)
        A[0:3, 3:6] = self.dt * np.eye(3)  # 位置对速度的影响

        # 速度对控制的影响
        B = np.zeros((n_states, n_controls))
        B[3:6, 0] = self.dt  # 推力对z轴速度的影响
        B[6:9, 1:4] = self.dt * np.eye(3)  # 角速度对姿态的影响

        return A, B

class UAVCostFunction:
    def __init__(self, goal_state, Q=None, R=None, Qf=None):
        """
        初始化成本函数。

        参数：
            goal_state (np.ndarray): 目标状态向量。
            Q (np.ndarray): 状态成本矩阵。
            R (np.ndarray): 控制成本矩阵。
            Qf (np.ndarray): 终止状态成本矩阵。
        """
        self.goal_state = goal_state
        self.Q = Q if Q is not None else np.eye(goal_state.shape[0])
        self.R = R if R is not None else np.eye(4)  # 假设控制维度为4
        self.Qf = Qf if Qf is not None else self.Q

    def cost(self, state, control, next_state):
        """
        计算即时成本。

        参数：
            state (np.ndarray): 当前状态向量。
            control (np.ndarray): 当前控制指令向量。
            next_state (np.ndarray): 下一个状态向量。

        返回：
            cost_value (float): 成本值。
        """
        state_error = state - self.goal_state
        cost_value = state_error.T @ self.Q @ state_error + control.T @ self.R @ control
        return cost_value

    def derivatives(self, state, control, next_state):
        """
        计算成本函数的一阶和二阶导数。

        参数：
            state (np.ndarray): 当前状态向量。
            control (np.ndarray): 当前控制指令向量。
            next_state (np.ndarray): 下一个状态向量。

        返回：
            (cx, cu, cxx, cuu, cxu)
        """
        state_error = state - self.goal_state

        cx = 2 * self.Q @ state_error
        cu = 2 * self.R @ control
        cxx = 2 * self.Q
        cuu = 2 * self.R
        cxu = np.zeros((self.Q.shape[0], self.R.shape[0]))  # 假设状态和控制独立

        return cx, cu, cxx, cuu, cxu

    def final_cost(self, state, goal_state):
        """
        计算终止成本。

        参数：
            state (np.ndarray): 终止状态向量。
            goal_state (np.ndarray): 目标状态向量。

        返回：
            cost_value (float): 终止成本值。
        """
        state_error = state - goal_state
        cost_value = state_error.T @ self.Qf @ state_error
        return cost_value

    def final_cost_derivatives(self, state, goal_state):
        """
        计算终止成本的一阶和二阶导数。

        参数：
            state (np.ndarray): 终止状态向量。
            goal_state (np.ndarray): 目标状态向量。

        返回：
            (cx, cxx)
        """
        state_error = state - self.goal_state

        cx = 2 * self.Qf @ state_error
        cxx = 2 * self.Qf

        return cx, cxx
