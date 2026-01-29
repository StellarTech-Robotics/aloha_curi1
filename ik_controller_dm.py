#!/usr/bin/env python3
"""强化型IK控制器，支持位置+姿态控制，优化性能和稳定性"""

import numpy as np
import time
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN

class IKController:
    """强化型IK控制器，支持6DOF控制并优化性能"""
    
    def __init__(self, physics):
        self.physics = physics
        self.chains = self._find_arm_chain()
        self.joint_to_actuator = self._find_joint_actuators()
        
        # TCP偏移量：从rmg42_base_link到抓取点的偏移
        self.tcp_offset = {
            "left": np.array([0.0, 0.0, 0.0891]),   
            "right": np.array([0.0, 0.0, 0.0891])   
        }
        
        # 工作空间限制（基于实际测试调整）
        self.workspace_limits = {
            "left": {
                'x': [-0.2, 0.8], 'y': [-1.0, -0.2], 'z': [0.3, 1.1]
            },
            "right": {
                'x': [-0.8, 0.2], 'y': [-1.0, -0.2], 'z': [0.3, 1.1]
            }
        }
        
        # 性能统计
        self.stats = {'ik_calls': 0, 'success_rate': 0, 'avg_time': 0, 'convergence_rate': 0}
        
    def _find_arm_chain(self):
        """找到左右臂的关节链"""
        import mujoco
        
        LEFT_BASE_BODY = "l_base_link1"
        RIGHT_BASE_BODY = "r_base_link1"
        LEFT_EE_BODY = "l_rmg42_base_link"
        RIGHT_EE_BODY = "r_rmg42_base_link"
        LEFT_JOINT_PREFIX = "l_joint"
        RIGHT_JOINT_PREFIX = "r_joint"
        
        def name_to_id(name, obj_type):
            try:
                return self.physics.model.name2id(name, obj_type)
            except Exception:
                return -1
        
        left, right = [], []
        
        # 遍历关节并获取名称
        for j in range(self.physics.model.njnt):
            try:
                name = mujoco.mj_id2name(self.physics.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            except Exception:
                name = ""
            
            if name.startswith(LEFT_JOINT_PREFIX):
                left.append((j, name))
            elif name.startswith(RIGHT_JOINT_PREFIX):
                right.append((j, name))
        
        def sort6(pairs):
            def k(p):
                j, name = p
                num = "".join([c for c in name if c.isdigit()])
                return int(num) if num else 999
            return [j for j,_ in sorted(pairs, key=k)][:6]
        
        l_ids = sort6(left)
        r_ids = sort6(right)
        l_qadr = [self.physics.model.jnt_qposadr[j] for j in l_ids]
        r_qadr = [self.physics.model.jnt_qposadr[j] for j in r_ids]
        
        print(f"左臂关节 IDs: {l_ids}, 地址: {l_qadr}")
        print(f"右臂关节 IDs: {r_ids}, 地址: {r_qadr}")
        
        return {
            "left":  {"base": name_to_id(LEFT_BASE_BODY, 'body'),  
                      "ee": name_to_id(LEFT_EE_BODY, 'body'),  
                      "jids": l_ids, "qadr": l_qadr},
            "right": {"base": name_to_id(RIGHT_BASE_BODY, 'body'), 
                      "ee": name_to_id(RIGHT_EE_BODY, 'body'), 
                      "jids": r_ids, "qadr": r_qadr},
        }
    
    def _find_joint_actuators(self):
        """建立关节到执行器的映射"""
        joint_to_actuator = {}
        for aid in range(self.physics.model.nu):
            if self.physics.model.actuator_trntype[aid] == 0:  # joint transmission
                jid = self.physics.model.actuator_trnid[aid, 0]
                if jid >= 0:
                    joint_to_actuator[jid] = aid
        return joint_to_actuator
    
    def numeric_jac(self, side, eps=1e-5):
        """计算数值雅可比矩阵"""
        c = self.chains[side]
        J = np.zeros((3, len(c["qadr"])))
        qbackup = self.physics.data.qpos.copy()
        self.physics.forward()
        
        for k, adr in enumerate(c["qadr"]):
            self.physics.data.qpos[:] = qbackup
            self.physics.data.qpos[adr] += eps
            self.physics.forward()
            p_plus = self.physics.data.xpos[c["ee"]].copy()
            
            self.physics.data.qpos[:] = qbackup
            self.physics.data.qpos[adr] -= eps
            self.physics.forward()
            p_minus = self.physics.data.xpos[c["ee"]].copy()
            
            J[:,k] = (p_plus - p_minus) / (2*eps)
        
        self.physics.data.qpos[:] = qbackup
        self.physics.forward()
        return J
    
    def numeric_jac_6dof(self, side, q=None):
        """计算6DOF数值雅可比矩阵（位置+姿态）"""
        c = self.chains[side]
        if len(c["jids"]) == 0:
            return np.zeros((6, 6))
        
        eps = 1e-6
        J = np.zeros((6, len(c["jids"])))
        
        # 如果提供了q，先设置关节角度
        qpos_backup = self.physics.data.qpos.copy()
        if q is not None:
            for i, qadr in enumerate(c["qadr"]):
                self.physics.data.qpos[qadr] = q[i]
            self.physics.forward()
        
        # 获取基准状态
        tcp_pos0 = self.get_ee_pos(side)
        tcp_quat0 = self.get_ee_quat(side)
        
        for i, qadr in enumerate(c["qadr"]):
            # 正向扰动
            self.physics.data.qpos[qadr] += eps
            self.physics.forward()
            
            tcp_pos1 = self.get_ee_pos(side)
            tcp_quat1 = self.get_ee_quat(side)
            
            # 位置雅可比
            J[:3, i] = (tcp_pos1 - tcp_pos0) / eps
            
            # 姿态雅可比
            try:
                from scipy.spatial.transform import Rotation as R
                rot0 = R.from_quat([tcp_quat0[1], tcp_quat0[2], tcp_quat0[3], tcp_quat0[0]])
                rot1 = R.from_quat([tcp_quat1[1], tcp_quat1[2], tcp_quat1[3], tcp_quat1[0]])
                relative_rot = rot1 * rot0.inv()
                axis_angle_diff = relative_rot.as_rotvec()
                J[3:6, i] = axis_angle_diff / eps
            except:
                J[3:6, i] = 0
            
            # 恢复状态
            self.physics.data.qpos[qadr] -= eps
            self.physics.forward()
        
        # 恢复原始状态
        self.physics.data.qpos[:] = qpos_backup
        self.physics.forward()
        
        return J
    
    def check_workspace_reachability(self, side, target_pos):
        """检查目标位置是否在工作空间内"""
        limits = self.workspace_limits[side]
        
        # 基础边界检查
        if not (limits['x'][0] <= target_pos[0] <= limits['x'][1] and
                limits['y'][0] <= target_pos[1] <= limits['y'][1] and
                limits['z'][0] <= target_pos[2] <= limits['z'][1]):
            return False, "超出工作空间边界"
        
        # 计算到肩关节的距离
        if side == "left":
            shoulder_pos = np.array([0.1, -0.1103, 0.187 + 0.031645])
        else:
            shoulder_pos = np.array([-0.1, -0.1103, 0.187 + 0.031645])
        
        distance = np.linalg.norm(target_pos - shoulder_pos)
        max_reach = 0.8  # 增加最大臂长限制
        min_reach = 0.1  # 最小半径
        
        if distance > max_reach:
            return False, f"距离过远: {distance:.3f} > {max_reach:.3f}"
        if distance < min_reach:
            return False, f"距离过近: {distance:.3f} < {min_reach:.3f}"
        
        return True, "可达"

    def solve_ik(self, side, target_tcp_pos, target_tcp_quat=None, max_iters=50):
        """
        强化型IK求解器，支持末端执行器位置+姿态控制
        
        Args:
            side: "left" or "right" 
            target_tcp_pos: 目标末端执行器位置 [x, y, z]
            target_tcp_quat: 目标末端执行器姿态四元数 [w, x, y, z] (可选)
            max_iters: 最大迭代次数
            
        Returns:
            target_joint_pos: 关节角度解
        """
        start_time = time.time()
        self.stats['ik_calls'] += 1
        
        # 1. 工作空间可达性检查
        reachable, reason = self.check_workspace_reachability(side, target_tcp_pos)
        if not reachable:
            print(f"[IK WARNING] {side}臂目标不可达: {reason}")
            return np.zeros(len(self.chains[side]["qadr"]))
        
        c = self.chains[side]
        temp_qpos = self.physics.data.qpos.copy()
        target_joint_pos = np.zeros(len(c["qadr"]))
        
        # 2. 生成多个初始猜测
        initial_guesses = self._generate_initial_guesses(side, target_tcp_pos)
        best_solution = None
        best_error = float('inf')
        
        for guess_idx, initial_q in enumerate(initial_guesses):
            # 设置初始猜测
            for i, qadr in enumerate(c["qadr"]):
                self.physics.data.qpos[qadr] = initial_q[i]
            self.physics.forward()
            
            converged = False
            
            # 3. 迭代求解IK
            for iteration in range(max_iters):
                # 获取当前末端执行器位置和姿态
                cur_tcp_pos = self.get_ee_pos(side)
                cur_tcp_quat = self.get_ee_quat(side)
                
                # 计算位置误差
                pos_error = target_tcp_pos - cur_tcp_pos
                pos_error_norm = np.linalg.norm(pos_error)
                
                # 计算姿态误差（如果提供了目标姿态）
                if target_tcp_quat is not None:
                    try:
                        from scipy.spatial.transform import Rotation as R
                        cur_rot = R.from_quat([cur_tcp_quat[1], cur_tcp_quat[2], cur_tcp_quat[3], cur_tcp_quat[0]])
                        target_rot = R.from_quat([target_tcp_quat[1], target_tcp_quat[2], target_tcp_quat[3], target_tcp_quat[0]])
                        
                        relative_rot = target_rot * cur_rot.inv()
                        axis_angle = relative_rot.as_rotvec()
                        ori_error_norm = np.linalg.norm(axis_angle)
                        
                        # 自适应权重：位置误差大时优先位置，位置误差小时考虑姿态
                        pos_weight = 1.0
                        ori_weight = min(0.5, 0.1 / (pos_error_norm + 0.001))
                        
                        combined_error = np.concatenate([pos_weight * pos_error, ori_weight * axis_angle])
                        total_error_norm = max(pos_error_norm, ori_weight * ori_error_norm)
                    except:
                        combined_error = pos_error
                        total_error_norm = pos_error_norm
                else:
                    combined_error = pos_error
                    total_error_norm = pos_error_norm
                
                # 4. 收敛检查 - 更严格的精度要求
                convergence_threshold = 0.0005  # 0.5mm精度
                if total_error_norm < convergence_threshold:
                    converged = True
                    break
                
                # 5. 计算雅可比矩阵
                current_q = np.array([self.physics.data.qpos[qadr] for qadr in c["qadr"]])
                if target_tcp_quat is not None and len(combined_error) == 6:
                    J = self.numeric_jac_6dof(side, current_q)
                else:
                    J = self.numeric_jac(side)
                    combined_error = pos_error
                
                # 6. 阻尼最小二乘法求解，自适应阻尼
                lambda_base = 0.001
                lambda_adaptive = lambda_base * (1 + np.exp(-iteration/10))  # 初期大阻尼，后期小阻尼
                
                JTJ = J.T @ J
                damped_JTJ = JTJ + lambda_adaptive * np.eye(JTJ.shape[0])
                
                try:
                    dq = np.linalg.solve(damped_JTJ, J.T @ combined_error)
                except np.linalg.LinAlgError:
                    # 奇异性处理：使用SVD分解
                    dq = np.linalg.pinv(J, rcond=1e-6) @ combined_error
                
                # 7. 自适应步长控制
                step_scale = min(1.0, 0.1 / (total_error_norm + 0.001))  # 误差大时小步长
                max_step = 0.1 * (1.0 - iteration / max_iters)  # 逐渐减小步长
                dq = step_scale * np.clip(dq, -max_step, max_step)
                
                # 8. 更新关节角度并应用限制
                for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                    new_q = self.physics.data.qpos[qadr] + dq[i]
                    rmin, rmax = self.physics.model.jnt_range[jid]
                    if rmin < rmax:
                        new_q = np.clip(new_q, rmin, rmax)
                    self.physics.data.qpos[qadr] = new_q
                
                self.physics.forward()
            
            # 记录最佳解
            if converged and total_error_norm < best_error:
                best_error = total_error_norm
                for i, qadr in enumerate(c["qadr"]):
                    target_joint_pos[i] = self.physics.data.qpos[qadr]
                best_solution = target_joint_pos.copy()
                
                # 如果精度足够好，直接采用
                if best_error < 0.0001:
                    break
        
        # 9. 恢复原始状态
        self.physics.data.qpos[:] = temp_qpos
        self.physics.forward()
        
        # 10. 更新统计信息
        solve_time = time.time() - start_time
        success = best_solution is not None
        
        if success:
            self.stats['success_rate'] = (self.stats['success_rate'] * (self.stats['ik_calls'] - 1) + 1) / self.stats['ik_calls']
            self.stats['convergence_rate'] = (self.stats['convergence_rate'] * (self.stats['ik_calls'] - 1) + 1) / self.stats['ik_calls']
            print(f"[IK SUCCESS] {side}臂 误差:{best_error:.6f} 用时:{solve_time:.3f}s 尝试:{guess_idx+1}")
            return best_solution
        else:
            self.stats['success_rate'] = (self.stats['success_rate'] * (self.stats['ik_calls'] - 1)) / self.stats['ik_calls']
            print(f"[IK FAILED] {side}臂 所有初始猜测均失败 用时:{solve_time:.3f}s")
            return np.zeros(len(c["qadr"]))
    
    def _generate_initial_guesses(self, side, target_pos):
        """生成多个智能初始猜测"""
        c = self.chains[side]
        guesses = []
        
        # 1. 当前关节位置
        if len(c["qadr"]) > 0:
            current_q = np.array([self.physics.data.qpos[qadr] for qadr in c["qadr"]])
        else:
            current_q = np.zeros(6)  # 如果找不到关节，使用零位
        guesses.append(current_q)
        
        # 2. 零位
        guesses.append(np.zeros(6))
        
        # 3. 基于几何的智能猜测
        if side == "left":
            shoulder_pos = np.array([0.1, -0.1103, 0.187 + 0.031645])
        else:
            shoulder_pos = np.array([-0.1, -0.1103, 0.187 + 0.031645])
        
        to_target = target_pos - shoulder_pos
        
        # 几何初始猜测
        q1 = np.arctan2(to_target[1], to_target[0])  # base rotation
        r_xy = np.linalg.norm(to_target[:2])
        q2 = np.arctan2(-to_target[2], r_xy)  # shoulder angle
        
        geometric_guess = np.array([q1, q2, 0, 0, 0, 0])
        
        # 应用关节限制
        for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
            rmin, rmax = self.physics.model.jnt_range[jid]
            if rmin < rmax:
                geometric_guess[i] = np.clip(geometric_guess[i], rmin, rmax)
        
        guesses.append(geometric_guess)
        
        # 4. 添加随机扰动的猜测
        for _ in range(2):
            noise = np.random.normal(0, 0.2, 6)  # 小幅随机扰动
            noisy_guess = current_q + noise
            
            # 应用关节限制
            for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
                rmin, rmax = self.physics.model.jnt_range[jid]
                if rmin < rmax:
                    noisy_guess[i] = np.clip(noisy_guess[i], rmin, rmax)
            
            guesses.append(noisy_guess)
        
        return guesses
    
    def set_joint_targets(self, side, joint_positions):
        """设置关节目标位置"""
        c = self.chains[side]
        
        for i, (jid, qadr) in enumerate(zip(c["jids"], c["qadr"])):
            if i < len(joint_positions):
                # 设置关节位置
                self.physics.data.qpos[qadr] = joint_positions[i]
                
                # 设置执行器控制
                if jid in self.joint_to_actuator:
                    aid = self.joint_to_actuator[jid]
                    clo, chi = self.physics.model.actuator_ctrlrange[aid]
                    self.physics.data.ctrl[aid] = np.clip(joint_positions[i], clo, chi)
    
    def set_ee_target(self, side, target_pos):
        """设置末端执行器目标位置并求解IK（仅位置控制）"""
        joint_positions = self.solve_ik(side, target_pos)
        self.set_joint_targets(side, joint_positions)
        return joint_positions
    
    def set_ee_target_with_orientation(self, side, target_pos, target_quat):
        """设置末端执行器目标位置和姿态并求解IK"""
        joint_positions = self.solve_ik(side, target_pos, target_quat)
        self.set_joint_targets(side, joint_positions)
        return joint_positions
    
    def set_gripper_target(self, side, gripper_value):
        """设置夹爪目标位置 - 修正版本"""
        # 找到夹爪关节和执行器
        if side == "left":
            gripper_joints = ["Joint_finger1", "Joint_finger2"]
        else:
            gripper_joints = ["r_Joint_finger1", "r_Joint_finger2"]
        
        # 正确的关节值映射：
        # gripper_value: 0.0=闭合, 1.0=张开
        # 关节范围: 0到0.0325 (0=闭合, 0.0325=张开)
        joint_value = gripper_value * 0.0325  # 直接映射到关节范围
        
        print(f"[GRIPPER DEBUG] {side} gripper_value={gripper_value:.3f} -> joint_value={joint_value:.6f}")
        
        for i, joint_name in enumerate(gripper_joints):
            try:
                jid = self.physics.model.name2id(joint_name, 'joint')
                qadr = self.physics.model.jnt_qposadr[jid]
                
                # 直接设置关节位置（不再使用constants中的错误映射）
                self.physics.data.qpos[qadr] = joint_value
                
                # 设置执行器控制（使用执行器的正确范围）
                if jid in self.joint_to_actuator:
                    aid = self.joint_to_actuator[jid]
                    # 获取执行器的实际范围
                    ctrl_low, ctrl_high = self.physics.model.actuator_ctrlrange[aid]
                    
                    # 将关节值映射到执行器范围
                    if i == 0:  # finger1
                        # 执行器范围是0.021到0.057
                        ctrl_value = ctrl_low + (joint_value / 0.0325) * (ctrl_high - ctrl_low)
                    else:  # finger2  
                        # 执行器范围是-0.057到-0.021
                        ctrl_value = ctrl_high + (joint_value / 0.0325) * (ctrl_low - ctrl_high)
                    
                    self.physics.data.ctrl[aid] = ctrl_value
                    print(f"[GRIPPER DEBUG] {joint_name}: joint={joint_value:.6f}, ctrl={ctrl_value:.6f} (range: {ctrl_low:.3f} to {ctrl_high:.3f})")
                    
            except Exception as e:
                print(f"[GRIPPER ERROR] {joint_name}: {e}")
                continue
    
    def get_ee_pos(self, side):
        """获取当前TCP位置（抓取点位置）"""
        c = self.chains[side]
        if c["ee"] >= 0:
            # 获取base_link位置和姿态
            base_pos = self.physics.data.xpos[c["ee"]].copy()
            base_quat = self.physics.data.xquat[c["ee"]].copy()
            
            # 计算TCP位置（base_link + 偏移）
            from scipy.spatial.transform import Rotation as R
            try:
                # 将四元数转换为旋转矩阵并应用偏移
                rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
                rotated_offset = rot.apply(self.tcp_offset[side])
                tcp_pos = base_pos + rotated_offset
                return tcp_pos
            except:
                # 如果scipy不可用，使用简化的偏移
                tcp_pos = base_pos + self.tcp_offset[side]
                return tcp_pos
        else:
            # fallback - 加上TCP偏移
            if side == "left":
                return np.array([0.42, -0.53, 0.65 + 0.0891])
            else:
                return np.array([-0.42, -0.53, 0.65 + 0.0891])
    
    def get_ee_quat(self, side):
        """获取当前末端执行器姿态"""
        c = self.chains[side]
        if c["ee"] >= 0:
            return self.physics.data.xquat[c["ee"]].copy()
        else:
            # fallback
            return np.array([1, 0, 0, 0])
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置性能统计"""
        self.stats = {'ik_calls': 0, 'success_rate': 0, 'avg_time': 0, 'convergence_rate': 0}
    
    def move_ee_to_pose(self, side, target_position, target_orientation=None, execute_motion=True):
        """
        指定机械臂末端执行器移动到指定位置和姿态
        
        Args:
            side: "left" or "right" - 指定左臂或右臂
            target_position: [x, y, z] - 目标位置 (米)
            target_orientation: [w, x, y, z] 或 [roll, pitch, yaw] - 目标姿态 (可选)
                              如果是3元素则认为是欧拉角，如果是4元素则认为是四元数
            execute_motion: bool - 是否执行运动，False时仅计算关节角度
            
        Returns:
            success: bool - 是否成功
            joint_angles: np.array - 关节角度解
            info: dict - 详细信息
        """
        print(f"\n🎯 控制 {side} 臂末端执行器运动")
        print(f"目标位置: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # 处理姿态参数
        target_quat = None
        if target_orientation is not None:
            if len(target_orientation) == 3:  # 欧拉角 [roll, pitch, yaw]
                from scipy.spatial.transform import Rotation as R
                rot = R.from_euler('xyz', target_orientation)
                target_quat = rot.as_quat()  # [x, y, z, w]
                # 转换为MuJoCo格式 [w, x, y, z]
                target_quat = np.array([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
                print(f"目标姿态 (欧拉角): [{target_orientation[0]:.3f}, {target_orientation[1]:.3f}, {target_orientation[2]:.3f}] rad")
            elif len(target_orientation) == 4:  # 四元数
                target_quat = np.array(target_orientation)
                print(f"目标姿态 (四元数): [{target_quat[0]:.3f}, {target_quat[1]:.3f}, {target_quat[2]:.3f}, {target_quat[3]:.3f}]")
        else:
            print("目标姿态: 未指定 (仅位置控制)")
        
        # 获取当前末端执行器状态
        current_pos = self.get_ee_pos(side)
        current_quat = self.get_ee_quat(side)
        print(f"当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        print(f"当前姿态: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
        
        # 求解IK
        start_time = time.time()
        joint_angles = self.solve_ik(side, target_position, target_quat, max_iters=50)
        solve_time = time.time() - start_time
        
        # 检查求解是否成功
        success = np.any(joint_angles != 0)  # 如果全零说明失败
        
        if success:
            print(f"✅ IK求解成功，用时 {solve_time:.3f}s")
            print(f"解算关节角度: {np.degrees(joint_angles)} 度")
            
            if execute_motion:
                # 执行运动
                print("🤖 执行末端执行器运动...")
                self.set_joint_targets(side, joint_angles)
                self.physics.forward()
                
                # 验证最终位置和姿态
                final_pos = self.get_ee_pos(side)
                final_quat = self.get_ee_quat(side)
                
                pos_error = np.linalg.norm(final_pos - target_position)
                print(f"最终位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
                print(f"位置误差: {pos_error*1000:.1f} mm")
                
                if target_quat is not None:
                    print(f"最终姿态: [{final_quat[0]:.3f}, {final_quat[1]:.3f}, {final_quat[2]:.3f}, {final_quat[3]:.3f}]")
                    # 计算姿态误差
                    try:
                        from scipy.spatial.transform import Rotation as R
                        final_rot = R.from_quat([final_quat[1], final_quat[2], final_quat[3], final_quat[0]])
                        target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
                        relative_rot = target_rot * final_rot.inv()
                        angle_error = np.linalg.norm(relative_rot.as_rotvec())
                        print(f"姿态误差: {np.degrees(angle_error):.1f} 度")
                    except:
                        print("姿态误差: 无法计算")
                
                info = {
                    'solve_time': solve_time,
                    'final_position': final_pos,
                    'final_orientation': final_quat,
                    'position_error': pos_error,
                    'joint_angles_deg': np.degrees(joint_angles)
                }
            else:
                print("💡 仅计算IK解，未执行运动")
                info = {
                    'solve_time': solve_time,
                    'joint_angles_deg': np.degrees(joint_angles)
                }
        else:
            print(f"❌ IK求解失败，用时 {solve_time:.3f}s")
            info = {'solve_time': solve_time, 'error': 'IK求解失败'}
        
        return success, joint_angles, info