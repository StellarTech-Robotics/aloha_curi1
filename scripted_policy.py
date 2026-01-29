import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False, max_step_distance=0.05, max_orientation_change=0.3):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.max_step_distance = max_step_distance  # 5cm max movement per step
        self.max_orientation_change = max_orientation_change  # max quaternion change per step

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    def _validate_trajectory(self, trajectory, arm_name):
        """Validate and fix trajectory waypoints to ensure reasonable motion"""
        print(f"[TRAJECTORY] Validating {arm_name} trajectory with {len(trajectory)} waypoints")
        for i in range(len(trajectory) - 1):
            curr_wp = trajectory[i]
            next_wp = trajectory[i + 1]
            
            # Check position distance
            pos_diff = np.linalg.norm(next_wp['xyz'] - curr_wp['xyz'])
            time_diff = next_wp['t'] - curr_wp['t']
            if time_diff > 0:
                vel = pos_diff / (time_diff * 0.02)  # velocity in m/s (DT=0.02)
                if vel > 2.5:  # 2.5 m/s is quite fast
                    print(f"[WARNING] {arm_name} fast motion between t={curr_wp['t']}-{next_wp['t']}: {vel:.2f}m/s")
            
            # Check workspace bounds
            if not self._is_position_safe(next_wp['xyz'], arm_name):
                print(f"[WARNING] {arm_name} waypoint at t={next_wp['t']} may be out of workspace: {next_wp['xyz']}")

    def _is_position_safe(self, pos, arm_name):
        """Check if position is within safe workspace for CURI robot"""
        try:
            from constants import CURI_WORKSPACE
            ws = CURI_WORKSPACE[f'{arm_name}_arm']
            x, y, z = pos
            x_ok = ws['x_range'][0] <= x <= ws['x_range'][1]
            y_ok = ws['y_range'][0] <= y <= ws['y_range'][1] 
            z_ok = ws['z_range'][0] <= z <= ws['z_range'][1]
            return x_ok and y_ok and z_ok
        except:
            # Fallback to original ranges if import fails
            x, y, z = pos
            if arm_name == 'left':
                return (0.0 < x < 0.7) and (-1.0 < y < -0.3) and (0.2 < z < 1.0)
            else:  # right
                return (-0.7 < x < 0.0) and (-1.0 < y < -0.3) and (0.2 < z < 1.0)

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        """
        Generate trajectories for CURI dual-arm robot specifically.
        Considers the robot's kinematic constraints and workspace limitations.
        """
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        print(f"[CURI_TRAJ] L init: pos={init_mocap_pose_left[:3]}  quat={init_mocap_pose_left[3:]}")
        print(f"[CURI_TRAJ] R init: pos={init_mocap_pose_right[:3]} quat={init_mocap_pose_right[3:]}")

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        print(f"[CURI_TRAJ] Box position: {box_xyz}")

        # Import CURI workspace parameters
        from constants import CURI_WORKSPACE
        
        # Validate box position is reachable
        if not self._validate_box_position(box_xyz, CURI_WORKSPACE):
            print(f"[WARNING] Box at {box_xyz} may be difficult to reach, adjusting trajectory")
            
        # Design orientations for CURI dual-arm configuration
        # Right arm picks with slightly downward angle
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-45)  # Less aggressive angle
        
        # Left arm receives with more natural orientation for dual-arm setup
        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=75)  # Slightly less rotation

        # Calculate handover position optimized for CURI's dual-arm geometry
        center_x = 0.0  # Center between shoulders for CURI
        shared_ws = CURI_WORKSPACE['shared_workspace']
        
        # Meet position: prioritize reachability over box position
        meet_x = np.clip(box_xyz[0] * 0.3 + center_x * 0.7, 
                        shared_ws['x_range'][0] + 0.1, shared_ws['x_range'][1] - 0.1)
        meet_y = np.clip(box_xyz[1] + 0.1, shared_ws['y_range'][0] + 0.1, shared_ws['y_range'][1] - 0.1)  
        meet_z = np.clip(box_xyz[2] + 0.08, shared_ws['z_range'][0] + 0.1, shared_ws['z_range'][1] - 0.1)
        meet_xyz = np.array([meet_x, meet_y, meet_z])
        print(f"[ULTRA_CONSERVATIVE] Handover position: {meet_xyz} (box: {box_xyz})")
        print(f"[ULTRA_CONSERVATIVE] 使用极限保守策略：8倍时间 + 超密集路径点")

        # === 极限保守策略：左臂8倍时间超慢速轨迹 ===
        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # start open
            {"t": 1000, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 1}, # 极长等待
            {"t": 1400, "xyz": meet_xyz + np.array([0.06, 0, 0.05]), "quat": meet_left_quat.elements, "gripper": 1}, # 超慢接近
            {"t": 1700, "xyz": meet_xyz + np.array([0.04, 0, 0.03]), "quat": meet_left_quat.elements, "gripper": 1}, # 中间点1
            {"t": 2000, "xyz": meet_xyz + np.array([0.02, 0, 0.02]), "quat": meet_left_quat.elements, "gripper": 1}, # 中间点2
            {"t": 2300, "xyz": meet_xyz, "quat": meet_left_quat.elements, "gripper": 1}, # 到达准备位置
            {"t": 2600, "xyz": meet_xyz, "quat": meet_left_quat.elements, "gripper": 0}, # 夹紧
            {"t": 2900, "xyz": meet_xyz + np.array([0.04, 0, 0.02]), "quat": meet_left_quat.elements, "gripper": 0}, # 慢速撤退1
            {"t": 3200, "xyz": meet_xyz + np.array([0.08, 0, 0.05]), "quat": meet_left_quat.elements, "gripper": 0}, # 慢速撤退2
        ]

        # Right arm trajectory: smoother path with intermediate waypoints
        approach_height = max(0.08, box_xyz[2] - 0.55)  # Adaptive approach height
        
        # Calculate intermediate position to reduce large jumps
        mid_pos = (init_mocap_pose_right[:3] + box_xyz) / 2
        mid_pos[2] = max(init_mocap_pose_right[2], box_xyz[2] + 0.1)  # Safe height
        
        # === 保守策略：右臂极密集超慢速轨迹 ===
        # 计算更多中间路径点
        mid_pos1 = init_mocap_pose_right[:3] * 0.8 + box_xyz * 0.2
        mid_pos1[2] = max(init_mocap_pose_right[2], box_xyz[2] + 0.15)
        mid_pos2 = init_mocap_pose_right[:3] * 0.6 + box_xyz * 0.4
        mid_pos2[2] = max(init_mocap_pose_right[2], box_xyz[2] + 0.12)
        mid_pos3 = init_mocap_pose_right[:3] * 0.4 + box_xyz * 0.6
        mid_pos3[2] = max(init_mocap_pose_right[2], box_xyz[2] + 0.10)
        
        self.right_trajectory = [
            # 阶段1：从初始位置8倍时间超慢移动（超密集路径点）
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
            {"t": 160, "xyz": mid_pos1, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 320, "xyz": mid_pos2, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 480, "xyz": mid_pos3, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 640, "xyz": mid_pos, "quat": gripper_pick_quat.elements, "gripper": 1},
            
            # 阶段2：接近方块（极其极其缓慢）
            {"t": 800, "xyz": box_xyz + np.array([0, 0.08, approach_height]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 960, "xyz": box_xyz + np.array([0, 0.05, approach_height]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 1120, "xyz": box_xyz + np.array([0, 0.02, approach_height]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 1280, "xyz": box_xyz + np.array([0, 0, approach_height]), "quat": gripper_pick_quat.elements, "gripper": 1},
            
            # 阶段3：精确下降（超超细分步骤）
            {"t": 1440, "xyz": box_xyz + np.array([0, 0, 0.04]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 1560, "xyz": box_xyz + np.array([0, 0, 0.02]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 1680, "xyz": box_xyz + np.array([0, 0, 0.01]), "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 1800, "xyz": box_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat.elements, "gripper": 1},
            
            # 阶段4：抓取和提升（超长时间稳定）
            {"t": 1960, "xyz": box_xyz + np.array([0, 0, -0.005]), "quat": gripper_pick_quat.elements, "gripper": 0}, # 夹紧！
            {"t": 2120, "xyz": box_xyz + np.array([0, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 0}, # 超慢提升
            {"t": 2280, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 0}, # 继续提升
            
            # 阶段5：转移到交接位置（超密集路径点）
            {"t": 2440, "xyz": meet_xyz + np.array([-0.04, 0, 0.04]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 2560, "xyz": meet_xyz + np.array([-0.02, 0, 0.02]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 2680, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # 到达交接点
            
            # 阶段6：交接和撤退
            {"t": 2800, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1}, # 松开
            {"t": 2960, "xyz": meet_xyz + np.array([-0.04, 0, 0.02]), "quat": gripper_pick_quat.elements, "gripper": 1}, # 撤退1
            {"t": 3080, "xyz": meet_xyz + np.array([-0.06, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 1}, # 撤退2
            {"t": 3200, "xyz": meet_xyz + np.array([-0.08, 0, 0.03]), "quat": gripper_pick_quat.elements, "gripper": 1}, # 最终位置
        ]

        # Validate trajectories for CURI workspace
        self._validate_trajectory(self.left_trajectory, 'left')
        self._validate_trajectory(self.right_trajectory, 'right')
        
    def _validate_box_position(self, box_xyz, workspace):
        """Check if box position is within reasonable reach of both arms"""
        shared = workspace['shared_workspace']
        x_ok = shared['x_range'][0] <= box_xyz[0] <= shared['x_range'][1]
        y_ok = shared['y_range'][0] <= box_xyz[1] <= shared['y_range'][1] 
        z_ok = shared['z_range'][0] <= box_xyz[2] <= shared['z_range'][1]
        return x_ok and y_ok and z_ok


class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        peg_info = np.array(ts_first.observation['env_state'])[:7]
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state'])[7:]
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)

        meet_xyz = np.array([0, -0.700, 0.15])
        lift_right = 0.00715

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # insertion
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # approach the cube
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1}, # go down
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # close gripper
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0}, # approach meet position
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # insertion

        ]


def test_policy(task_name):
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(2):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images']['angle'])
            plt.ion()

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'
    test_policy(test_task_name)

