#!/usr/bin/env python3
"""
末端执行器姿态控制方法大全
展示所有可用的姿态控制方式
"""

import numpy as np
import mujoco
import sys
sys.path.append('./assets')

from curi1_control import find_arm_chain, EndEffectorController

def method_1_euler_angles():
    """方法1: 欧拉角姿态控制"""
    print("🔹 方法1: 欧拉角姿态控制")
    print("=" * 40)
    
    # 初始化
    model = mujoco.MjModel.from_xml_path("assets/bimanual_curi1_transfer_cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    chains = find_arm_chain()
    ee_controller = EndEffectorController(chains)
    
    # 目标位置
    target_position = np.array([-0.12, -0.65, 0.65])
    
    # 方式1a: 直接使用度数
    roll_deg, pitch_deg, yaw_deg = 0, 30, 15  # 度数
    target_orientation = np.radians([roll_deg, pitch_deg, yaw_deg])  # 转换为弧度
    
    print(f"目标位置: {target_position}")
    print(f"目标姿态: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")
    
    success, angles, info = ee_controller.move_to_pose(
        side="right",
        target_position=target_position,
        target_orientation=target_orientation,
        execute_motion=False
    )
    
    print(f"结果: {'✅成功' if success else '❌失败'}")
    if success:
        print(f"位置误差: {info.get('position_error', 0)*1000:.1f}mm")
    
    # 方式1b: 直接使用弧度
    target_orientation_rad = np.array([0, np.pi/6, np.pi/12])  # roll=0, pitch=30°, yaw=15°
    print(f"\n使用弧度: {target_orientation_rad}")
    
    return success

def method_2_quaternions():
    """方法2: 四元数姿态控制"""
    print("\n🔹 方法2: 四元数姿态控制")
    print("=" * 40)
    
    model = mujoco.MjModel.from_xml_path("assets/bimanual_curi1_transfer_cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    chains = find_arm_chain()
    ee_controller = EndEffectorController(chains)
    
    target_position = np.array([-0.12, -0.65, 0.65])
    
    # 方式2a: 预定义四元数
    # 绕Y轴旋转30度的四元数
    angle = np.pi/6  # 30度
    target_quat = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])  # [w, x, y, z]
    
    print(f"目标位置: {target_position}")
    print(f"目标四元数: [w={target_quat[0]:.3f}, x={target_quat[1]:.3f}, y={target_quat[2]:.3f}, z={target_quat[3]:.3f}]")
    
    success, angles, info = ee_controller.move_to_pose(
        side="right",
        target_position=target_position,
        target_orientation=target_quat,
        execute_motion=False
    )
    
    print(f"结果: {'✅成功' if success else '❌失败'}")
    
    # 方式2b: 使用工具函数转换
    def euler_to_quaternion(roll, pitch, yaw):
        """欧拉角转四元数"""
        try:
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler('xyz', [roll, pitch, yaw])
            quat_scipy = rot.as_quat()  # [x, y, z, w]
            return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # [w, x, y, z]
        except:
            # 简化计算（仅用于演示）
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            
            return np.array([w, x, y, z])
    
    # 从欧拉角转换
    converted_quat = euler_to_quaternion(0, np.pi/6, 0)  # 30度俯仰
    print(f"转换的四元数: {converted_quat}")
    
    return success

def method_3_current_pose_adjustment():
    """方法3: 基于当前姿态的增量调整"""
    print("\n🔹 方法3: 基于当前姿态的增量调整")
    print("=" * 40)
    
    model = mujoco.MjModel.from_xml_path("assets/bimanual_curi1_transfer_cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    chains = find_arm_chain()
    ee_controller = EndEffectorController(chains)
    
    # 获取当前状态
    current_pos = ee_controller.get_ee_pos("right")
    current_quat = ee_controller.get_ee_quat("right")
    
    print(f"当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    print(f"当前姿态: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
    
    # 方法3a: 在当前姿态基础上做小幅调整
    try:
        from scipy.spatial.transform import Rotation as R
        
        # 当前姿态转为旋转
        current_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
        
        # 增量旋转：绕Z轴旋转15度
        delta_rot = R.from_euler('z', np.pi/12)  # 15度
        new_rot = current_rot * delta_rot
        
        # 转回四元数
        quat_scipy = new_rot.as_quat()
        new_quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        
        print(f"增量调整: 绕Z轴旋转15度")
        print(f"新姿态: [{new_quat[0]:.3f}, {new_quat[1]:.3f}, {new_quat[2]:.3f}, {new_quat[3]:.3f}]")
        
        success, angles, info = ee_controller.move_to_pose(
            side="right",
            target_position=current_pos,  # 保持位置不变
            target_orientation=new_quat,
            execute_motion=False
        )
        
        print(f"结果: {'✅成功' if success else '❌失败'}")
        return success
        
    except ImportError:
        print("⚠️ scipy不可用，无法演示增量调整")
        return False

def method_4_predefined_poses():
    """方法4: 预定义常用姿态"""
    print("\n🔹 方法4: 预定义常用姿态")
    print("=" * 40)
    
    # 定义常用姿态库
    PREDEFINED_POSES = {
        "horizontal_down": {
            "euler": [np.pi, 0, 0],  # 水平向下 (抓取姿态)
            "description": "末端执行器朝下，适合从上方抓取"
        },
        "diagonal_45": {
            "euler": [np.pi, np.pi/4, 0],  # 45度倾斜
            "description": "45度倾斜，适合倾斜面操作"
        },
        "side_approach": {
            "euler": [np.pi/2, 0, 0],  # 侧向接近
            "description": "侧向接近，适合侧面抓取"
        },
        "upward": {
            "euler": [0, 0, 0],  # 向上
            "description": "末端执行器朝上"
        },
        "forward": {
            "euler": [np.pi/2, 0, np.pi/2],  # 前向
            "description": "向前伸展"
        }
    }
    
    model = mujoco.MjModel.from_xml_path("assets/bimanual_curi1_transfer_cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    chains = find_arm_chain()
    ee_controller = EndEffectorController(chains)
    
    target_position = np.array([-0.12, -0.65, 0.65])
    
    print("可用的预定义姿态:")
    success_count = 0
    
    for pose_name, pose_info in PREDEFINED_POSES.items():
        print(f"\n'{pose_name}': {pose_info['description']}")
        euler = pose_info['euler']
        print(f"   欧拉角: [{np.degrees(euler[0]):.0f}°, {np.degrees(euler[1]):.0f}°, {np.degrees(euler[2]):.0f}°]")
        
        success, angles, info = ee_controller.move_to_pose(
            side="right",
            target_position=target_position,
            target_orientation=euler,
            execute_motion=False
        )
        
        if success:
            print(f"   ✅ 可用，误差: {info.get('position_error', 0)*1000:.1f}mm")
            success_count += 1
        else:
            print(f"   ❌ 不可用 (IK失败)")
    
    print(f"\n总结: {success_count}/{len(PREDEFINED_POSES)} 个姿态可用")
    return success_count > 0

def method_5_smooth_trajectory():
    """方法5: 平滑轨迹姿态控制 (使用curi1_control_posekeys.py中的方法)"""
    print("\n🔹 方法5: 平滑轨迹姿态控制")
    print("=" * 40)
    
    print("这种方法在 curi1_control_posekeys.py 中已经实现:")
    print("使用 move_ee_to_pose() 函数可以实现:")
    print("- 位置和姿态的平滑插值")
    print("- 最小加加速度 (min-jerk) 轨迹")
    print("- 实时墙钟同步")
    print("- 四元数SLERP插值")
    
    print("\n调用方式:")
    print("move_ee_to_pose(controller, side, target_pos, target_quat, seconds=2.0)")
    
    print("参数:")
    print("- controller: TargetController实例") 
    print("- side: 'left' 或 'right'")
    print("- target_pos: 目标位置 [x, y, z]")
    print("- target_quat: 目标四元数 [w, x, y, z]")
    print("- seconds: 运动时间 (默认2秒)")

def method_6_interactive_triggers():
    """方法6: 交互式按键触发"""
    print("\n🔹 方法6: 交互式按键触发 (推荐)")
    print("=" * 40)
    
    print("在 curi1_control_posekeys.py 中使用按键触发:")
    
    triggers = [
        ("O键", "位置 + 欧拉角", "side x y z roll pitch yaw(deg) duration", "right -0.12 -0.65 0.65 0 30 0 2"),
        ("Q键", "位置 + 四元数", "side x y z qw qx qy qz duration", "left -0.2 -0.6 0.6 0.966 0 0.259 0 1.5"),
    ]
    
    for key, desc, format_str, example in triggers:
        print(f"\n{key}: {desc}")
        print(f"   格式: {format_str}")
        print(f"   示例: {example}")
    
    print(f"\n使用步骤:")
    print(f"1. python assets/curi1_control_posekeys.py")
    print(f"2. 等待MuJoCo窗口启动") 
    print(f"3. 保持终端焦点，按对应按键")
    print(f"4. 根据提示输入参数")

def usage_recommendations():
    """使用建议"""
    print(f"\n💡 **使用建议**")
    print("=" * 60)
    
    recommendations = [
        ("初学者", "方法6 (按键触发) + 方法4 (预定义姿态)", "最直观，容易上手"),
        ("开发者", "方法1 (欧拉角API) + 方法3 (增量调整)", "编程灵活，适合集成"),
        ("高精度", "方法2 (四元数) + 方法5 (平滑轨迹)", "最精确，适合复杂任务"),
        ("调试", "方法3 (增量调整)", "基于当前状态，成功率最高")
    ]
    
    for user_type, methods, reason in recommendations:
        print(f"\n🎯 {user_type}: {methods}")
        print(f"   原因: {reason}")
    
    print(f"\n🔧 **技术要点**:")
    print("- 小角度 (< 30°) 更容易成功")
    print("- 先位置控制，再姿态控制")
    print("- 使用execute_motion=False预验证")
    print("- 基于当前姿态的调整最稳定")
    print("- 欧拉角直观，四元数精确")

def main():
    """主演示程序"""
    print("🎯 末端执行器姿态控制方法大全")
    print("=" * 60)
    
    try:
        # 演示各种方法
        method_1_euler_angles()
        method_2_quaternions() 
        method_3_current_pose_adjustment()
        method_4_predefined_poses()
        method_5_smooth_trajectory()
        method_6_interactive_triggers()
        
        # 使用建议
        usage_recommendations()
        
        print(f"\n🎉 所有姿态控制方法演示完成!")
        print(f"\n🚀 立即试用: python assets/curi1_control_posekeys.py")
        print(f"然后按 'O' 键输入: right -0.12 -0.65 0.65 0 30 0 2")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()