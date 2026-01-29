#!/usr/bin/env python3
"""
演示如何在curi1_control.py中使用按键触发位置姿态输入
"""

import numpy as np
import sys
sys.path.append('./assets')

from curi1_control import find_arm_chain, EndEffectorController
import mujoco

def demo_keyboard_triggers():
    """演示按键触发的位置姿态控制功能"""
    print("🎮 curi1_control.py 按键触发功能演示")
    print("=" * 60)
    
    # 加载模型和控制器 (模拟实际环境)
    model = mujoco.MjModel.from_xml_path("assets/bimanual_curi1_transfer_cube.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    chains = find_arm_chain()
    ee_controller = EndEffectorController(chains)
    
    print("🤖 机械臂控制器已初始化")
    print("\n📋 在实际运行 curi1_control.py 时，按键触发方式如下:")
    
    # 演示1: O键 - 欧拉角输入
    print(f"\n{'='*60}")
    print("🔹 按键触发方式 1: O键 (欧拉角控制)")
    print("   1. 运行: python enhanced_curi1_control.py")
    print("   2. 等待MuJoCo窗口启动")
    print("   3. 确保终端窗口处于焦点状态")  
    print("   4. 按 'O' 键")
    print("   5. 系统会提示输入格式:")
    print("      格式: side x y z roll pitch yaw(deg) duration(s)")
    print("      例如: right -0.12 -0.65 0.65 0 0 0 2")
    print("   6. 输入后按Enter，机械臂将执行运动")
    
    # 模拟用户输入
    demo_input_o = "right -0.12 -0.65 0.65 0 0 0 2"
    print(f"\n🎯 模拟用户输入: {demo_input_o}")
    parts = demo_input_o.split()
    side, x, y, z, r, p, yw, dur = parts
    pos = np.array([float(x), float(y), float(z)])
    print(f"   解析结果:")
    print(f"   - 机械臂: {side}")
    print(f"   - 目标位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print(f"   - 目标姿态: roll={r}°, pitch={p}°, yaw={yw}°")
    print(f"   - 运动时间: {dur}秒")
    
    # 测试IK可行性
    success, _, info = ee_controller.move_to_pose(
        side=side, target_position=pos, execute_motion=False
    )
    print(f"   ✅ IK验证: {'成功' if success else '失败'}")
    if success:
        print(f"   - 预期误差: {info.get('final_error', 0)*1000:.1f}mm")
    
    # 演示2: Q键 - 四元数输入
    print(f"\n{'='*60}")
    print("🔹 按键触发方式 2: Q键 (四元数控制)")
    print("   1. 按 'Q' 键")
    print("   2. 系统会提示输入格式:")
    print("      格式: side x y z qw qx qy qz duration(s)")
    print("      例如: left -0.2 -0.6 0.6 1 0 0 0 1.5")
    
    demo_input_q = "left -0.2 -0.6 0.6 1 0 0 0 1.5"
    print(f"\n🎯 模拟用户输入: {demo_input_q}")
    parts_q = demo_input_q.split()
    side_q, x_q, y_q, z_q, qw, qx, qy, qz, dur_q = parts_q
    pos_q = np.array([float(x_q), float(y_q), float(z_q)])
    quat = np.array([float(qw), float(qx), float(qy), float(qz)])
    print(f"   解析结果:")
    print(f"   - 机械臂: {side_q}")
    print(f"   - 目标位置: [{pos_q[0]:.3f}, {pos_q[1]:.3f}, {pos_q[2]:.3f}]")
    print(f"   - 目标四元数: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
    print(f"   - 运动时间: {dur_q}秒")
    
    # 测试IK可行性
    success_q, _, info_q = ee_controller.move_to_pose(
        side=side_q, target_position=pos_q, execute_motion=False
    )
    print(f"   ✅ IK验证: {'成功' if success_q else '失败'}")
    
    # 演示3: I键 - 仅位置输入
    print(f"\n{'='*60}")
    print("🔹 按键触发方式 3: I键 (仅位置控制)")
    print("   1. 按 'I' 键")
    print("   2. 系统会提示输入格式:")
    print("      格式: side x y z duration(s)")
    print("      例如: right -0.12 -0.65 0.65 2")
    
    demo_input_i = "right -0.12 -0.65 0.65 2"
    print(f"\n🎯 模拟用户输入: {demo_input_i}")
    parts_i = demo_input_i.split()
    side_i, x_i, y_i, z_i, dur_i = parts_i
    pos_i = np.array([float(x_i), float(y_i), float(z_i)])
    print(f"   解析结果:")
    print(f"   - 机械臂: {side_i}")
    print(f"   - 目标位置: [{pos_i[0]:.3f}, {pos_i[1]:.3f}, {pos_i[2]:.3f}]")
    print(f"   - 运动时间: {dur_i}秒")
    
    success_i, _, info_i = ee_controller.move_to_pose(
        side=side_i, target_position=pos_i, execute_motion=False
    )
    print(f"   ✅ IK验证: {'成功' if success_i else '失败'}")
    if success_i:
        print(f"   - 预期误差: {info_i.get('final_error', 0)*1000:.1f}mm")
    
    # 快捷按键演示
    print(f"\n{'='*60}")
    print("🔹 快捷按键:")
    print("   T键 - 快速测试: 右臂移动到方块上方")
    print("   Y键 - 快速测试: 左臂移动到侧面位置")
    
    print(f"\n{'='*60}")
    print("🚀 使用步骤总结:")
    print("1. 运行 python enhanced_curi1_control.py")
    print("2. 等待MuJoCo可视化窗口启动")
    print("3. 保持终端窗口为活动状态 (不要点击MuJoCo窗口)")
    print("4. 按对应的触发键 (O/Q/I/T/Y)")
    print("5. 根据提示输入位置和姿态参数")
    print("6. 观察机械臂在MuJoCo窗口中的运动")
    
    print(f"\n💡 使用技巧:")
    print("- 坐标系: X(前后), Y(左右), Z(上下)")
    print("- 方块位置参考: [-0.12, -0.65, 0.57]")
    print("- 安全高度: 在Z方向+0.08米 (方块上方8cm)")
    print("- 小角度更容易成功: roll,pitch,yaw < 30度")
    print("- 先试 I键(仅位置) 再试 O键(位置+姿态)")
    
    return True

def show_coordinate_system():
    """显示坐标系参考"""
    print(f"\n📐 坐标系参考图:")
    print("    +Z (上)")
    print("     |")
    print("     |")
    print("     o-----> +X (前)")  
    print("    /")
    print("   /")
    print(" +Y (左)")
    print()
    print("常用位置:")
    print("- 方块中心: [-0.12, -0.65, 0.57]")
    print("- 方块上方: [-0.12, -0.65, 0.65] (安全高度)")
    print("- 左侧位置: [0.2, -0.6, 0.6]")
    print("- 右侧位置: [-0.4, -0.6, 0.6]")

def main():
    """主演示程序"""
    try:
        demo_keyboard_triggers()
        show_coordinate_system()
        
        print(f"\n🎉 演示完成!")
        print("现在你可以运行 python enhanced_curi1_control.py 来试用按键控制功能")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()