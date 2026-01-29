#!/usr/bin/env python3
"""
测试平滑运动控制功能
演示如何使用增强的smooth pose control
"""

import numpy as np
import time

def minjerk(alpha):
    """最小加加速度插值 - S型平滑曲线"""
    return alpha**3 * (10 - 15*alpha + 6*alpha*alpha)

def simulate_smooth_motion(target_pos, seconds=3.0, fps=50, verbose=True):
    """
    模拟平滑运动控制过程（无需MuJoCo）
    演示时间同步和进度跟踪
    """
    if verbose:
        print(f"🎯 模拟平滑控制 -> 目标位置 {target_pos} (耗时 {seconds:.1f}秒)")
    
    steps = max(1, int(seconds * fps))
    dt_iter = seconds / steps
    
    # 模拟起始位置
    start_pos = np.array([0.0, -0.5, 0.5])
    
    if verbose:
        distance = np.linalg.norm(target_pos - start_pos)
        print(f"   移动距离: {distance*100:.1f}cm | 平均速度: {distance/seconds*100:.1f}cm/s")
        print(f"   执行 {steps} 步，每步 {dt_iter*1000:.1f}ms")
        print("   开始执行平滑运动...")

    start_time = time.perf_counter()
    positions = []

    for i in range(steps):
        step_start = time.perf_counter()

        alpha = (i + 1) / steps
        s = minjerk(alpha)  # S型插值

        # 位置插值
        current_pos = (1.0 - s) * start_pos + s * target_pos
        positions.append(current_pos.copy())

        # 进度显示
        if verbose and i % max(1, steps//10) == 0:
            progress = (i + 1) / steps * 100
            elapsed = time.perf_counter() - start_time
            estimated_total = elapsed / alpha if alpha > 0 else seconds
            remaining = max(0, estimated_total - elapsed)
            print(f"   进度: {progress:4.1f}% | 位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] | 剩余: {remaining:.1f}s", end='\r')

        # 时间同步 - 确保实时执行
        step_elapsed = time.perf_counter() - step_start
        sleep_time = dt_iter - step_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # 额外保险 - 确保最小执行时间
        if step_elapsed < dt_iter * 0.5:
            time.sleep(dt_iter * 0.2)

    # 最终状态
    if verbose:
        print()  # 换行
        final_pos = positions[-1]
        final_error = np.linalg.norm(target_pos - final_pos)
        total_time = time.perf_counter() - start_time
        print(f"   ✅ 运动完成! 耗时: {total_time:.2f}s | 误差: {final_error*1000:.1f}mm")

    return positions, total_time

def demo_speed_control():
    """演示不同速度的运动控制"""
    print("🎮 平滑运动速度控制演示")
    print("=" * 60)
    
    target_position = np.array([-0.12, -0.65, 0.65])  # 目标位置
    
    # 不同速度测试
    speed_tests = [
        {"duration": 1.0, "desc": "极快速度 (1秒)", "fps": 60},
        {"duration": 2.5, "desc": "正常速度 (2.5秒)", "fps": 50},
        {"duration": 5.0, "desc": "慢速度 (5秒)", "fps": 40},
    ]
    
    for i, test in enumerate(speed_tests, 1):
        print(f"\n{i}. {test['desc']}")
        print("-" * 40)
        
        positions, actual_time = simulate_smooth_motion(
            target_position, 
            seconds=test['duration'], 
            fps=test['fps'], 
            verbose=True
        )
        
        # 验证运动连续性
        velocities = []
        for j in range(1, len(positions)):
            dt = test['duration'] / len(positions)
            vel = np.linalg.norm(positions[j] - positions[j-1]) / dt
            velocities.append(vel)
        
        max_vel = max(velocities) if velocities else 0
        avg_vel = np.mean(velocities) if velocities else 0
        
        print(f"   📊 运动分析:")
        print(f"   - 实际耗时: {actual_time:.2f}s (目标: {test['duration']:.1f}s)")
        print(f"   - 轨迹点数: {len(positions)}")
        print(f"   - 最大速度: {max_vel*100:.1f}cm/s")
        print(f"   - 平均速度: {avg_vel*100:.1f}cm/s")
        print(f"   - 运动模式: {'连续平滑' if actual_time >= test['duration'] * 0.9 else '过快执行'}")
        
        if i < len(speed_tests):
            input("   按Enter继续下一个测试...")

def demo_usage_example():
    """演示实际使用方法"""
    print(f"\n🚀 实际使用方法演示")
    print("=" * 60)
    
    print("现在，在增强的 curi1_control_posekeys.py 中:")
    print()
    print("1. 运行程序:")
    print("   python assets/curi1_control_posekeys.py")
    print()
    print("2. 等待MuJoCo窗口启动后，按 'O' 键")
    print()
    print("3. 输入控制参数 (现在会显示平滑运动进度):")
    print("   格式: side x y z roll pitch yaw(deg) duration(s)")
    print("   示例: right -0.12 -0.65 0.65 0 0 0 3")
    print()
    print("4. 观察输出，你将看到:")
    print("   🎯 平滑控制 RIGHT 臂 -> 目标位置 [-0.12 -0.65  0.65] (耗时 3.0秒)")
    print("      移动距离: 15.2cm | 平均速度: 5.1cm/s")
    print("      执行 150 步，每步 20.0ms")
    print("      进度: 20.0% | 剩余: 2.4s")
    print("      进度: 40.0% | 剩余: 1.8s")
    print("      ...")
    print("      ✅ 运动完成! 耗时: 3.02s | 误差: 2.3mm")
    print()
    print("🎯 关键改进:")
    print("✅ 强制启用实时时间同步，防止瞬间移动")
    print("✅ 增加额外的时间保险机制")
    print("✅ 详细的进度显示和运动分析")
    print("✅ 连贯的S型加速度曲线")
    print("✅ 可配置的运动速度 (duration参数)")

def main():
    """主演示程序"""
    try:
        demo_speed_control()
        demo_usage_example()
        
        print(f"\n🎉 演示完成!")
        print("增强的平滑控制已集成到 assets/curi1_control_posekeys.py")
        print("现在O键和Q键都支持真正的平滑、可控速度运动!")
        
    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()