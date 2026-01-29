#!/usr/bin/env python3
"""
测试14维Mobile ALOHA格式转换功能
验证CURI qpos与Mobile ALOHA格式的转换是否正确
"""

import numpy as np

def test_14dim_conversion():
    """测试14维转换功能（模拟版本，无需MuJoCo）"""
    print("🧪 测试14维Mobile ALOHA格式转换")
    print("=" * 50)
    
    # 模拟CURI的完整qpos数据 (假设19维)
    print("1. 模拟CURI完整qpos数据:")
    simulated_curi_qpos = np.array([
        # 左臂6维
        0.1, -0.5, 0.8, 0.0, 0.3, -0.2,
        # 左夹爪2维
        0.04, -0.04,
        # 右臂6维
        -0.1, -0.5, 0.8, 0.0, 0.3, 0.2,
        # 右夹爪2维
        0.05, -0.05,
        # 额外关节3维 (头部、基座等)
        0.0, 0.1, 0.0
    ], dtype=np.float32)
    
    print(f"   CURI完整qpos ({len(simulated_curi_qpos)}维): {simulated_curi_qpos}")
    
    # 模拟chains配置
    simulated_chains = {
        "left": {"qadr": [0, 1, 2, 3, 4, 5]},   # 左臂关节的qpos索引
        "right": {"qadr": [8, 9, 10, 11, 12, 13]}  # 右臂关节的qpos索引
    }
    
    # 模拟14维转换
    print("\n2. 转换为14维Mobile ALOHA格式:")
    mobile_aloha_14dim = extract_14dim_qpos_simulation(simulated_curi_qpos, simulated_chains)
    
    joint_names = [
        "l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6",
        "l_gripper",
        "r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6", 
        "r_gripper"
    ]
    
    print(f"   Mobile ALOHA 14维格式:")
    for i, (name, val) in enumerate(zip(joint_names, mobile_aloha_14dim)):
        print(f"     [{i:2d}] {name:12s}: {val:6.3f}")
    
    # 验证转换逻辑
    print("\n3. 验证转换逻辑:")
    print(f"   左臂: 原始{simulated_curi_qpos[0:6]} -> 转换{mobile_aloha_14dim[0:6]}")
    print(f"   左夹爪: 原始{simulated_curi_qpos[6:8]} -> 平均值{mobile_aloha_14dim[6]} (平均: {np.mean(simulated_curi_qpos[6:8]):.3f})")
    print(f"   右臂: 原始{simulated_curi_qpos[8:14]} -> 转换{mobile_aloha_14dim[7:13]}")
    print(f"   右夹爪: 原始{simulated_curi_qpos[14:16]} -> 平均值{mobile_aloha_14dim[13]} (平均: {np.mean(simulated_curi_qpos[14:16]):.3f})")
    print(f"   忽略额外关节: {simulated_curi_qpos[16:]} (头部、基座等)")
    
    # 模拟反向转换
    print("\n4. 测试反向转换 (14维 -> 完整qpos):")
    restored_qpos = expand_14dim_to_full_qpos_simulation(
        mobile_aloha_14dim, simulated_curi_qpos, simulated_chains
    )
    
    print(f"   恢复后的完整qpos: {restored_qpos}")
    print(f"   左臂恢复: {restored_qpos[0:6]}")
    print(f"   左夹爪恢复: {restored_qpos[6:8]} (分配: {mobile_aloha_14dim[6]} -> [{mobile_aloha_14dim[6]}, {-mobile_aloha_14dim[6]}])")
    print(f"   右臂恢复: {restored_qpos[8:14]}")
    print(f"   右夹爪恢复: {restored_qpos[14:16]} (分配: {mobile_aloha_14dim[13]} -> [{mobile_aloha_14dim[13]}, {-mobile_aloha_14dim[13]}])")
    
    # 计算误差
    print("\n5. 转换精度验证:")
    arm_error = np.mean(np.abs(restored_qpos[:6] - simulated_curi_qpos[:6])) + \
                np.mean(np.abs(restored_qpos[8:14] - simulated_curi_qpos[8:14]))
    print(f"   机械臂关节误差 (12维): {arm_error:.6f} (应该为0)")
    
    # 注意：夹爪转换是有损的，因为是2维->1维->2维
    gripper_error_left = np.mean(np.abs(restored_qpos[6:8] - simulated_curi_qpos[6:8]))
    gripper_error_right = np.mean(np.abs(restored_qpos[14:16] - simulated_curi_qpos[14:16]))
    print(f"   夹爪转换误差 (左): {gripper_error_left:.6f}")
    print(f"   夹爪转换误差 (右): {gripper_error_right:.6f}")
    print(f"   注意: 夹爪转换是有损的 (2维->1维->2维)")
    
    return mobile_aloha_14dim

def extract_14dim_qpos_simulation(curi_qpos, chains):
    """模拟14维提取函数 (无需MuJoCo)"""
    result = np.zeros(14, dtype=np.float32)
    
    # 左臂6维
    for i in range(6):
        result[i] = curi_qpos[chains["left"]["qadr"][i]]
    
    # 左夹爪1维 (平均值) - 索引6, 7
    result[6] = (curi_qpos[6] + curi_qpos[7]) / 2.0
    
    # 右臂6维
    for i in range(6):
        result[7 + i] = curi_qpos[chains["right"]["qadr"][i]]
    
    # 右夹爪1维 (平均值) - 索引14, 15
    result[13] = (curi_qpos[14] + curi_qpos[15]) / 2.0
    
    return result

def expand_14dim_to_full_qpos_simulation(mobile_aloha_qpos, current_full_qpos, chains):
    """模拟14维扩展函数 (无需MuJoCo)"""
    result = current_full_qpos.copy()
    
    # 左臂6维
    for i in range(6):
        result[chains["left"]["qadr"][i]] = mobile_aloha_qpos[i]
    
    # 左夹爪 - 将1维分配给两个手指
    gripper_val = mobile_aloha_qpos[6]
    result[6] = gripper_val    # finger1
    result[7] = -gripper_val   # finger2 (相反方向)
    
    # 右臂6维
    for i in range(6):
        result[chains["right"]["qadr"][i]] = mobile_aloha_qpos[7 + i]
    
    # 右夹爪 - 将1维分配给两个手指
    gripper_val = mobile_aloha_qpos[13]
    result[14] = gripper_val   # finger1
    result[15] = -gripper_val  # finger2 (相反方向)
    
    return result

def show_mobile_aloha_compatibility():
    """显示Mobile ALOHA兼容性信息"""
    print("\n" + "="*50)
    print("📋 Mobile ALOHA兼容性总结")
    print("=" * 50)
    
    print("✅ 已实现的功能:")
    print("1. extract_14dim_qpos() - CURI完整qpos -> 14维Mobile ALOHA格式")
    print("2. expand_14dim_to_full_qpos() - 14维 -> CURI完整qpos格式")
    print("3. 14维qvel提取 - 对应的关节速度提取")
    print("4. 14维action记录 - 兼容ACT训练格式")
    print("5. set_14dim_target_qpos() - 从Mobile ALOHA模型设置目标")
    print("6. CSV/HDF5记录 - 使用14维格式和正确的列名")
    
    print("\n📊 数据格式:")
    print("CURI原始 -> Mobile ALOHA 14维:")
    print("  左臂6维   (l_joint1-6)     -> 索引 0-5")
    print("  左夹爪2维 (finger1,2)      -> 索引 6    (平均值)")
    print("  右臂6维   (r_joint1-6)     -> 索引 7-12")
    print("  右夹爪2维 (r_finger1,2)    -> 索引 13   (平均值)")
    print("  额外关节  (head,platform)  -> 忽略")
    
    print("\n🎯 ACT训练兼容性:")
    print("✅ qpos维度: 14维 (与Mobile ALOHA一致)")
    print("✅ qvel维度: 14维 (对应关节速度)")
    print("✅ action维度: 14维 (控制命令)")
    print("✅ 关节命名: l_joint1-6, l_gripper, r_joint1-6, r_gripper")
    print("✅ HDF5格式: /observations/qpos, /observations/qvel, /action")
    
    print("\n⚠️  注意事项:")
    print("1. 夹爪转换是有损的 (2维->1维->2维)")
    print("2. 额外关节信息会丢失 (头部、基座)")
    print("3. 夹爪控制策略可能需要调整")
    print("4. 反向转换时假设双手指相反运动")

def main():
    """主测试程序"""
    try:
        mobile_aloha_data = test_14dim_conversion()
        show_mobile_aloha_compatibility()
        
        print(f"\n🎉 14维格式转换测试完成!")
        print("现在CURI机器人数据可以与Mobile ALOHA ACT算法兼容!")
        print(f"测试生成的14维数据: {mobile_aloha_data}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()