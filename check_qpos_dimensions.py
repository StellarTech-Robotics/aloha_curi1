#!/usr/bin/env python3
"""
检查curi1_control.py中qpos的维度配置
"""

import sys
sys.path.append('./assets')

try:
    import mujoco
    import numpy as np
    from curi1_control import model, data, find_arm_chain
    
    print("🤖 CURI机器人关节配置分析")
    print("=" * 50)
    
    # 基本信息
    print(f"总关节数量 (nq): {model.nq}")
    print(f"总自由度数量 (nv): {model.nv}")
    print(f"总驱动器数量 (nu): {model.nu}")
    print(f"qpos维度: {data.qpos.shape}")
    print(f"qvel维度: {data.qvel.shape}")
    print(f"ctrl维度: {data.ctrl.shape}")
    
    print("\n📋 关节详细信息:")
    print("-" * 50)
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        qposadr = model.jnt_qposadr[i] if i < len(model.jnt_qposadr) else -1
        dofadr = model.jnt_dofadr[i] if i < len(model.jnt_dofadr) else -1
        jnt_type = model.jnt_type[i]
        
        # 关节类型名称
        type_names = {
            0: "free", 1: "ball", 2: "slide", 3: "hinge"
        }
        type_name = type_names.get(jnt_type, f"type_{jnt_type}")
        
        print(f"{i:2d}: {name:20s} | type: {type_name:5s} | qpos_adr: {qposadr:2d} | dof_adr: {dofadr:2d}")
    
    print("\n🦾 机械臂关节分析:")
    print("-" * 50)
    chains = find_arm_chain()
    
    for side in ["left", "right"]:
        c = chains[side]
        jids = c["jids"]
        qadr = c["qadr"]
        
        print(f"\n{side.upper()} 机械臂:")
        print(f"  关节IDs: {jids}")
        print(f"  qpos地址: {qadr}")
        print(f"  关节数量: {len(jids)}")
        
        for i, jid in enumerate(jids):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            qpos_val = data.qpos[qadr[i]]
            print(f"    {i}: {name} (qpos[{qadr[i]}] = {qpos_val:.3f})")
    
    print("\n🤏 夹爪关节分析:")
    print("-" * 50)
    
    # 左夹爪
    left_gripper = ["Joint_finger1", "Joint_finger2"]
    right_gripper = ["r_Joint_finger1", "r_Joint_finger2"]
    
    print("LEFT 夹爪:")
    for gname in left_gripper:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gname)
            qadr = model.jnt_qposadr[jid]
            qpos_val = data.qpos[qadr]
            print(f"  {gname} (joint_id: {jid}, qpos[{qadr}] = {qpos_val:.3f})")
        except:
            print(f"  {gname} - 未找到")
    
    print("RIGHT 夹爪:")
    for gname in right_gripper:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gname)
            qadr = model.jnt_qposadr[jid]
            qpos_val = data.qpos[qadr]
            print(f"  {gname} (joint_id: {jid}, qpos[{qadr}] = {qpos_val:.3f})")
        except:
            print(f"  {gname} - 未找到")
    
    print("\n📊 与Mobile ALOHA对比:")
    print("-" * 50)
    arm_joints = len(chains["left"]["jids"]) + len(chains["right"]["jids"])
    
    print(f"机械臂关节: {len(chains['left']['jids'])} (左) + {len(chains['right']['jids'])} (右) = {arm_joints}")
    
    # 计算夹爪关节数
    gripper_count = 0
    for gname in left_gripper + right_gripper:
        try:
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, gname)
            gripper_count += 1
        except:
            pass
    
    print(f"夹爪关节: {gripper_count}")
    print(f"总控制关节: {arm_joints + gripper_count}")
    
    if arm_joints == 12 and gripper_count == 4:
        print("✅ 符合Mobile ALOHA标准: 12(机械臂) + 4(夹爪) = 16")
    elif arm_joints == 12 and gripper_count == 2:
        print("⚠️ 类似Mobile ALOHA: 12(机械臂) + 2(夹爪) = 14")
    else:
        print(f"ℹ️ CURI配置: {arm_joints}(机械臂) + {gripper_count}(夹爪) = {arm_joints + gripper_count}")
    
    print(f"\n🎯 数据记录维度分析:")
    print(f"MuJoCo qpos总维度: {model.nq}")
    print(f"实际控制维度: {arm_joints + gripper_count}")
    
    if model.nq > arm_joints + gripper_count:
        print(f"额外维度: {model.nq - arm_joints - gripper_count} (可能包括基座、头部等)")
        
        # 找出额外的关节
        arm_joint_names = set()
        for side in ["left", "right"]:
            for jid in chains[side]["jids"]:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                arm_joint_names.add(name)
        
        gripper_joint_names = set(left_gripper + right_gripper)
        
        print("\n额外关节:")
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            if name not in arm_joint_names and name not in gripper_joint_names:
                qposadr = model.jnt_qposadr[i] 
                qpos_val = data.qpos[qposadr]
                print(f"  {name} (qpos[{qposadr}] = {qpos_val:.3f})")

except Exception as e:
    print(f"❌ 检查失败: {e}")
    print("请确保在正确的目录下运行，且MuJoCo环境正常")

if __name__ == "__main__":
    pass