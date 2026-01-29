import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("bimanual_curi1_transfer_cube.xml")  # bimanual_curi1_transfer_cube  three_dof_arm  six_dof_arm  curi1_left2
data = mujoco.MjData(model)

# 设置初始姿态
mujoco.mj_resetDataKeyframe(model, data, 0)  # 使用keyframe中的initial_pose

# 获取初始关节角度作为目标
target_qpos = data.qpos[:8].copy()  # 前16个是机械臂关节 target_qpos = data.qpos[:16].copy()
print("总自由度数 nq (qpos长度):", model.nq)  # 应该能正常输出，比如 26
print("关节自由度起始地址:")

# 仿真循环
viewer = mujoco.viewer.launch_passive(model, data)
while viewer.is_running():
    # 关键：每一步都要设置控制目标！
    # data.ctrl[:16] = target_qpos  # 持续发送位置命令
    
    # 可选：添加简单的重力补偿
    # mujoco.mj_inverse(model, data)
    # gravity_compensation = data.qfrc_bias[:16] * 0.5
    # data.ctrl[:16] += gravity_compensation
    
    mujoco.mj_step(model, data)
    viewer.sync()


# for i in range(model.njnt):
#     name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
#     qposadr = model.jnt_qposadr[i]
#     jtype = model.jnt_type[i]
#     jtypename = ["free", "ball", "slide", "hinge"][jtype]
#     print(f"{name:20s}: 地址 {qposadr:2d}, 类型 {jtypename}")

# mujoco.mj_resetDataKeyframe(model, data, 0)
# mujoco.mj_forward(model, data)
# print("Loaded qpos:", data.qpos)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print("仿真窗口已启动。关闭窗口即可退出。")
#     while viewer.is_running():
#         viewer.sync()