import mujoco

import mujoco, mujoco.viewer; 
model = mujoco.MjModel.from_xml_path('bimanual_curi1_ee_transfer_cube.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

def get_body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos  = data.xpos[bid].copy()   # (3,) world position
    quat = data.xquat[bid].copy()  # (4,) world quaternion (w,x,y,z)
    return pos, quat

# 例：左右末端
l_pos, l_quat = get_body_pose(model, data, "l_rmg42_base_link")
r_pos, r_quat = get_body_pose(model, data, "r_rmg42_base_link")
print("L EE:", l_pos, l_quat)
print("R EE:", r_pos, r_quat)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

l_pos, l_quat = get_body_pose(model, data, "l_rmg42_base_link")
r_pos, r_quat = get_body_pose(model, data, "r_rmg42_base_link")
print("L EE:", l_pos, l_quat)
print("R EE:", r_pos, r_quat)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("仿真窗口已启动。关闭窗口即可退出。")
    while viewer.is_running():
        viewer.sync()
