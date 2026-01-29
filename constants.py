import pathlib
import os

### Task parameters
def _safe_getlogin():
    try:
        return os.getlogin()
    except OSError:
        return None

_user = _safe_getlogin()
if _user == 'chaos':
    DATA_DIR = '/mnt/d/mycodes/act++/curi1/datasets'
else:
    DATA_DIR = '/mnt/d/mycodes/act++/curi1/datasets'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 1600,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'curi_auto_grasp':{
        'dataset_dir': DATA_DIR + '/curi_auto_grasp',
        'num_episodes': 10,
        'episode_len': 250,  # 约10秒的抓取序列
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}

TASK_CONFIGS = {
    'curi_auto_grasp':{
        'dataset_dir': DATA_DIR + '/curi_auto_grasp',
        'num_episodes': 10,
        'episode_len': 250,  # 约10秒的抓取序列
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    }

}

### Simulation envs fixed constants
DT = 0.01 #0.02
FPS = 100 #50
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]

# mapping JOINT_NAMES plan_A created by chatgpt5 2025.08.14
RIGHT_NAME_MAP = {
    "waist":"r_joint1", "shoulder":"r_joint2", "elbow":"r_joint3",
    "forearm_roll":"r_joint4", "wrist_angle":"r_joint5", "wrist_rotate":"r_joint6",
}
LEFT_NAME_MAP = {
    "waist":"l_joint1", "shoulder":"l_joint2", "elbow":"l_joint3",
    "forearm_roll":"l_joint4", "wrist_angle":"l_joint5", "wrist_rotate":"l_joint6",
}
RIGHT_ARM_JOINTS = [RIGHT_NAME_MAP[n] for n in JOINT_NAMES]
LEFT_ARM_JOINTS  = [LEFT_NAME_MAP[n]  for n in JOINT_NAMES]
ALL_JOINTS = RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS

# mapping JOINT_NAMES plan_B created by chatgpt5 2025.08.14
# JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]
# RIGHT_ARM_JOINTS = [f"r_{n}" for n in JOINT_NAMES]  # 或直接 ["r_joint1",...]
# LEFT_ARM_JOINTS  = [f"l_{n}" for n in JOINT_NAMES]


# START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
START_ARM_POSE = [-0.785, -0.785, -1.57, 1, 1, 0, 0.0, 0.0,  0.785, 0.785, 1.57, -1, -1, 0, 0.0, 0.0]

# CURI robot workspace parameters (optimized based on kinematic analysis)
CURI_WORKSPACE = {
    'left_arm': {
        'x_range': [-0.1, 0.7],   # Extended to include handover area
        'y_range': [-0.9, -0.3],  # Reachable Y range for left arm
        'z_range': [0.4, 1.0],    # Conservative height range
        'shoulder_pos': [0, 0, 0.187]  # From curi1_left.xml
    },
    'right_arm': {
        'x_range': [-0.7, 0.1],   # Extended to include handover area 
        'y_range': [-0.9, -0.3],  # Same Y range as left
        'z_range': [0.4, 1.0],    # Same Z range as left
        'shoulder_pos': [0, 0, 0.187]  # From curi1_right.xml
    },
    'shared_workspace': {  # Overlap area for reliable object handover
        'x_range': [-0.15, 0.15], # Narrower but safer handover zone
        'y_range': [-0.75, -0.45], # Closer to robot base for better reach
        'z_range': [0.55, 0.75]    # Table level to comfortable height
    }
}

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = -0.8
MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
