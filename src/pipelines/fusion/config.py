OUTPUT_SUBDIRS = ("keypoints3d", "metadata")

HEIGHT = 1.63
RIGID_BONES_RATIO = {
    ("left_elbow", "left_shoulder"): 0.186,
    ("left_wrist", "left_elbow"): 0.146,
    ("right_elbow", "right_shoulder"): 0.186,
    ("right_wrist", "right_elbow"): 0.146,
    ("left_knee", "left_hip"): 0.245,
    ("left_ankle", "left_knee"): 0.246,
    ("right_knee", "right_hip"): 0.245,
    ("right_ankle", "right_knee"): 0.246,
}

ROTATION_PARENT_JOINTS = {
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "left_toe": "left_ankle",
    "left_foot": "left_ankle",
    "right_toe": "right_ankle",
    "right_foot": "right_ankle",
}

NON_REPLACEABLE_ANCHORS = {"left_hip", "left_shoulder", "right_hip", "right_shoulder"}

# Distal joints only. Shoulder/hip stay on the destination camera so the chain
# is grafted onto the local torso after the cross-view similarity.
LIMB_CHAINS = {
    "left_arm": {
        "root": "left_shoulder",
        "joints": ("left_elbow", "left_wrist", "left_hand"),
        "bones": (
            ("left_elbow", "left_shoulder"),
            ("left_wrist", "left_elbow"),
            ("left_hand", "left_wrist"),
        ),
    },
    "right_arm": {
        "root": "right_shoulder",
        "joints": ("right_elbow", "right_wrist", "right_hand"),
        "bones": (
            ("right_elbow", "right_shoulder"),
            ("right_wrist", "right_elbow"),
            ("right_hand", "right_wrist"),
        ),
    },
    "left_leg": {
        "root": "left_hip",
        "joints": ("left_knee", "left_ankle", "left_foot", "left_toe"),
        "bones": (
            ("left_knee", "left_hip"),
            ("left_ankle", "left_knee"),
            ("left_foot", "left_ankle"),
            ("left_toe", "left_ankle"),
        ),
    },
    "right_leg": {
        "root": "right_hip",
        "joints": ("right_knee", "right_ankle", "right_foot", "right_toe"),
        "bones": (
            ("right_knee", "right_hip"),
            ("right_ankle", "right_knee"),
            ("right_foot", "right_ankle"),
            ("right_toe", "right_ankle"),
        ),
    },
}

TORSO_PART_IDS = {0, 3, 6, 9, 13, 14}
OCCLUSION_CHECK_JOINTS = {
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_toe",
    "left_foot",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "right_toe",
    "right_foot",
    "left_knee",
    "left_ankle",
    "right_knee",
    "right_ankle",
}

JOINT_TO_SMPL_PART_ID = {
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "left_toe": 10,
    "right_toe": 11,
}

JOINT_EXCLUDED_PART_IDS = {
    "left_elbow": {16, 18, 20},
    "left_wrist": {18, 20, 22},
    "left_hand": {20, 22},
    "left_knee": {1, 4, 7},
    "left_ankle": {4, 7, 10},
    "left_foot": {7, 10},
    "left_toe": {7, 10},
    "right_elbow": {17, 19, 21},
    "right_wrist": {19, 21, 23},
    "right_hand": {21, 23},
    "right_knee": {2, 5, 8},
    "right_ankle": {5, 8, 11},
    "right_foot": {8, 11},
    "right_toe": {8, 11},
}

ORIENTATION_EPSILON = 1e-2
HARMONIC_EPSILON = 1e-6
BELIEF_DELTA_CAP = 0.05

DEFAULT_OCCLUDED_FACTOR = 0.25
HUBER_DELTA = 0.05
BONE_LENGTH_MIN_SCALE = 0.96
BONE_LENGTH_MAX_SCALE = 1.04
STRICT_EQUALITY_BONES = True

# Fix #4: Ngưỡng belief tối thiểu của slave camera để fusion correction được phép kích hoạt.
# Nếu mean belief slave < ngưỡng này, bỏ qua correction để tránh áp similarity transform kém lên dữ liệu.
MIN_SLAVE_BELIEF_THRESHOLD = 0.0

# Fix #3: Số joint tối thiểu trong l_list để RANSAC có thể ước lượng transform đáng tin cậy.
# Dưới ngưỡng này, correction bị skip dù correction_enabled=True.
MIN_LLIST_FOR_RANSAC = 4

# Limb-winner: skip a chain whose mapped bone directions flip too far (bad Umeyama).
DEFAULT_MAX_BONE_ANGLE_DEG = 60.0
MIN_LIMB_SOURCE_BELIEF = 0.08
TORSO_HEIGHT_RATIO = 0.288
