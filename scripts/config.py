from pathlib import Path

# 项目根目录，所有默认路径都基于这里拼接。
ROOT = Path(__file__).resolve().parents[1]

# 项目默认输入、模型和输出位置。
DEFAULT_VIDEO_PATH = ROOT / "videos" / "sk.mp4"
DEFAULT_MODEL_PATH = ROOT / "models" / "best_yolo26s.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "yolo26s"
DEFAULT_HAND_POSE_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DEFAULT_LOG_DIR = ROOT / "outputs" / "logs"

# 手部骨骼输出配置。
ENABLE_HAND_POSE = True
HAND_POSE_SAMPLE_INTERVAL = 30

# 软件封装默认 TCP 配置。
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 5000
DEFAULT_TCP_BUFFER_SIZE = 102400

# 输出文件名，软件端可固定读取这两个文件。
RESULT_VIDEO_NAME = "result.mp4"
RESULT_JSON_NAME = "result.json"

# SOP 业务区域，格式为 [x1, y1, x2, y2]。
WORK_ROI = [1115, 1132, 1648, 1522]
SCREW_BIN_ROI = [1092, 505, 1410, 932]
TOOL_HOME_ROI = [782, 508, 1108, 930]

# YOLO 只负责物体和工具检测；手部由 MediaPipe hand_pose 模块负责。
TARGET_CLASSES = ["bearing", "cover", "tool"]

# SOP 步骤定义；软件端可按 step_id/key/name 展示步骤状态。
STEP_DEFINITIONS = [
    {
        "step_id": 1,
        "key": "bearing",
        "name": "放入方形轴承",
        "trigger_classes": ["bearing"],
        "roi_name": "work",
    },
    {
        "step_id": 2,
        "key": "cover",
        "name": "盖上圆形盖",
        "trigger_classes": ["cover"],
        "roi_name": "work",
    },
    {
        "step_id": 3,
        "key": "screw_action",
        "name": "手部骨骼或工具进入螺丝盘",
        "trigger_classes": ["tool"],
        "roi_name": "screw_bin",
        "allow_hand_pose": True,
    },
    {
        "step_id": 4,
        "key": "tool_return",
        "name": "工具放回原位",
        "trigger_classes": ["tool"],
        "roi_name": "tool_home",
    },
]

# 检测和输出默认参数。
CONFIDENCE_THRESHOLD = 0.25
DEFAULT_NMS_THRESHOLD = 0.7
DEFAULT_ENABLE_YOLO = True
DEFAULT_OUTPUT_VIDEO = True
DEFAULT_OUTPUT_JSON = True
DEFAULT_REALTIME_DISPLAY = False

# 图像增强默认参数。

STEP_STABLE_FRAMES = 3
TOOL_HOLD_FRAMES = 10

BOX_COLOR = (0, 0, 255)
ROI_COLOR = (0, 255, 255)
SCREW_BIN_COLOR = (255, 0, 0)
TOOL_HOME_COLOR = (0, 180, 255)
TEXT_COLOR = (0, 0, 255)
OK_COLOR = (0, 180, 0)
NG_COLOR = (0, 0, 255)
