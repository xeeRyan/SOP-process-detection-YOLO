"""应用级路径、网络和推理默认值。"""

from pathlib import Path

# 应用根目录，所有通用路径都基于这里拼接。
ROOT = Path(__file__).resolve().parents[1]

# 通用模型、日志与输出位置。
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "detect"
DEFAULT_HAND_POSE_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DEFAULT_LOG_DIR = ROOT / "outputs" / "logs"

# 手部骨骼输出配置。
ENABLE_HAND_POSE = True
HAND_POSE_SAMPLE_INTERVAL = 1

# 轻量目标跟踪与事件防抖默认参数。
TRACKING_IOU_THRESHOLD = 0.2
TRACKING_MAX_MISSING_FRAMES = 8
EVENT_LOST_TOLERANCE_FRAMES = 8

# 软件封装默认 TCP 配置。
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 9000
DEFAULT_TCP_BUFFER_SIZE = 102400

# 输出文件名，软件端可固定读取这两个文件。
RESULT_VIDEO_NAME = "result.mp4"
RESULT_JSON_NAME = "result.json"

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
TEXT_COLOR = (0, 0, 255)
OK_COLOR = (0, 180, 0)
NG_COLOR = (0, 0, 255)
