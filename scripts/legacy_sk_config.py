"""SK 示例的兼容配置。

新 SOP 应使用 projects/<project_id>/ 下的 project.json、rois.json 和
workflow.json。本模块只服务于未传项目配置的旧调用和历史工具。
"""

from scripts.config import ROOT

DEFAULT_SOP_PROJECT_DIR = ROOT / "projects" / "SK_DEMO"
DEFAULT_VIDEO_PATH = ROOT / "videos" / "sk.mp4"
DEFAULT_MODEL_PATH = ROOT / "models" / "best_yolo26s.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "yolo26s"

WORK_ROI = [1115, 1132, 1648, 1522]
SCREW_BIN_ROI = [1092, 505, 1410, 932]
TOOL_HOME_ROI = [782, 508, 1108, 930]

TARGET_CLASSES = ["bearing", "cover", "tool"]

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

SCREW_BIN_COLOR = (255, 0, 0)
TOOL_HOME_COLOR = (0, 180, 255)
