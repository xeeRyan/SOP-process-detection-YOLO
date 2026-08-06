# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import shutil

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()


def add_dir(path: str, target: str):
    source = ROOT / path
    if source.exists():
        return [(str(source), target)]
    return []


# 只读文档/部署辅助文件可以放在 PyInstaller 的 _internal 中。
# config、models、projects 等运行时资产必须位于 SOP_PYD.exe 同级目录：
# task_dispatcher 在 frozen 模式下以 sys.executable.parent 作为 APP_ROOT，
# 且项目创建、训练、检测输出都要求这些目录可写。
embedded_datas = []
embedded_datas += add_dir('docs', 'docs')
embedded_datas += add_dir('deploy', 'deploy')

runtime_asset_dirs = (
    'config',
    'models',
    'projects',
    'datasets',
    'videos',
)

# Collect package resources used by runtime imports and model export.
package_datas = []
package_datas += collect_data_files('ultralytics', include_py_files=False)
package_datas += collect_data_files('polars', include_py_files=False)

package_binaries = []
package_binaries += collect_dynamic_libs('mediapipe')
hiddenimports = []
for package_name in ('onnx', 'onnxruntime', 'onnxslim'):
    datas, binaries, imports = collect_all(package_name)
    package_datas += datas
    package_binaries += binaries
    hiddenimports += imports

hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('polars')
# SciPy imports its bundled Array API compatibility modules dynamically.
# PyInstaller cannot discover these imports through static analysis.
hiddenimports += collect_submodules('scipy._external.array_api_compat')
hiddenimports += [
    'cv2',
    'numpy',
    'torch',
    'torchvision',
    'onnx',
    'onnx.onnx_cpp2py_export',
    'onnxruntime',
    'onnxslim',
    'coloredlogs',
    'humanfriendly',
    'ml_dtypes',
    'sympy',
]


a = Analysis(
    ['SOP_PYD.py'],
    pathex=[str(ROOT)],
    binaries=package_binaries,
    datas=embedded_datas + package_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'pandas.tests',
        'scipy.tests',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SOP_PYD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SOP_PYD',
)


def copy_runtime_asset(source_name: str) -> None:
    """将运行时资产同步到 EXE 同级目录，避免被放入 _internal。"""

    source = ROOT / source_name
    if not source.exists():
        return
    target = ROOT / 'dist' / 'SOP_PYD' / source_name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        def ignore_runtime_residue(directory: str, names: list[str]) -> set[str]:
            ignored = {
                name
                for name in names
                if name in {'__pycache__', 'outputs', 'runs'}
                or name.endswith(('.pyc', '.pyo', '.tmp', '.temp'))
                or '.inprogress.' in name
            }
            # 全局 outputs/runs 由下方创建空目录；项目下的 outputs/runs
            # 也是历史运行产物，不应进入交付包。
            return ignored

        shutil.copytree(source, target, ignore=ignore_runtime_residue)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


for runtime_asset in runtime_asset_dirs:
    copy_runtime_asset(runtime_asset)

# 运行输出不从源码目录复制，交付包内只创建干净的可写目录。
for writable_dir in ('outputs', 'runs'):
    (ROOT / 'dist' / 'SOP_PYD' / writable_dir).mkdir(parents=True, exist_ok=True)


