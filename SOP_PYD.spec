# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()


def add_dir(path: str, target: str):
    source = ROOT / path
    if source.exists():
        return [(str(source), target)]
    return []


project_datas = []
project_datas += add_dir('config', 'config')
project_datas += add_dir('models', 'models')
project_datas += add_dir('datasets', 'datasets')
project_datas += add_dir('videos', 'videos')
project_datas += add_dir('docs', 'docs')
project_datas += add_dir('deploy', 'deploy')

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
    datas=project_datas + package_datas,
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


