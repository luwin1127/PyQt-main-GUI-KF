# mainGUI_alpha_release.spec
# -*- mode: python ; coding: utf-8 -*-

import os
import sys  # 添加 sys 模块导入

block_cipher = None

# 获取当前 .spec 文件所在目录
base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
icon_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'asserts', 'logo.ico')

# print(f"当前工作目录: {os.getcwd()}")
# print(f"spec 文件路径: {sys.argv[0]}")
# print(f"图标路径: {icon_path}")
# print(f"图标是否存在: {os.path.exists(icon_path)}")

a = Analysis(
    ['mainGUI_alpha_release.py'],
    datas=[
        (os.path.join(base_dir, 'asserts', '*'), 'asserts'),
        (os.path.join(base_dir, 'data', '*'), 'data'),
        (os.path.join(base_dir, 'figures', '*'), 'figures'),
        (os.path.join(base_dir, 'user_function', '*'), 'user_function'),
    ],
    pathex=[],  # 可添加额外的搜索路径
    binaries=[],
    hiddenimports=[],  # 手动添加可能缺失的导入
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='卡尔曼滤波仿真软件',  # 可执行文件名称
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 使用UPX压缩（需额外安装）
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # 设为False可隐藏控制台窗口
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path  # 添加图标路径参数
)