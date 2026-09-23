# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the Digiqual desktop app.

Single source of truth for both the macOS and Windows CI builds (see
.github/scripts/windows/build.ps1 and the macOS build step in
.github/workflows/build_app.yml) so the two platforms' PyInstaller
configuration can no longer silently drift apart.
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

ENTRY_POINT = os.path.join(SPECPATH, "run_app.py")

# Packages collected in full (submodules, data files and binaries) on every
# platform - mirrors the shared `--collect-all` flags the two platforms used
# to specify separately.
COLLECT_ALL_PACKAGES = [
    "digiqual",
    "shiny",
    "faicons",
    "shinyswatch",
    "htmltools",
    "pywebview",
    "matplotlib",
]

# pythonnet/clr_loader back pywebview's Windows-only edgechromium GUI backend;
# they don't exist as installable packages on macOS/Linux.
if sys.platform == "win32":
    COLLECT_ALL_PACKAGES += ["pythonnet", "clr_loader"]

HIDDEN_IMPORTS = [
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.lifespan.on",
    "engineio.async_drivers.threading",
]

datas = []
binaries = []
hiddenimports = list(HIDDEN_IMPORTS)

for package in COLLECT_ALL_PACKAGES:
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# `--collect-binaries webview` only pulled in webview's binaries, not its full
# data/hiddenimports set (those already come from the pywebview collect-all
# above, since "webview" is pywebview's import name).
binaries += collect_dynamic_libs("webview")

a = Analysis(
    [ENTRY_POINT],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Digiqual",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    upx=True,
    upx_exclude=[],
    name="Digiqual",
)

# `--windowed` on macOS additionally wraps the collected onedir build into a
# proper .app bundle; a .spec file must do this explicitly.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Digiqual.app",
        icon=None,
        bundle_identifier=None,
    )
