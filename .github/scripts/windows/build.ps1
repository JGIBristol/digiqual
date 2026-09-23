#Requires -Version 5.1
# Builds the Windows Digiqual.exe with PyInstaller and writes the .NET config
# needed to load assemblies from remote/OneDrive-synced directories.

$ErrorActionPreference = "Stop"

Set-Location "app"

# pythonnet/clr_loader are not direct dependencies of this app - they're pulled
# in transitively (with a sys_platform == 'win32' marker) by pywebview itself,
# and are already pinned in app/uv.lock. `uv run` below syncs them automatically,
# so no ad hoc/unpinned `uv pip install pythonnet` is needed here.
#
# PyInstaller flags live in digiqual.spec, shared with the macOS build, so the
# two platforms' bundling configuration can't silently drift apart.
uv run pyinstaller digiqual.spec --noconfirm

# Create .NET config to allow loading assemblies from OneDrive / remote directories
$configContent = @"
<configuration>
  <runtime>
    <loadFromRemoteSources enabled="true"/>
  </runtime>
</configuration>
"@
Set-Content -Path "dist/Digiqual/Digiqual.exe.config" -Value $configContent
