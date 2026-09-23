#Requires -Version 5.1
# Comprehensive Windows Integrity, File & HTTP Startup Check.
# Verifies the PyInstaller bundle contains the expected files, then launches
# Digiqual.exe and confirms its embedded HTTP server comes online.

$ErrorActionPreference = "Stop"

Write-Host "1. Running Pre-Flight File Integrity Checks..."

# 1. Check for Primary Executable
$exePath = "app/dist/Digiqual/Digiqual.exe"
if (-not (Test-Path $exePath)) {
    Write-Error "Missing primary executable: $exePath"
    exit 1
}
Write-Host "  Found Digiqual.exe"

# 2. Check for App Config file
$configPath = "app/dist/Digiqual/Digiqual.exe.config"
if (-not (Test-Path $configPath)) {
    Write-Error "Missing configuration file: $configPath"
    exit 1
}
Write-Host "  Found Digiqual.exe.config"

# 3. Check for core Python DLL
$pyDll = Get-ChildItem -Path "app/dist/Digiqual" -Recurse -Filter "python3*.dll" | Select-Object -First 1
if (-not $pyDll) {
    Write-Error "Could not find python3*.dll inside build bundle!"
    exit 1
}
Write-Host "  Found Python DLL at: $($pyDll.FullName)"

# 4. Check for pythonnet runtime library
$pythonnetDll = Get-ChildItem -Path "app/dist/Digiqual" -Recurse -Filter "Python.Runtime.dll" | Select-Object -First 1
if (-not $pythonnetDll) {
    Write-Error "Missing Python.Runtime.dll required by pythonnet!"
    exit 1
}
Write-Host "  Found Python.Runtime.dll at: $($pythonnetDll.FullName)"

# 5. Log WebView2 Runtime presence (pywebview's edgechromium backend depends on it)
Write-Host "2. Checking for Microsoft Edge WebView2 Runtime..."
$webview2Key = "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
$webview2Info = Get-ItemProperty -Path $webview2Key -ErrorAction SilentlyContinue
if ($webview2Info -and $webview2Info.pv) {
    Write-Host "  WebView2 Runtime found: version $($webview2Info.pv)"
} else {
    Write-Warning "  WebView2 Runtime not detected via registry. pywebview's edgechromium backend may fail to start."
}

# 6. Launch the process and verify HTTP server connectivity
Write-Host "3. Launching Digiqual.exe to test HTTP server and network connectivity..."
$logDir = "app/dist"
$stdoutLog = Join-Path $logDir "digiqual-stdout.log"
$stderrLog = Join-Path $logDir "digiqual-stderr.log"
$proc = Start-Process -FilePath $exePath -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog

$serverOnline = $false
$maxAttempts = 20
$attempt = 0
$boundPort = $null

while ($attempt -lt $maxAttempts -and -not $serverOnline) {
    Start-Sleep -Seconds 2
    $attempt++

    # Ensure process has not crashed
    if ($proc.HasExited) {
        Write-Error "Digiqual.exe exited prematurely with exit code $($proc.ExitCode)!"
        exit 1
    }

    # Detect the TCP port opened by the background process
    $connections = Get-NetTCPConnection -OwningProcess $proc.Id -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        $boundPort = $connections[0].LocalPort
        Write-Host "  Detected listening port: $boundPort (Attempt $attempt/$maxAttempts)"

        try {
            # Test local loopback connection while bypassing system proxy settings
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:$boundPort" -NoProxy -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                Write-Host "  HTTP 200 OK received from http://127.0.0.1:$boundPort"
                $serverOnline = $true
                break
            }
        } catch {
            Write-Host "  Waiting for HTTP response on port $boundPort... ($($_.Exception.Message))"
        }
    } else {
        Write-Host "  Waiting for Digiqual.exe to bind a TCP port... (Attempt $attempt/$maxAttempts)"
    }
}

# Cleanly terminate the background test process
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue

if (-not $serverOnline) {
    Write-Error "Failed to connect to the internal Shiny server within timeout! See uploaded digiqual-stdout.log / digiqual-stderr.log for details."
    exit 1
}

Write-Host "All Windows verification and HTTP health checks passed successfully!"
