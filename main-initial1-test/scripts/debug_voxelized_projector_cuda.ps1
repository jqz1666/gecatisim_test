param(
    [string]$Python = "python",
    [string[]]$PythonArgs = @("-m", "unittest", "gecatsim.tests.test_catsim.test_Voxelized_GPU"),
    [switch]$SkipBuild,
    [switch]$Install,
    [switch]$Restore,
    [switch]$Run,
    [switch]$Sanitize,
    [switch]$KeepGoing
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ClibDir = Join-Path $RepoRoot "gecatsim\clib_build"
$Src = Join-Path $ClibDir "src\voxelized_projector_cuda.cu"
$LibDir = Join-Path $RepoRoot "gecatsim\lib"
$ReleaseDll = Join-Path $LibDir "catsim_voxel_projector_cuda.dll"
$DebugDll = Join-Path $LibDir "catsim_voxel_projector_cuda_debug.dll"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Require-Command($Name, $Hint) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "$Name was not found. $Hint"
    }
    return $cmd
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory = $RepoRoot) {
    Write-Host "Command: $FilePath $($Arguments -join ' ')"
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        if ($KeepGoing) {
            Write-Warning "$FilePath exited with code $($process.ExitCode)."
        } else {
            throw "$FilePath exited with code $($process.ExitCode)."
        }
    }
}

function Ensure-VisualStudioCompiler {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "cl.exe was not found, and vswhere.exe was not found. Open a Visual Studio x64 Native Tools prompt or install Visual Studio Build Tools with the C++ workload."
    }

    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) {
        throw "Visual Studio C++ build tools were not found."
    }

    $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "vcvars64.bat was not found under $install."
    }

    Write-Step "Loading Visual Studio x64 compiler environment"
    $envDump = cmd.exe /c "`"$vcvars`" >nul && set"
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

function Build-DebugDll {
    Write-Step "Checking CUDA and compiler tools"
    Require-Command "nvcc" "Install CUDA Toolkit and ensure nvcc is on PATH." | Out-Null
    Ensure-VisualStudioCompiler

    if (-not (Test-Path $Src)) {
        throw "CUDA source file was not found: $Src"
    }
    if (-not (Test-Path $LibDir)) {
        New-Item -ItemType Directory -Path $LibDir | Out-Null
    }

    Write-Host "nvcc version:"
    & nvcc --version

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        Write-Host ""
        Write-Host "GPU status:"
        & nvidia-smi
    } else {
        Write-Warning "nvidia-smi was not found; continuing because nvcc is available."
    }

    Write-Step "Building debug CUDA projector DLL"
    $args = @(
        "-G",
        "-g",
        "-lineinfo",
        "-Xcompiler=/Zi /MD",
        "-Xlinker=/DEBUG",
        "-shared",
        $Src,
        "-o",
        $DebugDll
    )
    Invoke-Checked "nvcc" $args $RepoRoot
    Write-Host "Built debug DLL: $DebugDll" -ForegroundColor Green
}

function Install-DebugDll {
    if (-not (Test-Path $DebugDll)) {
        throw "Debug DLL does not exist yet: $DebugDll. Run without -SkipBuild first."
    }

    Write-Step "Installing debug DLL as active CUDA projector"
    if (Test-Path $ReleaseDll) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backup = "$ReleaseDll.release_backup_$stamp"
        Copy-Item -LiteralPath $ReleaseDll -Destination $backup
        Write-Host "Backed up current DLL to: $backup"
    }
    Copy-Item -LiteralPath $DebugDll -Destination $ReleaseDll -Force
    Write-Host "Active DLL is now the debug build: $ReleaseDll" -ForegroundColor Green
}

function Restore-LatestBackup {
    Write-Step "Restoring latest backed-up release DLL"
    $backup = Get-ChildItem -LiteralPath $LibDir -Filter "catsim_voxel_projector_cuda.dll.release_backup_*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $backup) {
        throw "No release backup was found in $LibDir."
    }

    Copy-Item -LiteralPath $backup.FullName -Destination $ReleaseDll -Force
    Write-Host "Restored: $($backup.FullName) -> $ReleaseDll" -ForegroundColor Green
}

function Run-PythonTarget {
    Write-Step "Running Python target"
    Require-Command $Python "Set -Python to the Python executable for this environment." | Out-Null

    $runner = $Python
    $args = $PythonArgs
    if ($Sanitize) {
        $sanitizer = Get-Command compute-sanitizer -ErrorAction SilentlyContinue
        if (-not $sanitizer) {
            $sanitizer = Get-Command cuda-memcheck -ErrorAction SilentlyContinue
        }
        if (-not $sanitizer) {
            throw "Neither compute-sanitizer nor cuda-memcheck was found on PATH."
        }
        $runner = $sanitizer.Source
        $args = @("--tool", "memcheck", "--leak-check", "full", $Python) + $PythonArgs
    }

    Invoke-Checked $runner $args $RepoRoot
}

Write-Step "Debug setup for voxelized_projector_cuda.cu"
Write-Host "Repository: $RepoRoot"
Write-Host "Source:     $Src"
Write-Host "Debug DLL:  $DebugDll"
Write-Host "Active DLL: $ReleaseDll"

if ($Restore) {
    Restore-LatestBackup
    return
}

if (-not $SkipBuild) {
    Build-DebugDll
}

if ($Install) {
    Install-DebugDll
} else {
    Write-Host ""
    Write-Host "Tip: add -Install to replace the active DLL with the debug build after making a backup."
}

if ($Run -or $Sanitize) {
    Run-PythonTarget
} else {
    Write-Host ""
    Write-Host "Tip: add -Run to execute the default unittest target, or pass -PythonArgs for a custom script/test."
    Write-Host "Example: .\scripts\debug_voxelized_projector_cuda.ps1 -Install -Sanitize -PythonArgs @('sim.py')"
}
