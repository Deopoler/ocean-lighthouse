param(
    [string]$Blend = (Join-Path (Split-Path -Parent $PSScriptRoot) "scene.blend"),
    [int]$Start = 1,
    [int]$End = 240
)

$ErrorActionPreference = "Stop"

$blenderExe = "C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"
if (-not (Test-Path -LiteralPath $blenderExe)) {
    Write-Error "Blender executable not found at: $blenderExe"
    exit 1
}

if (-not (Test-Path -LiteralPath $Blend)) {
    Write-Error "Blend file not found at: $Blend"
    exit 1
}

$blendDir = Split-Path -Parent (Resolve-Path -LiteralPath $Blend)
$renderDir = Join-Path $blendDir "render"
$framesDir = Join-Path $renderDir "frames"
$logPath = Join-Path $renderDir "render.log"
$mp4Path = Join-Path $renderDir "ocean_lighthouse.mp4"

New-Item -ItemType Directory -Force -Path $renderDir | Out-Null

Write-Host "Rendering frames $Start-$End from $Blend"
Write-Host "Logging to $logPath"

& $blenderExe -b $Blend -s $Start -e $End -a 2>&1 | Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) {
    Write-Error "Blender exited with code $LASTEXITCODE. See $logPath for details."
    exit $LASTEXITCODE
}

$frameFiles = @()
if (Test-Path -LiteralPath $framesDir) {
    $frameFiles = Get-ChildItem -LiteralPath $framesDir -Filter "frame_*.png" -File |
        Where-Object { $_.Length -gt 0 } |
        Sort-Object Name
}

if ($frameFiles.Count -eq 0) {
    Write-Warning "No rendered PNG frames found in $framesDir; skipping MP4 encoding."
    exit 0
}

$ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($null -eq $ffmpegCmd) {
    Write-Host "Note: ffmpeg was not found on PATH. Skipping MP4 encoding."
    Write-Host "Once ffmpeg is available, encode with:"
    Write-Host "  ffmpeg -y -framerate 24 -start_number $Start -i `"$framesDir\frame_%04d.png`" -c:v libx264 -crf 16 -pix_fmt yuv420p `"$mp4Path`""
    exit 0
}

$framePattern = Join-Path $framesDir "frame_%04d.png"
Write-Host "Encoding $mp4Path from $framesDir ($($frameFiles.Count) frames)..."
& $ffmpegCmd.Source -y -framerate 24 -start_number $Start -i $framePattern -c:v libx264 -crf 16 -pix_fmt yuv420p $mp4Path
if ($LASTEXITCODE -ne 0) {
    Write-Error "ffmpeg exited with code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "Encoded video: $mp4Path"
