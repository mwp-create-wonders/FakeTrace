param(
  [Parameter(Mandatory=$true)][string]$Config,
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [string]$Manifest = "",
  [string]$AudioDir = "",
  [string]$Device = "cuda",
  [double]$FakeThreshold = 0.5,
  [string]$ThresholdSummary = "",
  [switch]$SaveProbs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
  $args = @(
    ".\audio_app.py", "audio-predict",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--fake-threshold", "$FakeThreshold"
  )
  if ($ThresholdSummary) {
    $args += @("--threshold-summary", $ThresholdSummary)
  }
  if ($Manifest) {
    $args += @("--manifest", $Manifest)
  }
  elseif ($AudioDir) {
    $args += @("--audio-dir", $AudioDir)
  }
  else {
    throw "Provide either -Manifest or -AudioDir."
  }
  if ($SaveProbs) {
    $args += "--save-probs"
  }
  python @args
}
finally {
  Pop-Location
}
