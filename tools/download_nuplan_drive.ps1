<#
Download the nuPlan training archives from the team Google Drive folder to a local disk.

One-time setup (opens a browser for Google login; use the account the folder is shared with):
    rclone config create gdrive drive scope=drive.readonly

Then, from the repository root:
    powershell -ExecutionPolicy Bypass -File tools/download_nuplan_drive.ps1
    powershell -ExecutionPolicy Bypass -File tools/download_nuplan_drive.ps1 -Include "nuplan-v1.1_train_pittsburgh.zip"

rclone resumes interrupted transfers at the file level and skips files that are already
complete (same size), so the script is safe to re-run. Archives already present in
-SkipDir (default: Downloads) are not downloaded again.
#>
param(
    [string]$Remote = "gdrive",
    # The team Drive folder ID is not committed (public repository; nuPlan data may not be
    # redistributed). Pass -FolderId, or set NUPLAN_DRIVE_FOLDER_ID in the environment or .env.
    [string]$FolderId = $env:NUPLAN_DRIVE_FOLDER_ID,
    [string]$Dest = "D:\nuplan\downloads",
    [string]$SkipDir = "$HOME\Downloads",
    [string[]]$Include = @("nuplan-v1.1_train_*.zip"),
    [int]$Transfers = 4
)

$ErrorActionPreference = "Stop"
if (-not $FolderId -and (Test-Path ".env")) {
    $line = Get-Content ".env" | Where-Object { $_ -match '^\s*NUPLAN_DRIVE_FOLDER_ID\s*=' } | Select-Object -First 1
    if ($line) { $FolderId = ($line -split '=', 2)[1].Trim().Trim('"').Trim("'") }
}
if (-not $FolderId) { throw "Drive folder ID missing. Pass -FolderId or set NUPLAN_DRIVE_FOLDER_ID in .env (ask the data owners)." }
if (-not (Get-Command rclone -ErrorAction SilentlyContinue)) { throw "rclone not found. Install with: winget install Rclone.Rclone" }
if (-not ((rclone listremotes) -contains "${Remote}:")) { throw "rclone remote '$Remote' missing. Run: rclone config create $Remote drive scope=drive.readonly" }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# Build an explicit file list: remote names matching -Include, minus archives already local
# (e.g. the Boston ZIP in Downloads). Literal names avoid glob issues with "(2)" and spaces.
# "powershell -File" passes "a,b" as one string; split it so comma-separated lists work.
$Include = @($Include | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$remoteNames = rclone lsf "${Remote}:" --drive-root-folder-id $FolderId --files-only
$local = @(Get-ChildItem -Path $SkipDir -Filter "nuplan-v1.1_train_*.zip" -ErrorAction SilentlyContinue | ForEach-Object Name)
$wanted = $remoteNames | Where-Object { $n = $_; ($Include | Where-Object { $n -like $_ }) -and ($local -notcontains $n) }
foreach ($n in $local) { Write-Host "Already local, skipping: $n" }
if (-not $wanted) { Write-Host "Nothing to download."; exit 0 }
$listFile = Join-Path $Dest "download_list.txt"
$wanted | Set-Content -Path $listFile -Encoding ascii
$filters = @("--files-from", $listFile)

Write-Host "Files to fetch from Drive:"
rclone lsl "${Remote}:" --drive-root-folder-id $FolderId @filters

rclone copy "${Remote}:" $Dest `
    --drive-root-folder-id $FolderId @filters `
    --transfers $Transfers --order-by size,ascending --multi-thread-streams 8 `
    --retries 10 --low-level-retries 20 --progress --stats 30s `
    --log-file (Join-Path $Dest "rclone_download.log") --log-level INFO

Write-Host "`nVerifying sizes against Drive..."
rclone check "${Remote}:" $Dest --drive-root-folder-id $FolderId @filters --size-only --one-way
