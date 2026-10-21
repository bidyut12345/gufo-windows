param(
    [string]$Tag = "v0.20.0",
    [string]$Repo = "bidyut12345/gufo-windows",
    [string[]]$Files = @(),
    [string]$TargetCommitish = "",
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

# Retrieve token from git credential manager
$inputStr = "protocol=https`nhost=github.com`n`n"
$res = $inputStr | git credential fill
$tokenLine = $res | Where-Object { $_ -like "password=*" }
if (-not $tokenLine) { throw "No GitHub token found in git credential manager" }
$token = $tokenLine.Substring(9).Trim()
$env:GH_TOKEN = $token
$env:GH_PROMPT_DISABLED = "1"

$headers = @{
    "Authorization" = "token $token"
    "Accept"        = "application/vnd.github.v3+json"
}

$notesPath = if (Test-Path "dist\RELEASE_NOTES.md") { (Resolve-Path "dist\RELEASE_NOTES.md").Path } else { "" }
$bodyText = if ($notesPath) { [System.IO.File]::ReadAllText($notesPath, [System.Text.Encoding]::UTF8) } else { "Release $Tag" }

Write-Host "Checking for release '$Tag' in $Repo..." -ForegroundColor Cyan
$release = $null
try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Tag" -Headers $headers -Method Get
} catch {
    # 404 is expected if release does not exist yet
}

if (-not $release) {
    Write-Host "Creating release '$Tag' on GitHub..." -ForegroundColor Yellow
    $createPayload = @{
        tag_name   = $Tag
        name       = "Gufo for Windows $Tag"
        body       = $bodyText
        draft      = $false
        prerelease = $false
    }
    if ($TargetCommitish) {
        $createPayload["target_commitish"] = $TargetCommitish
    }
    $jsonPayload = $createPayload | ConvertTo-Json -Depth 5
    $jsonBytes = [System.Text.Encoding]::UTF8.GetBytes($jsonPayload)
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases" -Headers $headers -Method Post -Body $jsonBytes -ContentType "application/json; charset=utf-8"
    Write-Host "Created release ID: $($release.id)" -ForegroundColor Green
} else {
    Write-Host "Found existing release ID: $($release.id)" -ForegroundColor Green
}

# Upload files
foreach ($f in $Files) {
    if (-not (Test-Path $f)) {
        Write-Warning "File not found: $f"
        continue
    }
    $resolvedPath = (Resolve-Path $f).Path
    $fileName = [System.IO.Path]::GetFileName($resolvedPath)
    $fileSizeMB = [math]::Round(((Get-Item $resolvedPath).Length / 1MB), 2)

    # Check if asset already exists in release
    $existing = $release.assets | Where-Object { $_.name -eq $fileName }
    if ($existing) {
        Write-Host "Removing existing asset '$fileName' (ID: $($existing.id))..." -ForegroundColor Yellow
        Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/assets/$($existing.id)" -Headers $headers -Method Delete
    }

    Write-Host "Uploading $fileName ($fileSizeMB MB) to release $Tag via gh release upload..." -ForegroundColor Cyan
    & gh release upload $Tag $resolvedPath --repo $Repo --clobber
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload $fileName (gh exit code $LASTEXITCODE)"
    }
    Write-Host "`nSuccessfully uploaded $fileName!" -ForegroundColor Green
}

Write-Host "`nRelease URL: $($release.html_url)" -ForegroundColor Green
