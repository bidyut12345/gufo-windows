param(
    [string]$Tag = "v0.1.0-windows",
    [string]$Repo = "bidyut12345/gufo-windows",
    [string[]]$Files = @(
        "dist\gufo-windows-v0.1.0-windows.zip",
        "dist\gufo-windows-v0.1.0-bin.zip"
    ),
    [switch]$Force = $false
)

# Retrieve token from git credential manager
$inputStr = "protocol=https`nhost=github.com`n`n"
$res = $inputStr | git credential fill
$tokenLine = $res | Where-Object { $_ -like "password=*" }
if (-not $tokenLine) { throw "No GitHub token found in git credential manager" }
$token = $tokenLine.Substring(9)

$apiBase = "https://api.github.com"
$headers = @{
    "Authorization" = "Bearer $token"
    "User-Agent" = "Gufo-Release-Uploader"
    "Accept" = "application/vnd.github+json"
}

Write-Host "Fetching release for tag '$Tag' from $Repo..."
$release = $null
try {
    $release = Invoke-RestMethod -Uri "$apiBase/repos/$Repo/releases/tags/$Tag" -Headers $headers -Method Get
    Write-Host "Found existing release: ID=$($release.id), Name='$($release.name)'"
} catch {
    Write-Host "Release for tag '$Tag' does not exist yet. Creating..."
    $releaseNotes = if (Test-Path "dist\RELEASE_NOTES.md") { [System.IO.File]::ReadAllText((Resolve-Path "dist\RELEASE_NOTES.md").Path) } else { "Release $Tag" }
    $createBody = @{
        tag_name = $Tag
        target_commitish = "main"
        name = "Gufo for Windows $Tag"
        body = $releaseNotes
        draft = $false
        prerelease = $false
    } | ConvertTo-Json

    $release = Invoke-RestMethod -Uri "$apiBase/repos/$Repo/releases" -Headers $headers -Method Post -Body $createBody -ContentType "application/json"
    Write-Host "Created release ID=$($release.id)"
}

$releaseId = $release.id

# Refresh release info to get current assets
$release = Invoke-RestMethod -Uri "$apiBase/repos/$Repo/releases/$releaseId" -Headers $headers -Method Get
$existingAssets = $release.assets

foreach ($filePath in $Files) {
    if (-not (Test-Path $filePath)) {
        Write-Warning "File not found: $filePath, skipping."
        continue
    }

    $fileName = [System.IO.Path]::GetFileName($filePath)
    $resolvedPath = (Resolve-Path $filePath).Path
    $fileSizeMB = [math]::Round((Get-Item $resolvedPath).Length / 1MB, 2)

    # Check for existing asset
    if ($existingAssets) {
        $found = $existingAssets | Where-Object { $_.name -eq $fileName }
        if ($found) {
            if (-not $Force) {
                Write-Host "Asset '$fileName' already exists on release ($([math]::Round($found.size / 1MB, 2)) MB). Skipping (use -Force to re-upload)." -ForegroundColor Yellow
                continue
            }
            Write-Host "Deleting existing asset '$fileName' (ID: $($found.id))..."
            try {
                Invoke-RestMethod -Uri "$apiBase/repos/$Repo/releases/assets/$($found.id)" -Headers $headers -Method Delete
                Write-Host "Deleted existing asset."
                Start-Sleep -Seconds 3
            } catch {
                Write-Warning "Failed to delete asset $($found.id): $_"
            }
        }
    }

    Write-Host "Uploading $fileName ($fileSizeMB MB) with curl..."
    $uploadUrl = "https://uploads.github.com/repos/$Repo/releases/$releaseId/assets?name=$fileName"

    $authHeader = "Authorization: Bearer $token"
    $acceptHeader = "Accept: application/vnd.github+json"
    $typeHeader = "Content-Type: application/zip"

    Start-Sleep -Seconds 2
    & curl.exe -# -X POST `
        --retry 3 `
        --retry-delay 5 `
        -H $authHeader `
        -H $acceptHeader `
        -H $typeHeader `
        --data-binary "@$resolvedPath" `
        $uploadUrl

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nUploaded $fileName successfully!" -ForegroundColor Green
    } else {
        Write-Error "Failed to upload $fileName, curl exit code: $LASTEXITCODE"
    }
}

Write-Host "`nAll release checks completed!" -ForegroundColor Green
$relFinal = Invoke-RestMethod -Uri "$apiBase/repos/$Repo/releases/$releaseId" -Headers $headers -Method Get
Write-Host "Release URL: $($relFinal.html_url)"
Write-Host "Assets:"
foreach ($a in $relFinal.assets) {
    Write-Host " - $($a.name): $([math]::Round($a.size / 1MB, 2)) MB -> $($a.browser_download_url)"
}
