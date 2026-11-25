# Shared helpers for the Windows WebUI launchers: list LAN addresses and make
# sure Windows Firewall lets other machines reach the WebUI port.
#
#   . (Join-Path $PSScriptRoot "lan.ps1")
#   Show-WebUiUrls -WebHost $WebHost -Port $Port -Label "Image WebUI"

function Get-LanAddresses {
  Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
      $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and
      $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress
}

function Enable-WebUiFirewall {
  param([int]$Port, [string]$Name)
  $ruleName = "Gufo WebUI $Name TCP $Port"
  if (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue) { return }
  $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port `
      -Action Allow -Profile Private, Domain | Out-Null
    Write-Host "Added firewall rule '$ruleName' (Private/Domain networks)." -ForegroundColor DarkGray
  } else {
    Write-Host ("If other devices cannot connect, allow Python through Windows Firewall when prompted, or run once as admin:`n" +
      "  New-NetFirewallRule -DisplayName '$ruleName' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private,Domain") -ForegroundColor DarkYellow
  }
}

function Show-WebUiUrls {
  param([string]$WebHost, [int]$Port, [string]$Label)
  Write-Host "${Label}: http://127.0.0.1:${Port}/" -ForegroundColor Green
  if ($WebHost -eq "0.0.0.0") {
    foreach ($ip in Get-LanAddresses) {
      Write-Host "${Label} (LAN): http://${ip}:${Port}/" -ForegroundColor Green
    }
    Enable-WebUiFirewall -Port $Port -Name $Label
  }
}
