param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$GhArgs
)
$inputStr = "protocol=https`nhost=github.com`n`n"
$res = $inputStr | git credential fill
$tokenLine = $res | Where-Object { $_ -like "password=*" }
if ($tokenLine) {
    $env:GH_TOKEN = $tokenLine.Substring(9)
}
& C:\tools\gh.exe @GhArgs
