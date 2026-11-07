# Run from Windows PowerShell. Only the repository's exact USB-CAN is attached.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$distribution = 'Ubuntu-22.04'
$linuxProject = '/home/kaiwen/OSCBF-learn'
$instanceId = 'USB\VID_1D50&PID_606F\004F00374548570120353731'
$usbipd = (Get-Command usbipd.exe -ErrorAction Stop).Source

function Get-PiperUsb {
    $raw = & $usbipd state
    if ($LASTEXITCODE -ne 0) { throw 'usbipd state failed.' }
    $matches = @((($raw -join "`n") | ConvertFrom-Json).Devices |
        Where-Object { $_.InstanceId -eq $instanceId -and $_.BusId })
    if ($matches.Count -ne 1) { throw 'The expected Piper USB-CAN is not connected to Windows.' }
    return $matches[0]
}

& wsl.exe -d $distribution -- true
if ($LASTEXITCODE -ne 0) { throw 'Unable to start the Ubuntu WSL distribution.' }
$device = Get-PiperUsb
if (-not $device.PersistedGuid) {
    $binding = Start-Process -FilePath $usbipd -Verb RunAs -WindowStyle Hidden `
        -ArgumentList @('bind', '--busid', $device.BusId) -Wait -PassThru
    if ($binding.ExitCode -ne 0) { throw 'Windows USB sharing was not completed.' }
    $device = Get-PiperUsb
}
if (-not $device.ClientIPAddress) {
    & $usbipd attach --wsl $distribution --busid $device.BusId
    if ($LASTEXITCODE -ne 0) { throw 'USB attachment failed.' }
}
& wsl.exe -d $distribution -u root -- bash "$linuxProject/scripts/wsl_can_up.sh"
if ($LASTEXITCODE -ne 0) { throw 'CAN setup failed. Check the kernel module and USB identity.' }
& wsl.exe -d $distribution -- env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME `
    "$linuxProject/envs/piper-sdk/bin/python" -I "$linuxProject/scripts/read_joint_state.py"
if ($LASTEXITCODE -ne 0) { throw 'CAN is attached, but fresh Piper feedback was not verified. No robot command was sent.' }
