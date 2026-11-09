# CPU jobs under the limits agreed with the RSNA session (10/14): Windows side only, Idle priority, <= 8 threads,
# <= 4 GB per job (killed at 3.9 GB), sequential, small logs on C:. Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File cpu_safe_run.ps1 -JobList "dir|script.py arg1;;dir|script2.py"
param([string]$JobList)   # jobs separated by ";;" (an array does not bind through -File)
$Jobs = $JobList -split ";;"
$py = "C:\Users\user\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$logdir = "C:\Users\user\tfb\work\cpu_logs"; New-Item -ItemType Directory -Force $logdir | Out-Null
$env:OMP_NUM_THREADS = "8"; $env:MKL_NUM_THREADS = "8"; $env:OPENBLAS_NUM_THREADS = "8"; $env:NUMEXPR_MAX_THREADS = "8"
$env:TFB_THREADS = "8"; $env:PYTHONIOENCODING = "utf-8"
$limit = 3.9GB
foreach ($j in $Jobs) {
    $dir, $cmd = $j.Split("|", 2)
    $name = ($cmd.Split(" ")[0]) -replace "\.py$", ""
    $out = Join-Path $logdir "$name.log"; $err = Join-Path $logdir "$name.err"
    Add-Content (Join-Path $logdir "runner.log") ("{0} start {1}" -f (Get-Date -Format s), $cmd)
    $p = Start-Process -FilePath $py -ArgumentList $cmd -WorkingDirectory $dir -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -WindowStyle Hidden
    try { $p.PriorityClass = "Idle" } catch {}
    $peak = 0
    while (-not $p.HasExited) {
        Start-Sleep -Seconds 2
        try { $p.Refresh(); $ws = $p.PrivateMemorySize64; if ($ws -gt $peak) { $peak = $ws } } catch { $ws = 0 }
        if ($ws -gt $limit) {
            Stop-Process -Id $p.Id -Force
            Add-Content (Join-Path $logdir "runner.log") ("{0} KILLED_MEM {1} at {2:N2} GB" -f (Get-Date -Format s), $cmd, ($ws / 1GB))
            break
        }
    }
    $p.WaitForExit(); Add-Content (Join-Path $logdir "runner.log") ("{0} end {1} exit={2} peak={3:N2} GB" -f (Get-Date -Format s), $cmd, $p.ExitCode, ($peak / 1GB))
}
Add-Content (Join-Path $logdir "runner.log") ("{0} ALLDONE" -f (Get-Date -Format s))
