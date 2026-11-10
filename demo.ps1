# Chain of Draft 复现 Demo
# 用法：
#   1) 在本目录打开 PowerShell，执行 .\demo.ps1
#   2) 如果执行策略拦截，执行：
#      powershell -ExecutionPolicy Bypass -File .\demo.ps1
#   3) 需要分段暂停（录屏时配合 PPT 切换），执行：
#      .\demo.ps1 -Wait

param(
    [switch]$Wait
)

$ErrorActionPreference = 'Stop'

# 无论从哪里启动，都切到脚本所在目录
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Working directory: $(Get-Location)"
Write-Host ""

# 先检查正式运行产物是否存在
$required = @(
    'runs/full/metrics.csv',
    'runs/full/results.jsonl',
    'runs/full/manifest.json',
    'figures/full_metrics.png'
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Host "Missing required file: $path" -ForegroundColor Red
        exit 1
    }
}

Write-Host "============================================================"
Write-Host "1. Overall results: runs/full/metrics.csv"
Write-Host "============================================================"
Import-Csv runs/full/metrics.csv |
    Select-Object method, N,
        @{n='strict_acc';e={'{0:P0}' -f [double]$_.strict_accuracy}},
        @{n='official_acc';e={'{0:P0}' -f [double]$_.official_accuracy}},
        verdict_changes,
        mean_output_tokens,
        @{n='latency_s';e={[math]::Round([double]$_.mean_latency_seconds, 2)}},
        parse_errors |
    Format-Table -AutoSize

if ($Wait) {
    Read-Host "Press Enter to continue..." | Out-Null
}

Write-Host ""
Write-Host "============================================================"
Write-Host "2. Sample 1252: CoT vs CoD (runs/full/results.jsonl)"
Write-Host "============================================================"
python -c "import json; rows=[json.loads(l) for l in open('runs/full/results.jsonl',encoding='utf-8')]; rs=[x for x in rows if x['id']==1252 and x['method'] in ('cot','cod')]; [print(x['method'], x['output_tokens'], round(x['latency_seconds'],4), repr(x['output'][:90]), x['correct'], x['parse_error']) for x in rs]"

if ($Wait) {
    Read-Host "Press Enter to continue..." | Out-Null
}

Write-Host ""
Write-Host "============================================================"
Write-Host "3. Run manifest: runs/full/manifest.json"
Write-Host "============================================================"
python -c "import json; d=json.load(open('runs/full/manifest.json',encoding='utf-8')); print(d['phase'], d['completed'], d['config']['model_id'], d['config']['model_revision'][:8], d['config']['dtype'], d['config']['device'], len(d['sample_ids']))"

Write-Host ""
Write-Host "Figure: figures/full_metrics.png"
Write-Host "Demo complete."
