param(
    [string]$Config = "configs/main.yaml",
    [string]$Python = ".venv/Scripts/python.exe",
    [switch]$SkipReport,
    [string]$Tectonic = "",
    [string]$XeLaTex = "xelatex",
    [string]$BibTex = "bibtex"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not $SkipReport) {
        & $Python -m hs_safety.reporting --config $Config
        if ($LASTEXITCODE -ne 0) { throw "Report generation failed with exit code $LASTEXITCODE" }
    }
    # 优先在项目根目录解析可执行文件，避免切换到 papers 后丢失相对路径。
    if ($Tectonic) {
        $tectonicCommand = (Get-Command $Tectonic -ErrorAction Stop).Source
        # 缓存随项目保存，避免切换目录或新终端后误用用户级空缓存。
        if (-not $env:TECTONIC_CACHE_DIR) {
            $env:TECTONIC_CACHE_DIR = Join-Path $projectRoot ".cache/tectonic"
        }
    } else {
        $texCommand = (Get-Command $XeLaTex -ErrorAction Stop).Source
        $bibCommand = (Get-Command $BibTex -ErrorAction Stop).Source
    }
    Push-Location -LiteralPath (Join-Path $projectRoot "papers")
    try {
        foreach ($paperName in @("paper_en", "paper_zh")) {
            if ($Tectonic) {
                # Tectonic 使用 XeTeX 引擎并自动处理 BibTeX 和必要的重复编译。
                & $tectonicCommand --keep-logs --keep-intermediates "$paperName.tex"
                if ($LASTEXITCODE -ne 0) { throw "Tectonic build failed: $paperName" }
                Write-Host "Built papers/$paperName.pdf"
                continue
            }
            & $texCommand -interaction=nonstopmode -halt-on-error -file-line-error "$paperName.tex"
            if ($LASTEXITCODE -ne 0) { throw "Initial XeLaTeX pass failed: $paperName" }
            & $bibCommand $paperName
            if ($LASTEXITCODE -ne 0) { throw "BibTeX failed: $paperName" }
            for ($passNumber = 1; $passNumber -le 2; $passNumber++) {
                & $texCommand -interaction=nonstopmode -halt-on-error -file-line-error "$paperName.tex"
                if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $passNumber failed: $paperName" }
            }
            Write-Host "Built papers/$paperName.pdf"
        }
    } finally { Pop-Location }
} finally { Pop-Location }
