# Downloads the official PP-OCRv6 ONNX models (published by the RapidOCR
# project, re-packaged from PaddleOCR/PaddlePaddle weights) into this
# directory, alongside the matching character dictionary.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File models/download_models.ps1 [-Tier small|tiny|medium]
#
# Source of truth for these URLs:
#   https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
# Models are hosted on ModelScope under the RapidAI/RapidOCR model repo.

param(
    [ValidateSet("tiny", "small", "medium")]
    [string]$Tier = "small",
    [string]$DestDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$base = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"

$detUrl = "$base/onnx/PP-OCRv6/det/PP-OCRv6_det_$Tier.onnx"
$recUrl = "$base/onnx/PP-OCRv6/rec/PP-OCRv6_rec_$Tier.onnx"
# The angle/orientation classifier is not re-released per PP-OCRv6 tier;
# RapidOCR's PP-OCRv6 pipeline reuses the PP-OCRv5 PP-LCNet text-line
# orientation classifier.
$clsUrl = "$base/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"

# IMPORTANT: the character dictionary is tied to the rec model's tier.
# The "tiny" rec model uses a different (smaller) dictionary than
# "small"/"medium". Always download the dictionary that matches -Tier.
if ($Tier -eq "tiny") {
    $dictUrl = "$base/paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt"
} else {
    $dictUrl = "$base/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt"
}

New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

function Get-File($Url, $OutFile) {
    Write-Host "Downloading $Url -> $OutFile"
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
}

Get-File $detUrl  (Join-Path $DestDir "det.onnx")
Get-File $recUrl  (Join-Path $DestDir "rec.onnx")
Get-File $clsUrl  (Join-Path $DestDir "cls.onnx")
Get-File $dictUrl (Join-Path $DestDir "ppocrv6_dict.txt")

Write-Host "Done. Models placed in $DestDir (tier: $Tier)."
