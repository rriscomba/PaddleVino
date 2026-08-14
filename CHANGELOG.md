# Changelog

## Unreleased

- Initial release of PaddleVino, adapted from
  [RapidAI/RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx).
- **Models: PP-OCRv3 → PP-OCRv6.** The base repo shipped PP-OCRv3 by
  default. PaddleVino targets PP-OCRv6 (det/rec/cls, tiny/small/medium
  tiers) via `models/download_models.ps1` / `.sh`. Known risk: PP-OCRv6's
  character dictionary is not compatible with the old
  `models/ppocr_keys_v1.txt` (PP-OCRv3) file kept in this repo for
  reference — see README "PP-OCRv6 models" for details. This has not been
  benchmarked against `paddleocr`/`rapidocr` Python output on the same
  images; treat recognition accuracy as unverified until you've run your
  own comparison.
- **New: OpenVINO execution provider backend** (`--engine cpu|openvino`).
  Implemented via ONNX Runtime's generic
  `SessionOptions::AppendExecutionProvider("OpenVINO", ...)` API across
  `DbNet`, `AngleNet`, and `CrnnNet`, with a safe fallback to CPU and a
  printed warning if the linked ONNX Runtime build lacks OpenVINO support.
  The default CI-built release does **not** link an OpenVINO-enabled ONNX
  Runtime (see README for why) — the flag is real but only takes effect
  against a self-built ONNX Runtime with `--use_openvino`.
- **CLI redesign**: replaced the base repo's short-flag `getopt`-style CLI
  with long flags (`--input`, `--engine`, `--format`, ...), directory and
  `--recursive` input support, and JSON (default) or plain-text output
  with per-line text, confidence, and bounding box.
- **Build**: replaced the base repo's third-party static OpenCV/ONNX
  Runtime bundle downloads with vcpkg (OpenCV) and the official Microsoft
  ONNX Runtime release package — see README "Build" for rationale.
- **CI**: added `.github/workflows/build-windows.yml` building on
  `windows-latest`, bundling PP-OCRv6 models into the release zip, and
  publishing to GitHub Releases on `v*` tags.
- Out of scope for this release: PDF input (rasterize to images first),
  GPU backends other than OpenVINO (CUDA/DirectML code paths from the base
  repo are not wired into the new CLI).
