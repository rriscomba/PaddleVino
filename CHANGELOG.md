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
  CI now links Intel's official `Intel.ML.OnnxRuntime.OpenVino` NuGet
  package (downloaded as a plain zip, no .NET tooling needed) instead of
  the stock Microsoft CPU-only build, so `--engine openvino` should
  genuinely dispatch to OpenVINO in the release `.exe` — see README
  "Engine selection" for how this is packaged and the honesty caveat on
  what's confirmed vs. not yet confirmed by a live CI run.
- **CLI redesign**: replaced the base repo's short-flag `getopt`-style CLI
  with long flags (`--input`, `--engine`, `--format`, ...), directory and
  `--recursive` input support, and three output formats: JSON (default,
  full detail), `txt` (one detected text run per line with confidence),
  and `reading` (average confidence on the first line, then plain text
  re-flowed into reading order with same-line text runs merged, no boxes
  or labels).
- **Build**: replaced the base repo's third-party static OpenCV bundle
  download with vcpkg — see README "Build" for rationale.
- **Correctness fixes for the PP-OCRv6 model swap**: the base engine's
  preprocessing constants were still tuned for the old PP-OCRv3/v4 models
  it shipped with. Fixed two: the orientation classifier's fixed input
  size (was 48×192, the bundled PP-OCRv5 `ch_PP-LCNet_x0_25_textline_ori_cls_mobile`
  model needs 80×160 — this one crashed with an ONNX Runtime shape error),
  and the detector's pixel normalization (was ImageNet mean/std, the
  bundled PP-OCRv6 det model expects simple `(pixel/255-0.5)/0.5` — this
  one didn't crash, it silently degraded detection quality).
- **CI**: added `.github/workflows/build-windows.yml` building on
  `windows-latest`, bundling PP-OCRv6 models into the release zip, and
  publishing to GitHub Releases on `v*` tags.
- Out of scope for this release: PDF input (rasterize to images first),
  GPU backends other than OpenVINO (CUDA/DirectML code paths from the base
  repo are not wired into the new CLI).
