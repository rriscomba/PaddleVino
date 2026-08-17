# Changelog

## Unreleased (0.2.0)

- **License change: MIT → AGPL-3.0.** Required by the optional checkbox
  detection model (YOLO12n from
  [wendys-llc/checkbox-detector](https://huggingface.co/wendys-llc/checkbox-detector),
  exported by Ultralytics and declaring AGPL-3.0). The inherited OCR engine
  stays Apache-2.0 and keeps its notice in `THIRD_PARTY_LICENSES/`; the
  model's license text was added there too. Everything already published
  under MIT remains MIT permanently — the change applies going forward only.
- **New: cell structure detection** (`--detect-cells`). Finds field boxes and
  table cells by morphology — no model, no new dependency, no extra download.
  Reports them under a new JSON `cells` key, and in `--format reading`
  collapses the text runs inside a cell into a single element, which stops a
  two-line field from dragging its whole row out of alignment. All six
  morphology parameters are exposed as flags (`--cell-h-frac`,
  `--cell-v-frac`, `--cell-min-width`, `--cell-min-height`,
  `--cell-max-area`, `--cell-rectangularity`). Measured on the reference
  documents: 18 / 17 / 55 cells on three forms, and **0** on a plain-text
  page.
- **New: checkbox detection** (`--detect-checkboxes`, model file named by
  `--checkbox-model`, default `checkbox.onnx` in `--models-dir`). A YOLO12n
  model says what and where; ink density inside the box says whether it is
  ticked. Adds `document_type` and a `checkboxes` key (with `ink_ratio`,
  `confidence`, `source` and `snapped` for traceability) to the JSON, and
  injects each checkbox into `--format reading` as `[x]` / `[ ]` at its real
  position. Runs through the same ONNX Runtime path as the OCR models, so it
  honours `--engine openvino`. Three false-positive filters: mean HSV
  saturation, a document-type gate that only enables the low-confidence
  rescue once the conservative pass has already found several boxes, and
  snapping the network's loose box to the real rectangle before measuring
  ink. Every calibrated constant is a flag; `--checkbox-profile
  strict|balanced|aggressive` gives presets, and any explicit flag overrides
  the profile.
- **New: diagnostics.** `--debug-overlay <file>` writes the page with the
  detected boxes drawn on it (checkboxes green/red with their ink ratio and
  confidence, cells in blue); `--debug-checkbox-candidates` also emits the
  discarded candidates in the JSON with the reason each was dropped
  (`saturation`, `below-conf`, `rescue-pruned`, `nms`).
- All of the above default to off. With no new flag passed, `json`, `txt` and
  `reading` output is byte-for-byte what 0.1.0 produced (verified by diffing
  both binaries on the same images).
- `models/download_models.ps1` / `.sh` gained a `-Checkbox` / `checkbox`
  switch for the checkbox model, and the release zip now always bundles it,
  so `--detect-checkboxes` works straight out of the downloaded package.
  It was briefly behind an opt-in CI dispatch input, which could never be
  set on the tag pushes that publish a release — the feature shipped but
  the model it needs could not.
- Known limits: these detectors assume natively digital documents (straight
  borders, no skew, no scanner noise); there is no deskew step. Recall on the
  reference forms is 100% / 100% / 80%, with 3 achromatic false positives on
  the dense one that the saturation filter cannot catch.

## 0.1.0

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
