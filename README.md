# PaddleVino

A native Windows CLI OCR tool — no Python runtime required. It runs
PP-OCRv6 detection/classification/recognition models through ONNX Runtime,
with a switchable execution backend: plain CPU, or the OpenVINO execution
provider for Intel CPU/iGPU acceleration.

It builds automatically in the cloud on every push via GitHub Actions
(`windows-latest`), producing a downloadable, self-contained `.exe` — no
Windows machine is required to build it yourself.

## Origin

This project is adapted from [RapidAI/RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx),
a pure C++ (no Python) OCR engine built on ONNX Runtime + OpenCV, with
Windows/Linux/macOS build support already in place. Its detection/angle/
recognition pipeline (`DbNet`, `AngleNet`, `CrnnNet`, `OcrLite*`, the
Clipper polygon library, and general image utilities) is the base this
repository builds on. Changes made here:

- Bumped the default models from PP-OCRv3 to **PP-OCRv6**.
- Redesigned the CLI around long flags with JSON output (see below).
- Added an OpenVINO execution-provider selection path (`--engine openvino`).
- Replaced the base repo's third-party static OpenCV/ONNX Runtime download
  mechanism with vcpkg (OpenCV) + the official Microsoft ONNX Runtime
  release package (see "Build" below for why).
- Added a GitHub Actions workflow that builds a Windows release entirely in
  the cloud, bundling official PP-OCRv6 weights into the shipped package.

## License

- This repository's own code is MIT-licensed — see [`LICENSE`](LICENSE).
- The OCR engine (`src/`, `include/`) is adapted from
  [RapidAI/RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx), which is
  Apache-2.0. That license is preserved at
  [`THIRD_PARTY_LICENSES/RapidOcrOnnx-LICENSE-Apache-2.0.txt`](THIRD_PARTY_LICENSES/RapidOcrOnnx-LICENSE-Apache-2.0.txt)
  and continues to apply to the derived engine source files.
- Model weights: PP-OCRv6 models are published by the PaddleOCR project
  (Baidu/PaddlePaddle), free to use for inference; redistributed here in
  ONNX form via the RapidAI/RapidOCR project's model mirror on ModelScope.
  Review PaddleOCR's model license before commercial redistribution.

## CLI usage

```
paddlevino.exe --input imagen.png --engine openvino --output resultado.json
paddlevino.exe --input carpeta\ --recursive --format json
paddlevino.exe --input photo.jpg --format txt
paddlevino.exe --input scan.png --format reading
```

| Flag | Description | Default |
| --- | --- | --- |
| `--input <path>` | Image file or a directory of images (**required**) | — |
| `--recursive` | When `--input` is a directory, search it recursively | off |
| `--output <path>` | Write results to this file instead of stdout | stdout |
| `--format json\|txt\|reading` | Output format (see below) | `json` |
| `--engine cpu\|openvino` | Execution backend | `cpu` |
| `--models-dir <dir>` | Directory containing the model files | `models` |
| `--det <file>` | Detection model file name | `det.onnx` |
| `--cls <file>` | Angle classification model file name | `cls.onnx` |
| `--rec <file>` | Recognition model file name | `rec.onnx` |
| `--keys <file>` | Character dictionary file name | `ppocrv6_dict.txt` |
| `--threads <int>` | ONNX Runtime thread count | `4` |
| `--padding <int>` | Border padding added to input images | `50` |
| `--max-side-len <int>` | Resize long side to this value, `0` = no resize | `1024` |
| `--box-score-thresh <f>` | Detection box score threshold | `0.5` |
| `--box-thresh <f>` | Detection binarization threshold | `0.3` |
| `--unclip-ratio <f>` | Detection box expansion ratio | `1.6` |
| `--no-angle` | Disable the angle classification model | angle enabled |
| `--no-most-angle` | Disable "most probable angle" voting | enabled |
| `--reading-row-overlap <f>` | `--format reading` row-clustering threshold (see below) | `0.5` |
| `--version`, `-v` | Print version and exit | — |
| `--help`, `-h` | Print usage and exit | — |

Cell-structure detection (see "Cell structure detection" below). All of these
are off by default, so behaviour without them is unchanged:

| Flag | Description | Default |
| --- | --- | --- |
| `--detect-cells` | Detect field/table cells and report them under the JSON `cells` key | off |
| `--cell-h-frac <int>` | Horizontal kernel divisor (`width/N`) | `28` |
| `--cell-v-frac <int>` | Vertical kernel divisor (`height/N`) | `80` |
| `--cell-min-width <f>` | Minimum cell width as a fraction of the page width | `0.012` |
| `--cell-min-height <f>` | Minimum cell height as a fraction of the page height | `0.006` |
| `--cell-max-area <f>` | Maximum cell area as a fraction of the page | `0.6` |
| `--cell-rectangularity <f>` | Minimum `contourArea / boundingBoxArea` | `0.7` |
| `--reading-column-gap <f>` | Horizontal gap (in line heights) above which `--format reading` separates two runs with `\|`; only applies with `--detect-cells` | `0.8` |

Checkbox detection (see "Checkbox detection" below). Also all off by default:

| Flag | Description | Default |
| --- | --- | --- |
| `--detect-checkboxes` | Detect checkboxes and whether they are ticked | off |
| `--checkbox-model <file>` | Model file name inside `--models-dir` | `checkbox.onnx` |
| `--checkbox-conf <f>` | Confidence threshold of the main pass | `0.25` |
| `--checkbox-iou <f>` | NMS IoU, crossing classes | `0.45` |
| `--checkbox-max-saturation <f>` | Maximum mean HSV saturation; drops colour logos | `20.0` |
| `--checkbox-input-size <int>` | Model input side (letterboxed) | `1024` |
| `--checkbox-form-min <int>` | Boxes the conservative pass must find for the document to count as a form; `0` disables the gate | `3` |
| `--no-checkbox-rescue` | Disable the low-confidence rescue | rescue on |
| `--checkbox-rescue-conf <f>` | Minimum confidence of a rescue candidate | `0.04` |
| `--checkbox-rescue-min-cluster <int>` | Aligned candidates needed in one X column | `2` |
| `--checkbox-rescue-x-tol <f>` | Pixel tolerance for grouping candidates into a column | `20.0` |
| `--checkbox-rescue-spacing-tol <f>` | Row-spacing tolerance (multiple of the median gap) | `1.6` |
| `--checkbox-dedup-y-frac <f>` | Y distance (fraction of height) below which two detections are the same object | `0.6` |
| `--no-checkbox-snap` | Use the raw network box without snapping | snap on |
| `--checkbox-snap-margin <int>` | Pixel margin around the box when looking for the real rectangle | `4` |
| `--checkbox-snap-min-area <f>` | Minimum candidate area as a fraction of the search region | `0.07` |
| `--checkbox-snap-max-area <f>` | Maximum, likewise | `0.95` |
| `--checkbox-snap-min-aspect <f>` | Minimum accepted aspect | `0.4` |
| `--checkbox-snap-max-aspect <f>` | Maximum accepted aspect (wide boxes exist, ~2.4) | `3.0` |
| `--checkbox-snap-rectangularity <f>` | Minimum candidate rectangularity | `0.75` |
| `--checkbox-ink-thresh <f>` | Ink density above which a box counts as ticked | `0.05` |
| `--checkbox-ink-border <f>` | Fraction of the side cropped inwards to exclude the border | `0.25` |
| `--checkbox-ink-dark <int>` | Grey level below which a pixel counts as ink | `128` |

Diagnostics:

| Flag | Description | Default |
| --- | --- | --- |
| `--debug-overlay <file>` | Write a copy of the page with the detected boxes drawn on it: checkboxes green (ticked) / red (empty) with their ink ratio and confidence, cells in blue | off |
| `--debug-checkbox-candidates` | Also emit the discarded candidates in the JSON, with the reason each was dropped | off |

JSON output is an array with one entry per processed image:

```json
[
  {
    "file": "imagen.png",
    "detect_time_ms": 123.4,
    "lines": [
      {
        "text": "Hola mundo",
        "confidence": 0.97,
        "box_score": 0.91,
        "box": [[10,10],[120,10],[120,40],[10,40]]
      }
    ]
  }
]
```

`--format txt` prints one line per detected text run, each tagged with its
own confidence — a text run is whatever the detector boxed as a single
region, which for a printed line containing a label and a value (e.g. a
form's "Nombre" and the name next to it) is usually *two* runs, so they show
up as two separate lines even though they sit on the same row of the
document.

`--format reading` is meant for reading a scanned document, not inspecting
detections: the first line is the image's average confidence (e.g.
`confidence=96.83%`), and every line after that is plain text with no boxes,
scores, or per-line labels. Text runs are re-flowed into rows by clustering
boxes whose vertical spans have an intersection-over-union above
`--reading-row-overlap` (default `0.5`), ordering rows top-to-bottom, and
ordering runs left-to-right within a row — so a label and its value on the
same printed line come back on the same output line, two spaces apart.

This clustering is a heuristic, not a guarantee: dense small text (e.g. a
digital-signature stamp block) has less pixel margin between physical
lines relative to the detector's own box-height jitter, so it's more prone
to lines being merged or split incorrectly than normal body text. If you
see that, tune `--reading-row-overlap` — raise it (e.g. `0.6`-`0.7`) if
unrelated lines are being merged, lower it (e.g. `0.3`-`0.4`) if runs that
belong on the same line are staying split — and re-run.

## Cell structure detection

`--detect-cells` finds the field boxes and table cells of a document from
its ruling lines, using classic morphology only — no model, no extra
dependency, no extra download. Opening the inverted-binary page with a long
horizontal kernel leaves only the horizontal rules, likewise for vertical;
the union is the table's skeleton, and the holes it encloses are the cells.

The two kernels are deliberately asymmetric. The horizontal one is long
(`width/28`) and does the real filtering: only a ruling line survives it, no
text does — which is why a plain-text page yields zero cells. The vertical
one is short (`height/80`) because the field boxes of typical forms are only
~20 px tall; with a long vertical kernel their sides don't survive the
opening, the cell never closes, and most short fields are lost.

With `--detect-cells`, JSON output gains a `cells` key alongside `lines`:

```json
{
  "file": "form.png",
  "detect_time_ms": 123.4,
  "lines": [ ... ],
  "cells": [
    { "box": [[255,520],[600,520],[600,550],[255,550]] }
  ]
}
```

`--detect-cells` also changes how `--format reading` assembles the page: text
runs whose centre falls inside the same cell are collapsed into a single
element (joined by spaces, top-to-bottom then left-to-right, positioned at
the cell), so a two-line address field stops dragging its whole row out of
alignment, and a label and its value stop landing on different lines. Cells
are structural units, so they are separated from their neighbours with ` | `;
between two loose text runs the separator depends on the horizontal gap,
controlled by `--reading-column-gap` (in line heights, default `0.8`).

With cell grouping active, a lower `--reading-row-overlap` works better —
around `0.25` in the prototype. The global default stays at `0.5` so plain
`--format reading` is unchanged.

## Checkbox detection

`--detect-checkboxes` needs a YOLO checkbox model (`checkbox.onnx` in
`--models-dir` by default; download it with
`models/download_models.ps1 -Checkbox`). The work is split between two very
different mechanisms, because each is good at exactly what the other is bad
at:

- **the network says WHAT is a checkbox and WHERE it is.** It is semantic:
  it does not mistake a "D" for a box. Classic contour vision alone produced
  21 false positives on a page of pure text, where letters like `D` and `B`
  are closed, near-rectangular contours the same size as a checkbox.
- **ink density says WHETHER it is ticked.** It is a physical measurement of
  the box interior (border excluded), so it does not depend on the model
  having seen that box style before. The separation is wide: empty `0.000`,
  a typed X `0.08`–`0.17`, a dark fill `0.60`–`0.91`.

Three filters keep false positives down:

- **saturation** — real checkboxes are achromatic (mean HSV saturation
  0–0.07); a colour logo is 90–134. This is what removes the corporate logo
  the network detects at 0.65 confidence.
- **the document-type gate** — a conservative pass runs first. If it finds
  fewer than `--checkbox-form-min` boxes the document is not a form, the
  low-confidence rescue never runs, and the result is returned as is. This is
  what guarantees zero false positives on plain-text documents; the JSON
  `document_type` field reports the verdict (`form` or `text`).
- **box snapping** — the network's boxes are loose and tend to swallow the
  neighbouring table cell's border; that extra ink turns empty boxes into
  "checked". Snapping finds the real rectangle first, so the ink measurement
  is clean. The JSON `snapped` field says whether it succeeded.

With `--detect-checkboxes`, JSON output gains `document_type` and a
`checkboxes` key:

```json
{
  "file": "anexo.jpg",
  "detect_time_ms": 123.4,
  "document_type": "form",
  "lines": [ ... ],
  "checkboxes": [
    { "state": "checked", "box": [[64,1425],[99,1425],[99,1446],[64,1446]],
      "ink_ratio": 0.087, "confidence": 0.67,
      "source": "main", "snapped": true }
  ]
}
```

`ink_ratio` is the measured value, not just the verdict: if a box comes back
misclassified, that number says exactly where to put `--checkbox-ink-thresh`.
`source` distinguishes what the model saw at normal confidence (`main`) from
what the heuristics recovered (`rescue`).

Known limits, measured on the reference documents: recall is 100% on two
forms and 80% on a dense one (boxes that share a border inside a table have
their contours fused), with 3 false positives on that same document (a field
border, three merged dates, and a handwritten signature — all achromatic, so
the saturation filter does not catch them).

Cell detection assumes **natively digital** documents (a PDF or spreadsheet
rendered to an image: straight borders, no skew, no scanner noise). A
skewed scan breaks the geometric premises; there is no deskew step in the
pipeline today.

PDF input is out of scope for this tool (the base engine has no PDF
support); rasterize pages to images first if you need that. This may be
revisited in a future release.

## Engine selection: CPU vs OpenVINO

`--engine openvino` makes `DbNet`, `AngleNet`, and `CrnnNet` call
`SessionOptions::AppendExecutionProvider("OpenVINO", {})` on their ONNX
Runtime sessions before loading the models. If the linked ONNX Runtime
build does not have the OpenVINO execution provider compiled in, this
call throws; PaddleVino catches that, prints a clear warning to stderr,
and continues on the plain CPU execution provider — it never fails
silently, and it never crashes because the flag was passed.

**How the release `.exe` gets a real OpenVINO-enabled ONNX Runtime**: this
is exactly how Python's `onnxruntime-openvino` PyPI wheel works too — it
doesn't require OpenVINO to be separately installed on the machine either;
it just bundles the prebuilt OpenVINO runtime binaries inside the wheel,
so `pip install` drops them straight into `site-packages`. There is a native
(C++) equivalent of that same prebuilt binary, just distributed via NuGet
instead of PyPI: **Intel's official
[`Intel.ML.OnnxRuntime.OpenVino`](https://www.nuget.org/packages/Intel.ML.OnnxRuntime.OpenVino)**
package. A `.nupkg` is a plain zip, so no `.NET`/`nuget.exe`/`dotnet` tooling
is needed to use it from a C++ project — the CI workflow
(`.github/workflows/build-windows.yml`) downloads it directly from the
NuGet v3 flat-container API, unzips it, and copies the flattened headers and
`runtimes/win-x64/native/*.{dll,lib,xml}` (which includes both the OpenVINO
execution provider and the OpenVINO runtime DLLs/`plugins.xml` it depends
on, plus the plain CPU execution provider) into `third_party/onnxruntime/`,
the same layout `-DONNXRUNTIME_ROOTDIR` expects. This replaced an earlier,
incorrect assumption in this README that no native OpenVINO-enabled ONNX
Runtime build existed for Windows/C++ — it does, it was just filed under
NuGet instead of a GitHub release.

**Honesty note**: this packaging approach has not yet been confirmed by a
green CI run that actually exercises `--engine openvino` end-to-end (e.g.
checking ONNX Runtime's verbose logs for the OpenVINO EP loading). If the
NuGet package's internal folder layout differs from what's assumed above,
or a runtime dependency is missing, the CI build step may need small
adjustments — but even in that case `--engine openvino` degrades to the
CPU fallback described above rather than crashing, so it's safe to try.

## PP-OCRv6 models

Model weights are **not** committed to this repository (they are large
binary files). Fetch them with:

```powershell
powershell -ExecutionPolicy Bypass -File models/download_models.ps1 -Tier small
```

```bash
./models/download_models.sh small
```

`-Tier`/`tier` accepts `tiny`, `small` (default), or `medium` — PP-OCRv6's
three published size classes. This downloads `det.onnx`, `rec.onnx`,
`cls.onnx`, and the matching `ppocrv6_dict.txt` dictionary into `models/`.

Sources (verified against the actual RapidOCR repo at the time of
writing, not guessed):

- Detection/recognition models: hosted on ModelScope under
  `RapidAI/RapidOCR` at tag `v3.9.2`, as referenced by
  [`python/rapidocr/default_models.yaml`](https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml)
  in the [RapidAI/RapidOCR](https://github.com/RapidAI/RapidOCR) project,
  which republishes PaddleOCR's official PP-OCRv6 weights as ONNX.
- Angle/orientation classifier: PP-OCRv6 does not ship its own classifier
  in that manifest — it reuses the PP-OCRv5 PP-LCNet text-line
  orientation model (`ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx`).

**Risk / known caveat — dictionary compatibility**: PP-OCRv6's character
dictionary (`ppocrv6_dict.txt`) is **not** the same file as PP-OCRv3's
`models/ppocr_keys_v1.txt` (kept in this repo only as a reference artifact
from the base RapidOcrOnnx project — it is not used by default and is not
downloaded by the scripts above). Mixing a v3 dictionary with a v6
recognition model, or vice versa, will silently produce garbled/incorrect
recognized text rather than an error, because both are simple
line-per-character files of similar shape. Always use the dictionary that
`download_models.ps1`/`.sh` fetches for the tier you're using. Also note:
the `tiny` rec model uses a **different, smaller** dictionary
(`ppocrv6_tiny_dict.txt`) than `small`/`medium` (`ppocrv6_dict.txt`) — the
download scripts pick the correct one automatically based on `-Tier`, but
if you swap model files manually, swap the dictionary too.

The CI release workflow downloads the `small` tier by default (configurable
via the `model_tier` workflow input) and bundles it into the shipped zip,
so the release `.exe` works out of the box.

## Build

### Why vcpkg + the official ONNX Runtime release, instead of RapidOcrOnnx's static bundles

The base repo downloads prebuilt static OpenCV/ONNX Runtime bundles from
third-party GitHub release mirrors (`RapidAI/OpenCVBuilder`,
`RapidAI/OnnxruntimeBuilder`). Those exist and are actively maintained, but
this repo instead uses:

- **OpenCV via [vcpkg](https://vcpkg.io/)** (`opencv4:x64-windows`) — vcpkg
  ships preinstalled on GitHub Actions' `windows-latest` image, is
  maintained by Microsoft, and gives reproducible, versioned builds without
  depending on a third party's release cadence for prebuilt binaries.
- **ONNX Runtime via Microsoft's own official prebuilt release**
  (`onnxruntime-win-x64-<version>.zip` from
  [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)) —
  first-party, dynamically linked, includes the CPU execution provider.

This trades a slightly longer first CI run (vcpkg builds OpenCV from
source; subsequent runs hit the `actions/cache`-backed vcpkg cache) for not
depending on a third party's binary distribution channel staying available
and correctly named indefinitely.

### Local build (Windows, for reference)

1. Install [CMake](https://cmake.org/) ≥ 3.18 and Visual Studio 2022 (or
   Build Tools) with the "Desktop development with C++" workload.
2. Install [vcpkg](https://vcpkg.io/) and run
   `vcpkg install opencv4:x64-windows`.
3. Download an `onnxruntime-win-x64-<version>.zip` release from
   [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases)
   and extract it somewhere, e.g. `C:\onnxruntime`.
4. Fetch the models: `powershell -ExecutionPolicy Bypass -File models/download_models.ps1`.
5. Configure and build:
   ```powershell
   cmake -S . -B build -A x64 `
     -DCMAKE_BUILD_TYPE=Release `
     -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake `
     -DVCPKG_TARGET_TRIPLET=x64-windows `
     -DONNXRUNTIME_ROOTDIR=C:/onnxruntime
   cmake --build build --config Release
   ```
6. `build/Release/paddlevino.exe` is the built CLI; it needs
   `onnxruntime.dll` (copied there automatically by the build) and the
   OpenCV DLLs from vcpkg's `installed/x64-windows/bin` next to it, plus a
   `models/` directory with the PP-OCRv6 files.

### CI (GitHub Actions)

See [`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml).
It runs on push to `main`, on `v*` tags, and via manual dispatch; builds on
`windows-latest`; downloads PP-OCRv6 models; and uploads
`paddlevino-windows-x64.zip` as a build artifact on every run. When
triggered by a `v*` tag, it also attaches that zip to a GitHub Release.

## Status of verification

This CLI has been developed and code-reviewed in an environment without a
Windows machine available to compile or run it. The GitHub Actions
workflow is the actual, only verification path for the Windows build — if
you're reading this shortly after it was written, check the Actions tab
for the latest run status before assuming it's green.
