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
| `--version`, `-v` | Print version and exit | — |
| `--help`, `-h` | Print usage and exit | — |

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
boxes that overlap vertically by more than half of the shorter box's height,
ordering rows top-to-bottom, and ordering runs left-to-right within a row —
so a label and its value on the same printed line come back on the same
output line, two spaces apart.

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

**Important, please read**: the official release `.exe` built by this
repo's GitHub Actions workflow links the **stock Microsoft ONNX Runtime
Windows package**, which does **not** include the OpenVINO execution
provider. Building ONNX Runtime from source with `--use_openvino` was
evaluated and rejected for the default CI pipeline: it requires the full
OpenVINO toolkit installed on the runner and commonly takes well over an
hour, on top of the rest of the build, which risks exceeding practical CI
time and is fragile on a shared `windows-latest` runner. There is also no
official prebuilt native (C++) ONNX Runtime binary with OpenVINO EP for
Windows — Microsoft/Intel only publish OpenVINO EP through the Python
`onnxruntime-openvino` PyPI package and a C#/.NET NuGet package
(`Intel.ML.OnnxRuntime.OpenVino`), neither of which is usable from this
C++ CLI.

So today: `--engine openvino` is implemented and will genuinely use the
OpenVINO EP **if** you link PaddleVino against a build of ONNX Runtime
that has it compiled in (e.g. one you build yourself locally with
`--use_openvino`, following
[the official ONNX Runtime OpenVINO EP build docs](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)),
by pointing `-DONNXRUNTIME_ROOTDIR` at that build. The default GitHub
Actions release build does not do this, and using `--engine openvino`
against that release `.exe` will print the fallback warning and run on
CPU. This is documented here rather than silently shipping a flag that
looks like it works but doesn't.

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
