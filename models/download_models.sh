#!/usr/bin/env bash
# Downloads the official PP-OCRv6 ONNX models (published by the RapidOCR
# project, re-packaged from PaddleOCR/PaddlePaddle weights) into this
# directory, alongside the matching character dictionary.
#
# Usage:
#   ./models/download_models.sh [tiny|small|medium] [checkbox]
#
# Source of truth for these URLs:
#   https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
# Models are hosted on ModelScope under the RapidAI/RapidOCR model repo.
#
# The trailing "checkbox" argument additionally downloads the optional
# checkbox detection model used by --detect-checkboxes. It is opt-in on
# purpose: ~10.8 MB most users do not need, and it is AGPL-3.0 licensed (see
# THIRD_PARTY_LICENSES/checkbox-detector-LICENSE-AGPL-3.0.txt).
set -euo pipefail

TIER="${1:-small}"
case "$TIER" in
    tiny|small|medium) ;;
    *) echo "Usage: $0 [tiny|small|medium] [checkbox]" >&2; exit 1 ;;
esac

WANT_CHECKBOX=0
if [ "${2:-}" = "checkbox" ]; then
    WANT_CHECKBOX=1
elif [ -n "${2:-}" ]; then
    echo "Usage: $0 [tiny|small|medium] [checkbox]" >&2; exit 1
fi

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"

DET_URL="$BASE/onnx/PP-OCRv6/det/PP-OCRv6_det_${TIER}.onnx"
REC_URL="$BASE/onnx/PP-OCRv6/rec/PP-OCRv6_rec_${TIER}.onnx"
# The angle/orientation classifier is not re-released per PP-OCRv6 tier;
# RapidOCR's PP-OCRv6 pipeline reuses the PP-OCRv5 PP-LCNet text-line
# orientation classifier.
CLS_URL="$BASE/onnx/PP-OCRv5/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"

# IMPORTANT: the character dictionary is tied to the rec model's tier.
# The "tiny" rec model uses a different (smaller) dictionary than
# "small"/"medium". Always download the dictionary that matches $TIER.
if [ "$TIER" = "tiny" ]; then
    DICT_URL="$BASE/paddle/PP-OCRv6/rec/PP-OCRv6_rec_tiny/ppocrv6_tiny_dict.txt"
else
    DICT_URL="$BASE/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt"
fi

download() {
    local url="$1" out="$2"
    echo "Downloading $url -> $out"
    curl -fL --retry 3 -o "$out" "$url"
}

download "$DET_URL"  "$DEST_DIR/det.onnx"
download "$REC_URL"  "$DEST_DIR/rec.onnx"
download "$CLS_URL"  "$DEST_DIR/cls.onnx"
download "$DICT_URL" "$DEST_DIR/ppocrv6_dict.txt"

if [ "$WANT_CHECKBOX" = "1" ]; then
    CHECKBOX_URL="https://huggingface.co/wendys-llc/checkbox-detector/resolve/main/checkbox_yolo12n.onnx"
    download "$CHECKBOX_URL" "$DEST_DIR/checkbox.onnx"
    echo "Checkbox model downloaded. It is AGPL-3.0 licensed - see THIRD_PARTY_LICENSES/."
fi

echo "Done. Models placed in $DEST_DIR (tier: $TIER)."
