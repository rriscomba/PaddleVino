#ifndef __PADDLEVINO_MAIN_H__
#define __PADDLEVINO_MAIN_H__

// CLI usage/help text for the paddlevino executable.
// Long-flag, JSON-friendly interface. See README.md for full documentation.

static const char *usageMsg =
    "paddlevino --input <file|dir> [options]\n";

static const char *examplesMsg =
    "Examples:\n"
    "  paddlevino.exe --input imagen.png --engine openvino --output resultado.json\n"
    "  paddlevino.exe --input carpeta\\ --recursive --format json\n"
    "  paddlevino --input photo.jpg --format txt\n";

static const char *requiredMsg =
    "  --input <path>        Image file or a directory of images (required)\n";

static const char *optionalMsg =
    "  --recursive            When --input is a directory, search it recursively\n"
    "  --output <path>        Write results to this file instead of stdout\n"
    "  --format json|txt|reading  Output format (default: json)\n"
    "                         json:    full detail (text, confidence, boxes)\n"
    "                         txt:     one detected text run per line, with confidence\n"
    "                         reading: first line is the average confidence, then\n"
    "                                  clean text re-flowed into reading order (text\n"
    "                                  runs on the same physical line are merged),\n"
    "                                  no boxes or per-line labels\n"
    "  --engine cpu|openvino  Execution backend (default: cpu)\n"
    "  --models-dir <dir>     Directory containing the model files (default: models)\n"
    "  --det <file>           Detection model file name (default: det.onnx)\n"
    "  --cls <file>           Angle classification model file name (default: cls.onnx)\n"
    "  --rec <file>           Recognition model file name (default: rec.onnx)\n"
    "  --keys <file>          Character dictionary file name (default: ppocrv6_dict.txt)\n"
    "  --threads <int>        ONNX Runtime thread count (default: 4)\n"
    "  --padding <int>        Border padding added to input images (default: 50)\n"
    "  --max-side-len <int>   Resize long side to this value, 0 = no resize (default: 1024)\n"
    "  --box-score-thresh <f> Detection box score threshold (default: 0.5)\n"
    "  --box-thresh <f>       Detection binarization threshold (default: 0.3)\n"
    "  --unclip-ratio <f>     Detection box expansion ratio (default: 1.6)\n"
    "  --no-angle             Disable the angle classification model\n"
    "  --no-most-angle        Disable \"most probable angle\" voting\n"
    "  --reading-row-overlap <f>  --format reading row-clustering threshold:\n"
    "                         minimum vertical-span IoU for two text runs to be\n"
    "                         merged onto the same output line. Raise it (e.g.\n"
    "                         0.6-0.7) if unrelated lines are getting merged;\n"
    "                         lower it (e.g. 0.3-0.4) if runs on the same line\n"
    "                         (common with small/dense text) are staying split\n"
    "                         (default: 0.5)\n"
    "\n"
    "  Cell structure detection (all off by default):\n"
    "  --detect-cells         Detect field/table cells by morphology and report\n"
    "                         them under the JSON \"cells\" key\n"
    "  --cell-h-frac <int>    Horizontal kernel divisor (width/N). This is the\n"
    "                         filter that keeps false positives at zero on plain\n"
    "                         text; lowering it relaxes that (default: 28)\n"
    "  --cell-v-frac <int>    Vertical kernel divisor (height/N). The most\n"
    "                         sensitive one: raise it if short field boxes are\n"
    "                         being missed (default: 80)\n"
    "  --cell-min-width <f>   Minimum cell width as a fraction of the page width\n"
    "                         (default: 0.012)\n"
    "  --cell-min-height <f>  Minimum cell height as a fraction of the page height\n"
    "                         (default: 0.006)\n"
    "  --cell-max-area <f>    Maximum cell area as a fraction of the page; drops\n"
    "                         section frames (default: 0.6)\n"
    "  --cell-rectangularity <f>  Minimum contourArea/boundingBoxArea (default: 0.7)\n"
    "\n"
    "  Diagnostics:\n"
    "  --debug-overlay <file> Write a copy of the page with the detected boxes\n"
    "                         drawn on it (cells in blue). With several input\n"
    "                         images the page index is appended to the stem\n";

static const char *otherMsg =
    "  --version, -v           Print version and exit\n"
    "  --help, -h              Print this help and exit\n";

#endif //__PADDLEVINO_MAIN_H__
