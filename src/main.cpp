#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <utility>
#include "main.h"
#include "version.h"
#include "OcrLite.h"
#include "OcrUtils.h"
#include "EngineType.h"
#include "CellDetector.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string input;
    bool recursive = false;
    std::string output;
    std::string format = "json";
    std::string engine = "cpu";
    std::string modelsDir = "models";
    std::string detName = "det.onnx";
    std::string clsName = "cls.onnx";
    std::string recName = "rec.onnx";
    std::string keysName = "ppocrv6_dict.txt";
    int numThread = 4;
    int padding = 50;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.5f;
    float boxThresh = 0.3f;
    float unClipRatio = 1.6f;
    bool doAngle = true;
    bool mostAngle = true;
    float readingRowOverlap = 0.5f;
    // Only used once --detect-cells / --detect-checkboxes restructure the
    // reading output; plain --format reading keeps its two-space join.
    float readingColumnGap = 0.8f;

    // --- cell structure detection (off by default) ---
    bool detectCellsEnabled = false;
    CellDetectorParams cellParams;

    // --- diagnostics (off by default) ---
    std::string debugOverlay;
};

// Per-page results produced by the optional detectors. Empty/disabled by
// default, so the JSON stays byte-for-byte identical when no new flag is used.
struct PageExtras {
    bool hasCells = false;
    std::vector<Cell> cells;
};

void printHelp(FILE *out, const char *argv0) {
    fprintf(out, " ------- Usage -------\n");
    fprintf(out, "%s %s\n", argv0, usageMsg);
    fprintf(out, " ------- Required -------\n%s\n", requiredMsg);
    fprintf(out, " ------- Options -------\n%s\n", optionalMsg);
    fprintf(out, " ------- Other -------\n%s\n", otherMsg);
    fprintf(out, " ------- Examples -------\n%s\n", examplesMsg);
}

bool hasImageExtension(const fs::path &p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
           ext == ".tif" || ext == ".tiff" || ext == ".webp";
}

std::vector<std::string> collectImages(const std::string &input, bool recursive) {
    std::vector<std::string> images;
    fs::path root(input);
    if (fs::is_regular_file(root)) {
        images.push_back(root.string());
        return images;
    }
    if (!fs::is_directory(root)) {
        return images;
    }
    if (recursive) {
        for (auto &entry: fs::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && hasImageExtension(entry.path())) {
                images.push_back(entry.path().string());
            }
        }
    } else {
        for (auto &entry: fs::directory_iterator(root)) {
            if (entry.is_regular_file() && hasImageExtension(entry.path())) {
                images.push_back(entry.path().string());
            }
        }
    }
    std::sort(images.begin(), images.end());
    return images;
}

std::string jsonEscape(const std::string &s) {
    std::ostringstream out;
    for (char c: s) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

float averageConfidence(const std::vector<float> &charScores) {
    if (charScores.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s: charScores) sum += s;
    return sum / static_cast<float>(charScores.size());
}

// Serializes an axis-aligned rectangle as the same 4-point polygon shape the
// "lines" boxes already use: top-left, top-right, bottom-right, bottom-left.
void writeRectPoints(std::ostringstream &out, double x1, double y1, double x2, double y2) {
    out << "[[" << x1 << "," << y1 << "],[" << x2 << "," << y1 << "],["
        << x2 << "," << y2 << "],[" << x1 << "," << y2 << "]]";
}

std::string resultToJson(const std::string &file, const OcrResult &result, bool ok, const std::string &error,
                         const PageExtras &extras) {
    std::ostringstream out;
    out << "{\"file\":\"" << jsonEscape(file) << "\",";
    if (!ok) {
        out << "\"error\":\"" << jsonEscape(error) << "\",\"lines\":[]}";
        return out.str();
    }
    out << "\"detect_time_ms\":" << result.detectTime << ",\"lines\":[";
    for (size_t i = 0; i < result.textBlocks.size(); ++i) {
        const TextBlock &b = result.textBlocks[i];
        if (i > 0) out << ",";
        out << "{\"text\":\"" << jsonEscape(b.text) << "\",";
        out << "\"confidence\":" << averageConfidence(b.charScores) << ",";
        out << "\"box_score\":" << b.boxScore << ",";
        out << "\"box\":[";
        for (size_t p = 0; p < b.boxPoint.size(); ++p) {
            if (p > 0) out << ",";
            out << "[" << b.boxPoint[p].x << "," << b.boxPoint[p].y << "]";
        }
        out << "]}";
    }
    out << "]";
    if (extras.hasCells) {
        out << ",\"cells\":[";
        for (size_t i = 0; i < extras.cells.size(); ++i) {
            const Cell &c = extras.cells[i];
            if (i > 0) out << ",";
            out << "{\"box\":";
            writeRectPoints(out, c.x, c.y, c.x + c.width, c.y + c.height);
            out << "}";
        }
        out << "]";
    }
    out << "}";
    return out.str();
}

std::string resultToText(const std::string &file, const OcrResult &result, bool ok, const std::string &error) {
    std::ostringstream out;
    out << "==== " << file << " ====\n";
    if (!ok) {
        out << "ERROR: " << error << "\n";
        return out.str();
    }
    for (const TextBlock &b: result.textBlocks) {
        out << b.text << "\t(confidence=" << averageConfidence(b.charScores) << ")\n";
    }
    return out.str();
}

// --debug-overlay names a single file, but --input can be a whole directory.
// With more than one image the index is appended to the stem so the pages
// don't overwrite each other.
std::string overlayPath(const std::string &base, size_t index, size_t total) {
    if (total <= 1) return base;
    fs::path p(base);
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    if (ext.empty()) ext = ".png";
    fs::path parent = p.parent_path();
    std::string name = stem + "_" + std::to_string(index) + ext;
    return parent.empty() ? name : (parent / name).string();
}

// Draws what the optional detectors found on top of the page: cells in blue.
// Tuning ~25 numeric knobs blind is not viable; this is the tool that makes
// them adjustable.
void writeDebugOverlay(const std::string &path, const cv::Mat &pageBgr, const PageExtras &extras) {
    cv::Mat vis = pageBgr.clone();
    for (const Cell &c: extras.cells) {
        cv::rectangle(vis, cv::Rect(c.x, c.y, c.width, c.height), cv::Scalar(255, 0, 0), 2);
    }
    if (!cv::imwrite(path, vis)) {
        fprintf(stderr, "Could not write debug overlay: %s\n", path.c_str());
    }
}

// Groups detected text blocks into reading-order rows: blocks whose boxes'
// vertical spans have an intersection-over-union above rowOverlapThresh are
// considered part of the same physical line (the detector emits one box
// per text run, so two runs on the same printed line still arrive as
// separate blocks and need to be re-joined here), then rows are ordered
// top-to-bottom and blocks within a row left-to-right.
std::vector<std::vector<size_t>> groupIntoRows(const std::vector<TextBlock> &blocks, float rowOverlapThresh) {
    struct VSpan {
        int y0, y1;
    };
    std::vector<VSpan> span(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        int y0 = blocks[i].boxPoint.front().y, y1 = blocks[i].boxPoint.front().y;
        for (const auto &pt: blocks[i].boxPoint) {
            y0 = (std::min)(y0, pt.y);
            y1 = (std::max)(y1, pt.y);
        }
        span[i] = {y0, y1};
    }

    std::vector<size_t> order(blocks.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return span[a].y0 < span[b].y0;
    });

    std::vector<std::vector<size_t>> rows;
    std::vector<VSpan> rowSpan;
    for (size_t idx: order) {
        int bestRow = -1;
        // Intersection-over-union of the vertical spans, not intersection
        // over the shorter span: for small, tightly-packed text (e.g. a
        // digital-signature stamp) the box-height jitter between adjacent
        // physical lines is a large fraction of each line's own height, so
        // "inter/shorter" was merging distinct lines together. IoU is
        // stricter for lines that only touch/overlap slightly while still
        // accepting near-identical spans (genuine same-line runs).
        float bestOverlap = rowOverlapThresh;
        for (size_t r = 0; r < rows.size(); ++r) {
            int interLo = (std::max)(span[idx].y0, rowSpan[r].y0);
            int interHi = (std::min)(span[idx].y1, rowSpan[r].y1);
            int inter = (std::max)(0, interHi - interLo);
            int unionLo = (std::min)(span[idx].y0, rowSpan[r].y0);
            int unionHi = (std::max)(span[idx].y1, rowSpan[r].y1);
            int uni = unionHi - unionLo;
            if (uni <= 0) continue;
            float overlap = (float) inter / (float) uni;
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                bestRow = (int) r;
            }
        }
        if (bestRow < 0) {
            rows.push_back({idx});
            rowSpan.push_back(span[idx]);
        } else {
            rows[bestRow].push_back(idx);
            rowSpan[bestRow].y0 = (std::min)(rowSpan[bestRow].y0, span[idx].y0);
            rowSpan[bestRow].y1 = (std::max)(rowSpan[bestRow].y1, span[idx].y1);
        }
    }

    // Order rows top-to-bottom, and blocks within each row left-to-right.
    std::sort(rows.begin(), rows.end(), [&](const std::vector<size_t> &a, const std::vector<size_t> &b) {
        return span[a.front()].y0 < span[b.front()].y0;
    });
    for (auto &row: rows) {
        std::sort(row.begin(), row.end(), [&](size_t a, size_t b) {
            return blocks[a].boxPoint.front().x < blocks[b].boxPoint.front().x;
        });
    }
    return rows;
}

struct Bounds {
    int x1, y1, x2, y2;
};

Bounds boundsOf(const std::vector<cv::Point> &pts) {
    Bounds b{pts.front().x, pts.front().y, pts.front().x, pts.front().y};
    for (const cv::Point &p: pts) {
        b.x1 = (std::min)(b.x1, p.x);
        b.y1 = (std::min)(b.y1, p.y);
        b.x2 = (std::max)(b.x2, p.x);
        b.y2 = (std::max)(b.y2, p.y);
    }
    return b;
}

std::vector<cv::Point> rectPoints(int x1, int y1, int x2, int y2) {
    return {cv::Point(x1, y1), cv::Point(x2, y1), cv::Point(x2, y2), cv::Point(x1, y2)};
}

// Builds the elements that go into groupIntoRows() when the optional
// detectors are active. Two transformations, both from the Python prototype
// (merge_celdas.py):
//
//  1. text runs whose centre falls inside the same cell are collapsed into a
//     single synthetic block (joined by spaces, top-to-bottom then
//     left-to-right, box = the cell), so a two-line address field stops
//     dragging its whole row out of alignment;
//  2. a cell that CONTAINS checkboxes is not a field with one value but a
//     frame grouping several options (the annexes list at the foot of a
//     form). Collapsing it wrecks that section, so its runs are left loose.
//
// The result is fed to groupIntoRows() unmodified.
std::vector<TextBlock> buildReadingBlocks(const std::vector<TextBlock> &blocks, const PageExtras &extras,
                                          std::vector<bool> &isCellOut) {
    std::vector<TextBlock> outBlocks;
    isCellOut.clear();

    // Cells that hold a checkbox must not collapse.
    std::vector<bool> isContainer(extras.cells.size(), false);

    std::vector<std::vector<size_t>> perCell(extras.cells.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        Bounds b = boundsOf(blocks[i].boxPoint);
        const double cx = (b.x1 + b.x2) / 2.0, cy = (b.y1 + b.y2) / 2.0;
        long long bestArea = 0;
        int best = -1;
        for (size_t c = 0; c < extras.cells.size(); ++c) {
            if (isContainer[c]) continue;
            const Cell &cell = extras.cells[c];
            if (cx < cell.x || cx > cell.x + cell.width || cy < cell.y || cy > cell.y + cell.height) continue;
            const long long area = (long long) cell.width * cell.height;
            if (best < 0 || area < bestArea) {
                best = (int) c;
                bestArea = area;
            }
        }
        if (best < 0) {
            outBlocks.push_back(blocks[i]);
            isCellOut.push_back(false);
        } else {
            perCell[best].push_back(i);
        }
    }

    for (size_t c = 0; c < perCell.size(); ++c) {
        if (perCell[c].empty()) continue;
        std::vector<size_t> items = perCell[c];
        std::sort(items.begin(), items.end(), [&](size_t a, size_t b) {
            Bounds ba = boundsOf(blocks[a].boxPoint), bb = boundsOf(blocks[b].boxPoint);
            const double ca = (ba.y1 + ba.y2) / 2.0, cb = (bb.y1 + bb.y2) / 2.0;
            if (ca != cb) return ca < cb;
            return ba.x1 < bb.x1;
        });
        std::string text;
        for (size_t idx: items) {
            if (blocks[idx].text.empty()) continue;
            if (!text.empty()) text += " ";
            text += blocks[idx].text;
        }
        if (text.empty()) continue;
        const Cell &cell = extras.cells[c];
        TextBlock tb{};
        tb.boxPoint = rectPoints(cell.x, cell.y, cell.x + cell.width, cell.y + cell.height);
        tb.text = text;
        outBlocks.push_back(tb);
        isCellOut.push_back(true);
    }

    return outBlocks;
}

// Renders one row. A cell is a structural unit (label vs. field) and is always
// separated with " | "; between two loose text runs the horizontal gap decides,
// measured in line heights, because a wide gap in running text can just be
// paragraph spacing.
std::string renderReadingRow(const std::vector<TextBlock> &blocks, const std::vector<bool> &isCell,
                             const std::vector<size_t> &row, float columnGapRatio) {
    std::vector<double> heights;
    for (size_t idx: row) {
        Bounds b = boundsOf(blocks[idx].boxPoint);
        heights.push_back((std::max)(1.0, (double) (b.y2 - b.y1)));
    }
    std::sort(heights.begin(), heights.end());
    const double refH = heights[heights.size() / 2];

    std::string outText = blocks[row.front()].text;
    for (size_t i = 1; i < row.size(); ++i) {
        const size_t prev = row[i - 1], cur = row[i];
        std::string sep;
        if (isCell[prev] || isCell[cur]) {
            sep = " | ";
        } else {
            const double gap = (double) boundsOf(blocks[cur].boxPoint).x1 - boundsOf(blocks[prev].boxPoint).x2;
            sep = gap > columnGapRatio * refH ? " | " : " ";
        }
        outText += sep + blocks[cur].text;
    }
    return outText;
}

std::string resultToReading(const std::string &file, const OcrResult &result, bool ok, const std::string &error,
                            float rowOverlapThresh, const PageExtras &extras, float columnGapRatio) {
    std::ostringstream out;
    if (!ok) {
        out << "confidence=0.00% (error)\n" << "ERROR (" << file << "): " << error << "\n";
        return out.str();
    }
    float sum = 0.0f;
    for (const TextBlock &b: result.textBlocks) sum += averageConfidence(b.charScores);
    float avgConfidence = result.textBlocks.empty() ? 0.0f : sum / (float) result.textBlocks.size();
    out << "confidence=" << (avgConfidence * 100.0f) << "%\n";

    // Without any of the optional detectors this is byte-for-byte the
    // original behaviour: no restructuring, runs joined by two spaces.
    if (!extras.hasCells) {
        for (const std::vector<size_t> &row: groupIntoRows(result.textBlocks, rowOverlapThresh)) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) out << "  ";
                out << result.textBlocks[row[i]].text;
            }
            out << "\n";
        }
        return out.str();
    }

    std::vector<bool> isCell;
    std::vector<TextBlock> blocks = buildReadingBlocks(result.textBlocks, extras, isCell);
    for (const std::vector<size_t> &row: groupIntoRows(blocks, rowOverlapThresh)) {
        out << renderReadingRow(blocks, isCell, row, columnGapRatio) << "\n";
    }
    return out.str();
}

} // namespace

int main(int argc, char **argv) {
    if (argc <= 1) {
        printHelp(stderr, argv[0]);
        return 1;
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char *flagName) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", flagName);
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--input") opt.input = next("--input");
        else if (arg == "--recursive") opt.recursive = true;
        else if (arg == "--output") opt.output = next("--output");
        else if (arg == "--format") opt.format = next("--format");
        else if (arg == "--engine") opt.engine = next("--engine");
        else if (arg == "--models-dir") opt.modelsDir = next("--models-dir");
        else if (arg == "--det") opt.detName = next("--det");
        else if (arg == "--cls") opt.clsName = next("--cls");
        else if (arg == "--rec") opt.recName = next("--rec");
        else if (arg == "--keys") opt.keysName = next("--keys");
        else if (arg == "--threads") opt.numThread = std::stoi(next("--threads"));
        else if (arg == "--padding") opt.padding = std::stoi(next("--padding"));
        else if (arg == "--max-side-len") opt.maxSideLen = std::stoi(next("--max-side-len"));
        else if (arg == "--box-score-thresh") opt.boxScoreThresh = std::stof(next("--box-score-thresh"));
        else if (arg == "--box-thresh") opt.boxThresh = std::stof(next("--box-thresh"));
        else if (arg == "--unclip-ratio") opt.unClipRatio = std::stof(next("--unclip-ratio"));
        else if (arg == "--no-angle") opt.doAngle = false;
        else if (arg == "--no-most-angle") opt.mostAngle = false;
        else if (arg == "--reading-row-overlap") opt.readingRowOverlap = std::stof(next("--reading-row-overlap"));
        else if (arg == "--reading-column-gap") opt.readingColumnGap = std::stof(next("--reading-column-gap"));
        else if (arg == "--detect-cells") opt.detectCellsEnabled = true;
        else if (arg == "--cell-h-frac") opt.cellParams.hFrac = std::stoi(next("--cell-h-frac"));
        else if (arg == "--cell-v-frac") opt.cellParams.vFrac = std::stoi(next("--cell-v-frac"));
        else if (arg == "--cell-min-width") opt.cellParams.minWidthFrac = std::stof(next("--cell-min-width"));
        else if (arg == "--cell-min-height") opt.cellParams.minHeightFrac = std::stof(next("--cell-min-height"));
        else if (arg == "--cell-max-area") opt.cellParams.maxAreaFrac = std::stof(next("--cell-max-area"));
        else if (arg == "--cell-rectangularity") opt.cellParams.rectangularity = std::stof(next("--cell-rectangularity"));
        else if (arg == "--debug-overlay") opt.debugOverlay = next("--debug-overlay");
        else if (arg == "--version" || arg == "-v") {
            printf("%s\n", VERSION);
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            printHelp(stdout, argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            printHelp(stderr, argv[0]);
            return 1;
        }
    }

    if (opt.input.empty()) {
        fprintf(stderr, "--input is required\n\n");
        printHelp(stderr, argv[0]);
        return 1;
    }
    if (opt.format != "json" && opt.format != "txt" && opt.format != "reading") {
        fprintf(stderr, "--format must be 'json', 'txt' or 'reading'\n");
        return 1;
    }
    EngineType engine = EngineType::CPU;
    if (opt.engine == "openvino") {
        engine = EngineType::OpenVINO;
    } else if (opt.engine != "cpu") {
        fprintf(stderr, "--engine must be 'cpu' or 'openvino'\n");
        return 1;
    }

    std::string detPath = opt.modelsDir + "/" + opt.detName;
    std::string clsPath = opt.modelsDir + "/" + opt.clsName;
    std::string recPath = opt.modelsDir + "/" + opt.recName;
    std::string keysPath = opt.modelsDir + "/" + opt.keysName;

    for (auto &pair: std::vector<std::pair<std::string, std::string>>{
            {"det model",  detPath},
            {"cls model",  clsPath},
            {"rec model",  recPath},
            {"keys file",  keysPath}}) {
        if (!isFileExists(pair.second)) {
            fprintf(stderr, "%s not found: %s\n", pair.first.c_str(), pair.second.c_str());
            return 1;
        }
    }

    std::vector<std::string> images = collectImages(opt.input, opt.recursive);
    if (images.empty()) {
        fprintf(stderr, "No input images found at: %s\n", opt.input.c_str());
        return 1;
    }

    OcrLite ocrLite;
    ocrLite.setNumThread(opt.numThread);
    ocrLite.initLogger(
            false, //isOutputConsole -- keep stdout clean for JSON/txt piping
            false, //isOutputPartImg
            false);//isOutputResultImg
    ocrLite.setEngine(engine);
    ocrLite.initModels(detPath, clsPath, recPath, keysPath);

    std::ostream *out = &std::cout;
    std::ofstream outFile;
    if (!opt.output.empty()) {
        outFile.open(opt.output, std::ios::binary);
        if (!outFile.is_open()) {
            fprintf(stderr, "Could not open output file: %s\n", opt.output.c_str());
            return 1;
        }
        out = &outFile;
    }

    if (opt.format == "json") *out << "[";
    for (size_t i = 0; i < images.size(); ++i) {
        const std::string &imgPath = images[i];
        bool ok = true;
        std::string error;
        OcrResult result{};
        try {
            fs::path p(imgPath);
            std::string dir = p.parent_path().string();
            if (!dir.empty()) dir += "/";
            std::string name = p.filename().string();
            result = ocrLite.detect(dir.c_str(), name.c_str(), opt.padding, opt.maxSideLen,
                                     opt.boxScoreThresh, opt.boxThresh, opt.unClipRatio,
                                     opt.doAngle, opt.mostAngle);
        } catch (const std::exception &e) {
            ok = false;
            error = e.what();
        }

        // The optional detectors work on the source pixels, but
        // OcrLite::detect() takes paths and does not hand the decoded image
        // back, so the page is read once more here. TextBlock::boxPoint is
        // already in original-image coordinates (the engine reverts both the
        // padding and the maxSideLen resize), so no transform is needed to
        // put OCR boxes and CV boxes in the same space.
        PageExtras extras;
        const bool needsPixels = opt.detectCellsEnabled || !opt.debugOverlay.empty();
        if (ok && needsPixels) {
            cv::Mat pageBgr = cv::imread(imgPath, cv::IMREAD_COLOR);
            if (pageBgr.empty()) {
                fprintf(stderr, "Could not re-read image for cell/checkbox detection: %s\n", imgPath.c_str());
            } else {
                cv::Mat gray;
                cv::cvtColor(pageBgr, gray, cv::COLOR_BGR2GRAY);
                if (opt.detectCellsEnabled) {
                    extras.hasCells = true;
                    extras.cells = detectCells(gray, opt.cellParams);
                }
                if (!opt.debugOverlay.empty()) {
                    writeDebugOverlay(overlayPath(opt.debugOverlay, i, images.size()), pageBgr, extras);
                }
            }
        }

        if (opt.format == "json") {
            if (i > 0) *out << ",";
            *out << resultToJson(imgPath, result, ok, error, extras);
        } else if (opt.format == "reading") {
            *out << resultToReading(imgPath, result, ok, error, opt.readingRowOverlap, extras,
                                    opt.readingColumnGap);
        } else {
            *out << resultToText(imgPath, result, ok, error);
        }
    }
    if (opt.format == "json") *out << "]";
    if (opt.format == "json") *out << "\n";

    return 0;
}
