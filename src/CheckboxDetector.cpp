#include "CheckboxDetector.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace {

struct Item {
    float x1, y1, x2, y2;
    float confidence;
    int classId;
    bool fromRescue;
};

Item toItem(const CheckboxDetection &d, bool fromRescue) {
    return Item{d.x1, d.y1, d.x2, d.y2, d.score, d.classId, fromRescue};
}

CheckboxCandidate toCandidate(const Item &it, const char *reason) {
    return CheckboxCandidate{it.x1, it.y1, it.x2, it.y2, it.confidence, it.classId, reason};
}

// Mean HSV saturation of a crop. Real checkboxes are grey/white/black
// (saturation ~0); a false positive on a colour logo sits at ~90-140 --
// a very clean separation, and the only thing that removes the corporate
// logo YOLO detects with 0.65 confidence.
float cropSaturation(const cv::Mat &hsv, const Item &it) {
    int x1 = (std::max)(0, (int) it.x1);
    int y1 = (std::max)(0, (int) it.y1);
    int x2 = (std::max)((std::max)(0, (int) it.x2), x1 + 1);
    int y2 = (std::max)((std::max)(0, (int) it.y2), y1 + 1);
    x2 = (std::min)(x2, hsv.cols);
    y2 = (std::min)(y2, hsv.rows);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    cv::Mat crop = hsv(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    std::vector<cv::Mat> ch;
    cv::split(crop, ch);
    return (float) cv::mean(ch[1])[0];
}

// Greedy NMS. Crossing classes on purpose: checked/unchecked are mutually
// exclusive STATES of the same object, so two overlapping boxes of different
// classes are the same checkbox detected twice. Per-class NMS would keep
// both, because neither is ever compared against the other.
std::vector<size_t> nms(const std::vector<Item> &items, float iouThresh) {
    std::vector<size_t> idxs(items.size());
    for (size_t i = 0; i < idxs.size(); ++i) idxs[i] = i;
    std::stable_sort(idxs.begin(), idxs.end(), [&](size_t a, size_t b) {
        return items[a].confidence > items[b].confidence;
    });

    std::vector<size_t> keep;
    while (!idxs.empty()) {
        const size_t i = idxs.front();
        keep.push_back(i);
        if (idxs.size() == 1) break;
        const double areaI = (items[i].x2 - items[i].x1) * (double) (items[i].y2 - items[i].y1);
        std::vector<size_t> rest;
        for (size_t k = 1; k < idxs.size(); ++k) {
            const size_t j = idxs[k];
            const double ix = (std::min)(items[i].x2, items[j].x2) - (std::max)(items[i].x1, items[j].x1);
            const double iy = (std::min)(items[i].y2, items[j].y2) - (std::max)(items[i].y1, items[j].y1);
            const double inter = (std::max)(0.0, ix) * (std::max)(0.0, iy);
            const double areaJ = (items[j].x2 - items[j].x1) * (double) (items[j].y2 - items[j].y1);
            const double iou = inter / (areaI + areaJ - inter + 1e-9);
            if (iou < iouThresh) rest.push_back(j);
        }
        idxs.swap(rest);
    }
    return keep;
}

// Final merge: the same object detected twice (typical for very low
// confidence rescue candidates, where localisation is imprecise) ends up with
// Y centres very close relative to its own height, even when the IoU is not
// enough for plain NMS to join them. Keep the more confident one.
std::vector<size_t> dedupByCenter(const std::vector<Item> &items, const std::vector<size_t> &in, float yDistFrac) {
    std::vector<size_t> order(in);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return items[a].confidence > items[b].confidence;
    });
    std::vector<size_t> kept;
    for (size_t i: order) {
        const double icx = (items[i].x1 + items[i].x2) / 2.0;
        const double icy = (items[i].y1 + items[i].y2) / 2.0;
        const double ih = items[i].y2 - items[i].y1;
        bool dup = false;
        for (size_t k: kept) {
            const double kcx = (items[k].x1 + items[k].x2) / 2.0;
            const double kcy = (items[k].y1 + items[k].y2) / 2.0;
            const double kh = items[k].y2 - items[k].y1;
            const double ref = (std::max)(ih, kh);
            if (std::fabs(icx - kcx) < ref && std::fabs(icy - kcy) < yDistFrac * ref) {
                dup = true;
                break;
            }
        }
        if (!dup) kept.push_back(i);
    }
    return kept;
}

// Groups candidates by X proximity -- a checklist column has several rows
// sharing (roughly) the same X.
std::vector<std::vector<size_t>> clusterColumns(const std::vector<Item> &items, const std::vector<size_t> &in,
                                                float xTol) {
    std::vector<size_t> order(in);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (items[a].x1 + items[a].x2) < (items[b].x1 + items[b].x2);
    });
    std::vector<std::vector<size_t>> clusters;
    for (size_t i: order) {
        const double cx = (items[i].x1 + items[i].x2) / 2.0;
        bool placed = false;
        for (auto &cl: clusters) {
            const size_t last = cl.back();
            const double lastCx = (items[last].x1 + items[last].x2) / 2.0;
            if (std::fabs(lastCx - cx) < xTol) {
                cl.push_back(i);
                placed = true;
                break;
            }
        }
        if (!placed) clusters.push_back({i});
    }
    return clusters;
}

// Inside a column, a real checklist has roughly evenly spaced rows. A stray
// rescue candidate (e.g. the corner of a text box that happens to fall in the
// same X column) sits much further from its neighbours than the rest: keep the
// longest contiguous run whose row spacing is reasonably uniform.
std::vector<size_t> pruneUnevenRows(const std::vector<Item> &items, const std::vector<size_t> &cluster,
                                    float spacingTol) {
    if (cluster.size() < 3) return cluster;
    std::vector<size_t> rows(cluster);
    std::stable_sort(rows.begin(), rows.end(), [&](size_t a, size_t b) {
        return (items[a].y1 + items[a].y2) < (items[b].y1 + items[b].y2);
    });
    std::vector<double> centers;
    centers.reserve(rows.size());
    for (size_t i: rows) centers.push_back((items[i].y1 + items[i].y2) / 2.0);

    std::vector<double> gaps;
    for (size_t i = 0; i + 1 < centers.size(); ++i) gaps.push_back(centers[i + 1] - centers[i]);
    std::vector<double> sortedGaps(gaps);
    std::sort(sortedGaps.begin(), sortedGaps.end());
    const double medianGap = sortedGaps[sortedGaps.size() / 2];
    if (medianGap <= 0) return cluster;

    std::vector<size_t> best{rows[0]};
    std::vector<size_t> current{rows[0]};
    for (size_t i = 0; i < gaps.size(); ++i) {
        if (gaps[i] <= medianGap * spacingTol) {
            current.push_back(rows[i + 1]);
        } else {
            current.assign(1, rows[i + 1]);
        }
        if (current.size() > best.size()) best = current;
    }
    return best;
}

// Snaps the network's box to the checkbox's REAL rectangle. Needed because
// YOLO returns loose boxes that usually swallow the neighbouring table cell's
// border; that extra ink counts as a mark and turns empty boxes into
// "checked". Returns false when no valid rectangle was found and the original
// box is kept.
bool snapToBox(const cv::Mat &gray, const CheckboxParams &p, float bx1, float by1, float bx2, float by2,
               cv::Rect &out) {
    const int x1 = (int) std::lround(bx1), y1 = (int) std::lround(by1);
    const int x2 = (int) std::lround(bx2), y2 = (int) std::lround(by2);
    out = cv::Rect(x1, y1, x2 - x1, y2 - y1);

    const int rx1 = (std::max)(0, x1 - p.snapMargin);
    const int ry1 = (std::max)(0, y1 - p.snapMargin);
    const int rx2 = (std::min)(gray.cols, x2 + p.snapMargin);
    const int ry2 = (std::min)(gray.rows, y2 + p.snapMargin);
    if (rx2 <= rx1 || ry2 <= ry1) return false;

    cv::Mat roi = gray(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1));
    cv::Mat binary;
    cv::threshold(roi, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    const double roiArea = (double) roi.rows * roi.cols;
    bool found = false;
    cv::Rect best;
    for (const auto &cnt: contours) {
        cv::Rect r = cv::boundingRect(cnt);
        const double area = (double) r.width * r.height;
        if (area < p.snapMinArea * roiArea || area > p.snapMaxArea * roiArea) continue;
        if (r.height == 0) continue;
        // Deliberately wide aspect range: not every form uses square boxes --
        // the ones in the annexe are wide rectangles (aspect ~2.4), and a 2.0
        // limit left them unsnapped, so the box stayed huge and swallowed the
        // neighbouring letter.
        const double aspect = (double) r.width / r.height;
        if (aspect < p.snapMinAspect || aspect > p.snapMaxAspect) continue;
        if (cv::contourArea(cnt) / area < p.snapRectangularity) continue;
        // The SMALLEST valid rectangle: if the ROI spans the table cell and
        // the checkbox inside it, the one we want is the inner one. The X of a
        // ticked box does not compete here because it is not rectangular.
        if (!found || area < (double) best.width * best.height) {
            best = r;
            found = true;
        }
    }
    if (!found) return false;
    out = cv::Rect(rx1 + best.x, ry1 + best.y, best.width, best.height);
    return true;
}

// Ink density inside the box, excluding the border.
float inkRatio(const cv::Mat &gray, const cv::Rect &box, const CheckboxParams &p) {
    const int inset = (std::max)(2, (int) (p.inkBorder * (std::min)(box.width, box.height)));
    int x1 = box.x + inset, y1 = box.y + inset;
    int x2 = box.x + box.width - inset, y2 = box.y + box.height - inset;
    x1 = (std::max)(0, x1);
    y1 = (std::max)(0, y1);
    x2 = (std::min)(gray.cols, x2);
    y2 = (std::min)(gray.rows, y2);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    cv::Mat interior = gray(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    // An absolute cut, not Otsu: on a uniform patch (an empty box) Otsu
    // invents a threshold and reports noise as ink.
    cv::Mat dark = interior < p.inkDark;
    return (float) (cv::countNonZero(dark) / (double) (dark.rows * dark.cols));
}

}// namespace

CheckboxResult detectCheckboxes(CheckboxNet &net, const cv::Mat &bgr, const CheckboxParams &params,
                                bool collectCandidates) {
    CheckboxResult result;
    if (bgr.empty()) return result;

    cv::Mat gray, hsv;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // One forward pass at the lowest threshold that could still be needed:
    // the prototype ran the network twice with two thresholds, but the raw
    // output is identical, so the split is done here instead.
    const float lowest = params.rescueEnabled ? (std::min)(params.conf, params.rescueConf) : params.conf;
    const std::vector<CheckboxDetection> raw = net.predict(bgr, params.inputSize, lowest);

    std::vector<Item> items;
    std::vector<size_t> mainIdx;
    std::vector<size_t> rescueIdx;
    for (const CheckboxDetection &d: raw) {
        const bool isRescue = d.score < params.conf;
        Item it = toItem(d, isRescue);
        // High-confidence boxes go through the saturation filter too: the
        // colour logo sometimes scores above the main threshold.
        if (cropSaturation(hsv, it) > params.maxSaturation) {
            if (collectCandidates) result.discarded.push_back(toCandidate(it, "saturation"));
            continue;
        }
        items.push_back(it);
        (isRescue ? rescueIdx : mainIdx).push_back(items.size() - 1);
    }

    // ---- conservative pass: this is what decides the document type
    std::vector<Item> mainItems;
    for (size_t i: mainIdx) mainItems.push_back(items[i]);
    std::vector<size_t> pass1 = dedupByCenter(mainItems, nms(mainItems, params.iou), params.dedupYFrac);
    result.isForm = (int) pass1.size() >= params.formMin;

    std::vector<Item> finalItems;
    if (params.rescueEnabled && result.isForm) {
        std::vector<Item> rescueItems;
        for (size_t i: rescueIdx) rescueItems.push_back(items[i]);

        // NMS BEFORE grouping by column: at such low confidence the same
        // object produces many near-identical phantom detections, and if they
        // are not collapsed first they ruin the "evenly spaced rows" maths.
        std::vector<size_t> survived = nms(rescueItems, params.iou);
        if (collectCandidates) {
            std::vector<bool> kept(rescueItems.size(), false);
            for (size_t i: survived) kept[i] = true;
            for (size_t i = 0; i < rescueItems.size(); ++i) {
                if (!kept[i]) result.discarded.push_back(toCandidate(rescueItems[i], "nms"));
            }
        }

        std::vector<bool> accepted(rescueItems.size(), false);
        std::vector<Item> rescued;
        for (const auto &cluster: clusterColumns(rescueItems, survived, params.rescueXTol)) {
            std::vector<size_t> pruned = pruneUnevenRows(rescueItems, cluster, params.rescueSpacingTol);
            if ((int) pruned.size() >= params.rescueMinCluster) {
                for (size_t i: pruned) {
                    accepted[i] = true;
                    rescued.push_back(rescueItems[i]);
                }
            }
        }
        if (collectCandidates) {
            for (size_t i: survived) {
                if (!accepted[i]) result.discarded.push_back(toCandidate(rescueItems[i], "rescue-pruned"));
            }
        }

        std::vector<Item> all(mainItems);
        all.insert(all.end(), rescued.begin(), rescued.end());
        std::vector<size_t> keep = dedupByCenter(all, nms(all, params.iou), params.dedupYFrac);
        std::vector<bool> kept(all.size(), false);
        for (size_t i: keep) kept[i] = true;
        for (size_t i = 0; i < all.size(); ++i) {
            if (kept[i]) finalItems.push_back(all[i]);
            else if (collectCandidates) result.discarded.push_back(toCandidate(all[i], "nms"));
        }
    } else {
        std::vector<bool> kept(mainItems.size(), false);
        for (size_t i: pass1) kept[i] = true;
        for (size_t i = 0; i < mainItems.size(); ++i) {
            if (kept[i]) finalItems.push_back(mainItems[i]);
            else if (collectCandidates) result.discarded.push_back(toCandidate(mainItems[i], "nms"));
        }
        // The rescue path never ran: either it is disabled, or the gate
        // decided this document is not a form.
        if (collectCandidates) {
            for (size_t i: rescueIdx) result.discarded.push_back(toCandidate(items[i], "below-conf"));
        }
    }

    // ---- snap, then measure ink. The state comes from the measurement, not
    // from the network's classifier, which is why it is right 100% of the time
    // on the prototype's three forms.
    for (const Item &it: finalItems) {
        cv::Rect box;
        bool snapped = false;
        if (params.snapEnabled) {
            snapped = snapToBox(gray, params, it.x1, it.y1, it.x2, it.y2, box);
        } else {
            box = cv::Rect((int) std::lround(it.x1), (int) std::lround(it.y1),
                           (int) std::lround(it.x2) - (int) std::lround(it.x1),
                           (int) std::lround(it.y2) - (int) std::lround(it.y1));
        }
        const float ratio = inkRatio(gray, box, params);
        Checkbox cb{};
        cb.x1 = box.x;
        cb.y1 = box.y;
        cb.x2 = box.x + box.width;
        cb.y2 = box.y + box.height;
        cb.checked = ratio > params.inkThresh;
        cb.inkRatio = ratio;
        cb.confidence = it.confidence;
        cb.fromRescue = it.fromRescue;
        cb.snapped = snapped;
        result.checkboxes.push_back(cb);
    }

    std::sort(result.checkboxes.begin(), result.checkboxes.end(), [](const Checkbox &a, const Checkbox &b) {
        if (a.y1 != b.y1) return a.y1 < b.y1;
        return a.x1 < b.x1;
    });
    return result;
}
