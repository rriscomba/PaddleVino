#include "CellDetector.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>

std::vector<Cell> detectCells(const cv::Mat &gray, const CellDetectorParams &params) {
    std::vector<Cell> cells;
    if (gray.empty()) return cells;

    const int w = gray.cols;
    const int h = gray.rows;

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    const int hFrac = (std::max)(1, params.hFrac);
    const int vFrac = (std::max)(1, params.vFrac);
    cv::Mat hk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size((std::max)(10, w / hFrac), 1));
    cv::Mat vk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, (std::max)(6, h / vFrac)));

    cv::Mat horiz, vert;
    cv::morphologyEx(binary, horiz, cv::MORPH_OPEN, hk);
    cv::morphologyEx(binary, vert, cv::MORPH_OPEN, vk);

    cv::Mat grid;
    cv::bitwise_or(horiz, vert, grid);
    cv::dilate(grid, grid, cv::Mat::ones(3, 3, CV_8U));

    // The cells are the holes the skeleton leaves behind: with RETR_CCOMP they
    // are the level-1 contours (those that have a parent).
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(grid, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    const double minW = (double) params.minWidthFrac * w;
    const double minH = (double) params.minHeightFrac * h;
    const double pageArea = (double) w * (double) h;

    std::vector<Cell> candidates;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (!hierarchy.empty() && hierarchy[i][3] == -1) {
            continue;// outer contour (the rule itself), not a hole
        }
        cv::Rect r = cv::boundingRect(contours[i]);
        if (r.width < minW || r.height < minH) continue;
        if (((double) r.width * r.height) / pageArea > params.maxAreaFrac) continue;
        if (cv::contourArea(contours[i]) / ((double) r.width * r.height) < params.rectangularity) continue;
        candidates.push_back(Cell{r.x, r.y, r.width, r.height});
    }

    // Drop containers: if a cell spans other cells we want the leaves (the
    // real cells), not the frame that groups them. 2 px of slack.
    for (size_t a = 0; a < candidates.size(); ++a) {
        const int ax1 = candidates[a].x, ay1 = candidates[a].y;
        const int ax2 = ax1 + candidates[a].width, ay2 = ay1 + candidates[a].height;
        bool contains = false;
        for (size_t b = 0; b < candidates.size() && !contains; ++b) {
            if (a == b) continue;
            const int bx1 = candidates[b].x, by1 = candidates[b].y;
            const int bx2 = bx1 + candidates[b].width, by2 = by1 + candidates[b].height;
            if (bx1 >= ax1 - 2 && by1 >= ay1 - 2 && bx2 <= ax2 + 2 && by2 <= ay2 + 2) {
                contains = true;
            }
        }
        if (!contains) cells.push_back(candidates[a]);
    }

    std::sort(cells.begin(), cells.end(), [](const Cell &a, const Cell &b) {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });
    return cells;
}
