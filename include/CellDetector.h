#ifndef __PADDLEVINO_CELL_DETECTOR_H__
#define __PADDLEVINO_CELL_DETECTOR_H__

#include <opencv2/core.hpp>
#include <vector>

// Detects the CELLS of a document (field boxes, table cells) from their ruling
// lines, with classic morphology -- no model, no extra dependency.
//
// Idea: opening the inverted-binary image with a long horizontal kernel leaves
// only the horizontal rules; likewise for vertical. Their union is the
// "skeleton" of the table, and the holes it encloses are the cells.
//
// The kernels are asymmetric on purpose. The horizontal one is long
// (width/28) and acts as the real filter: only a table rule survives it, no
// text does -- that is what keeps false positives at zero on plain-text pages.
// The vertical one must be SHORT (height/80 ~ 16 px) because the field boxes
// of these forms are only ~20 px tall; with a long vertical kernel their sides
// do not survive the opening, the cell never closes, and 10 of 18 fields are
// lost (measured: h/28 -> 8 cells on pagina2, h/80 -> 18).
struct CellDetectorParams {
    int hFrac = 28;                 // horizontal kernel divisor (width / N)
    int vFrac = 80;                 // vertical kernel divisor (height / N)
    float minWidthFrac = 0.012f;    // minimum cell width as a fraction of the page width
    float minHeightFrac = 0.006f;   // minimum cell height as a fraction of the page height
    float maxAreaFrac = 0.6f;       // maximum cell area as a fraction of the page area
    float rectangularity = 0.7f;    // minimum contourArea / boundingBoxArea
};

// A detected cell, in original-image pixel coordinates.
struct Cell {
    int x;
    int y;
    int width;
    int height;
};

// Returns the leaf cells found in `gray` (a single channel 8-bit image),
// sorted top-to-bottom then left-to-right.
std::vector<Cell> detectCells(const cv::Mat &gray, const CellDetectorParams &params);

#endif //__PADDLEVINO_CELL_DETECTOR_H__
