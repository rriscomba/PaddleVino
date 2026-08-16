#ifndef __PADDLEVINO_CHECKBOX_DETECTOR_H__
#define __PADDLEVINO_CHECKBOX_DETECTOR_H__

#include "CheckboxNet.h"

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Division of labour, measured on the Python prototype:
//   YOLO  -> WHAT is a checkbox and WHERE it is (semantic: it does not
//            mistake a "D" for a box, because it learned the concept;
//            classic vision alone gives 21 false positives on a text page)
//   ink   -> WHETHER it is ticked (a physical measurement: it does not
//            depend on having seen that box style before)
struct CheckboxParams {
    // §6.3 detection
    float conf = 0.25f;
    float iou = 0.45f;
    float maxSaturation = 20.0f;
    int inputSize = 1024;

    // §6.4 document-type gate. The conservative pass runs first; only if it
    // already found this many boxes is the document a form and the rescue
    // path safe to enable. 0 disables the gate.
    int formMin = 3;

    // §6.5 low-confidence rescue
    bool rescueEnabled = true;
    float rescueConf = 0.04f;
    int rescueMinCluster = 2;
    float rescueXTol = 20.0f;
    float rescueSpacingTol = 1.6f;
    float dedupYFrac = 0.6f;

    // §6.6 box snapping
    bool snapEnabled = true;
    int snapMargin = 4;
    float snapMinArea = 0.07f;
    float snapMaxArea = 0.95f;
    float snapMinAspect = 0.4f;
    float snapMaxAspect = 3.0f;
    float snapRectangularity = 0.75f;

    // §6.7 ink density
    float inkThresh = 0.05f;
    float inkBorder = 0.25f;
    int inkDark = 128;
};

struct Checkbox {
    int x1, y1, x2, y2;
    bool checked;
    float inkRatio;
    float confidence;
    bool fromRescue;// "source": false = main, true = rescue
    bool snapped;   // false = snapping fell back to the raw network box
};

// A detection that did not make it into the result, with the reason it was
// dropped: "saturation", "below-conf", "rescue-pruned" or "nms". Only
// collected when asked for (--debug-checkbox-candidates).
struct CheckboxCandidate {
    float x1, y1, x2, y2;
    float confidence;
    int classId;
    std::string reason;
};

struct CheckboxResult {
    std::vector<Checkbox> checkboxes;
    bool isForm = false;
    std::vector<CheckboxCandidate> discarded;
};

// Runs the full pipeline: inference -> saturation filter -> NMS -> document
// gate -> optional low-confidence rescue -> snap -> ink density.
CheckboxResult detectCheckboxes(CheckboxNet &net, const cv::Mat &bgr, const CheckboxParams &params,
                                bool collectCandidates);

#endif //__PADDLEVINO_CHECKBOX_DETECTOR_H__
