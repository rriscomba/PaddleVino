#ifndef __PADDLEVINO_CHECKBOX_NET_H__
#define __PADDLEVINO_CHECKBOX_NET_H__

#include "EngineType.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// One raw detection from the YOLO checkbox model, already mapped back to
// original-image pixel coordinates. Nothing has been filtered yet beyond the
// confidence threshold: no saturation filter, no NMS, no snapping.
struct CheckboxDetection {
    float x1;
    float y1;
    float x2;
    float y2;
    int classId;// 0 = checked, 1 = unchecked
    float score;
};

// YOLO12n checkbox detector (wendys-llc/checkbox-detector) run through ONNX
// Runtime. Follows the DbNet/AngleNet/CrnnNet pattern so it inherits the
// `--engine openvino` execution-provider path for free.
class CheckboxNet {
public:
    ~CheckboxNet();

    void setNumThread(int numOfThread);

    void setEngine(EngineType engine);

    void initModel(const std::string &pathStr);

    // Runs one forward pass over `srcBgr` and returns every prediction whose
    // class score reaches `confThresh`, in original-image coordinates.
    std::vector<CheckboxDetection> predict(const cv::Mat &srcBgr, int inputSize, float confThresh);

private:
    Ort::Session *session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CheckboxNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;
};

#endif //__PADDLEVINO_CHECKBOX_NET_H__
