#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "OcrStruct.h"
#include "EngineType.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

class DbNet {
public:
    ~DbNet();

    void setNumThread(int numOfThread);

    void setGpuIndex(int gpuIndex);

    void setEngine(EngineType engine);

    void initModel(const std::string &pathStr);

    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
                                      float boxThresh, float unClipRatio);

private:
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;

    // The bundled PP-OCRv6 det model (see RapidOCR's config.yaml, Det block,
    // ocr_version: PP-OCRv6) uses simple (pixel/255 - 0.5)/0.5 normalization,
    // not the ImageNet mean/std the PP-OCRv3/v4 det model used.
    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
};


#endif //__OCR_DBNET_H__
