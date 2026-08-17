#include "CheckboxNet.h"
#include "OcrUtils.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>

void CheckboxNet::setEngine(EngineType engine) {
    if (engine != EngineType::OpenVINO) {
        return;
    }
    try {
        std::unordered_map<std::string, std::string> options;
        sessionOptions.AppendExecutionProvider("OpenVINO", options);
        fprintf(stderr, "checkbox: using OpenVINO execution provider\n");
    } catch (const Ort::Exception &e) {
        fprintf(stderr, "checkbox: OpenVINO execution provider unavailable (%s), falling back to CPU\n", e.what());
    }
}

CheckboxNet::~CheckboxNet() {
    delete session;
    inputNamesPtr.clear();
    outputNamesPtr.clear();
}

void CheckboxNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    sessionOptions.SetInterOpNumThreads(numThread);
    sessionOptions.SetIntraOpNumThreads(numThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void CheckboxNet::initModel(const std::string &pathStr) {
#ifdef _WIN32
    std::wstring modelPath = strToWstr(pathStr);
    session = new Ort::Session(env, modelPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
#endif
    inputNamesPtr = getInputNames(session);
    outputNamesPtr = getOutputNames(session);
}

std::vector<CheckboxDetection> CheckboxNet::predict(const cv::Mat &srcBgr, int inputSize, float confThresh) {
    std::vector<CheckboxDetection> detections;
    if (session == nullptr || srcBgr.empty() || inputSize <= 0) return detections;

    const int imgW = srcBgr.cols;
    const int imgH = srcBgr.rows;

    // ----- letterbox: keep aspect ratio, pad with grey (114,114,114), centred
    const double scale = (std::min)((double) inputSize / imgW, (double) inputSize / imgH);
    const int newW = (std::max)(1, (int) std::lround(imgW * scale));
    const int newH = (std::max)(1, (int) std::lround(imgH * scale));
    const int padX = (inputSize - newW) / 2;
    const int padY = (inputSize - newH) / 2;

    // INTER_AREA, not INTER_LINEAR: the prototype resizes with PIL, whose
    // BILINEAR scales the filter support by the reduction factor (i.e. it
    // antialiases). A full page is downscaled ~2x to reach 1024, so plain
    // INTER_LINEAR aliases badly and the network loses boxes -- measured, it
    // dropped a 0.35-confidence checkbox on a sample form and shifted every score.
    // With INTER_AREA the C++ output matches the Python golden files.
    cv::Mat resized;
    cv::resize(srcBgr, resized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
    cv::Mat canvas(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(padX, padY, newW, newH)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    // ----- NCHW float32, RGB, divided by 255
    const size_t plane = (size_t) inputSize * inputSize;
    std::vector<float> inputTensorValues(plane * 3);
    for (int y = 0; y < inputSize; ++y) {
        const uchar *row = rgb.ptr<uchar>(y);
        for (int x = 0; x < inputSize; ++x) {
            const size_t offset = (size_t) y * inputSize + x;
            inputTensorValues[offset] = row[x * 3 + 0] / 255.0f;
            inputTensorValues[plane + offset] = row[x * 3 + 1] / 255.0f;
            inputTensorValues[2 * plane + offset] = row[x * 3 + 2] / 255.0f;
        }
    }

    std::array<int64_t, 4> inputShape{1, 3, inputSize, inputSize};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size());
    std::vector<const char *> inputNames = {inputNamesPtr.data()->get()};
    std::vector<const char *> outputNames = {outputNamesPtr.data()->get()};
    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                     inputNames.size(), outputNames.data(), outputNames.size());
    if (outputTensor.empty() || !outputTensor.front().IsTensor()) return detections;

    // ----- output (1, 6, N) -> transposed to (N, 6) = cx, cy, w, h, score_class0, score_class1
    std::vector<int64_t> outputShape = outputTensor.front().GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() != 3) return detections;
    const int64_t attrs = outputShape[1];
    const int64_t count = outputShape[2];
    if (attrs < 6 || count <= 0) return detections;
    const float *data = outputTensor.front().GetTensorMutableData<float>();

    for (int64_t i = 0; i < count; ++i) {
        const float s0 = data[4 * count + i];
        const float s1 = data[5 * count + i];
        const int classId = s1 > s0 ? 1 : 0;// argmax, ties go to class 0 like numpy's argmax
        const float score = classId == 1 ? s1 : s0;
        if (score < confThresh) continue;

        const float cx = data[0 * count + i];
        const float cy = data[1 * count + i];
        const float bw = data[2 * count + i];
        const float bh = data[3 * count + i];

        CheckboxDetection d{};
        d.x1 = (float) (std::max)(0.0, (cx - bw / 2.0f - padX) / scale);
        d.y1 = (float) (std::max)(0.0, (cy - bh / 2.0f - padY) / scale);
        d.x2 = (float) (std::min)((double) imgW, (cx + bw / 2.0f - padX) / scale);
        d.y2 = (float) (std::min)((double) imgH, (cy + bh / 2.0f - padY) / scale);
        d.classId = classId;
        d.score = score;
        detections.push_back(d);
    }
    return detections;
}
