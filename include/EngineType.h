#ifndef __PADDLEVINO_ENGINE_TYPE_H__
#define __PADDLEVINO_ENGINE_TYPE_H__

// Execution backend requested on the command line for the three ONNX Runtime
// sessions (detection / angle-classification / recognition).
enum class EngineType {
    CPU = 0,      // Plain ONNX Runtime CPU execution provider (always available).
    OpenVINO = 1, // Attempt the OpenVINO execution provider, falling back to CPU
                  // with a warning if the linked ONNX Runtime build does not
                  // have OpenVINO EP compiled in.
};

inline const char *engineTypeName(EngineType engine) {
    switch (engine) {
        case EngineType::OpenVINO:
            return "openvino";
        case EngineType::CPU:
        default:
            return "cpu";
    }
}

#endif //__PADDLEVINO_ENGINE_TYPE_H__
