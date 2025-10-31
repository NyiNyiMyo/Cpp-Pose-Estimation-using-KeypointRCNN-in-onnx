// keypointrcnn_webcam.cpp
// Real-time Keypoint R-CNN inference from webcam using ONNX Runtime (CUDA if available) + OpenCV
// Build with ONNX Runtime (GPU enabled) and OpenCV.
// Example (Linux): g++ -O3 keypointrcnn_webcam.cpp -o kpwebcam `pkg-config --cflags --libs opencv4` -lonnxruntime

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <chrono>

struct Det {
    float x1, y1, x2, y2;
    float score;
    int label;
};

float IOU_box(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6f);
}

std::vector<int> NMS_indices(const std::vector<Det>& dets, float iou_thresh) {
    std::vector<int> idxs(dets.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) { return dets[a].score > dets[b].score; });
    std::vector<int> keep;
    while (!idxs.empty()) {
        int i = idxs[0];
        keep.push_back(i);
        std::vector<int> remaining;
        for (size_t k = 1; k < idxs.size(); ++k) {
            int j = idxs[k];
            if (IOU_box(dets[i], dets[j]) <= iou_thresh) remaining.push_back(j);
        }
        idxs = std::move(remaining);
    }
    return keep;
}

int main(int argc, char** argv) {
    try {
        // -------------------------
        // Settings (edit as needed)
        // -------------------------
        const std::wstring model_path = L"keypointrcnn_pose_human.onnx"; // set path
        const int inputW = 640;
        const int inputH = 480;
        const float confThresh = 0.5f;
        const float iouThresh = 0.5f;
        const float kpt_score_thresh = 0.3f; // threshold for drawing keypoints

        // COCO-like classes (mostly person for keypoint r-cnn)
        std::vector<std::string> classes = {
            "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
            "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
            "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
            "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
            "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
            "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
            "clock","vase","scissors","teddy bear","hair drier","toothbrush"
        };

        // COCO 17 skeleton pairs
        std::vector<std::pair<int, int>> skeleton_pairs = {
            {0,1},{0,2},{1,3},{2,4},
            {0,5},{0,6},{5,6},{5,7},{7,9},
            {6,8},{8,10},{5,11},{6,12},
            {11,12},{11,13},{13,15},{12,14},{14,16}
        };

        // -------------------------
        // ONNX Runtime init
        // -------------------------
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "KeypointRCNN");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        // Recommended optimizations:
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();

        bool use_cuda = false;
        // Try append CUDA EP (requires ORT built with CUDA)
        try {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            use_cuda = true;
            std::cout << "[INFO] Using ONNX Runtime CUDA Execution Provider\n";
        }
        catch (const std::exception& e) {
            std::cout << "[WARN] CUDA EP unavailable, falling back to CPU: " << e.what() << "\n";
        }

        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "[INFO] Model loaded successfully (" << (use_cuda ? "GPU" : "CPU") << ")\n";

        // prepare output names (request all outputs)
        size_t out_count = session.GetOutputCount();
        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
        out_name_ptrs.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            out_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(out_name_ptrs.back().get());
        }
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();

        // Pre-allocate input buffer (reuse every frame)
        std::vector<float> input_tensor_values((size_t)1 * 3 * inputH * inputW);

        std::array<int64_t, 4> input_dims = { 1, 3, inputH, inputW };
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // -------------------------
        // Open webcam
        // -------------------------
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open webcam\n";
            return -1;
        }
        // set capture resolution (optional)
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

        cv::namedWindow("KeypointRCNN Webcam", cv::WINDOW_AUTOSIZE);

        // Timing / FPS
        int frame_count = 0;
        double avg_infer_ms = 0.0;
        auto t_last_fps = std::chrono::high_resolution_clock::now();

        while (true) {
            cv::Mat frame_bgr;
            if (!cap.read(frame_bgr)) {
                std::cerr << "Failed to grab frame\n";
                break;
            }

            int origW = frame_bgr.cols;
            int origH = frame_bgr.rows;

            // Preprocess: BGR -> RGB, resize, normalize, HWC->CHW, fill input tensor
            cv::Mat frame_rgb;
            cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

            cv::Mat resized;
            cv::resize(frame_rgb, resized, cv::Size(inputW, inputH));
            resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

            // Fill the preallocated input buffer in channel-major order
            size_t idx = 0;
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < inputH; ++y) {
                    const cv::Vec3f* row_ptr = resized.ptr<cv::Vec3f>(y);
                    for (int x = 0; x < inputW; ++x) {
                        input_tensor_values[idx++] = row_ptr[x][c];
                    }
                }
            }

            // Create input tensor (point to input_tensor_values.data())
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size()
            );

            // Run inference and measure time
            auto t0 = std::chrono::high_resolution_clock::now();
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, output_names.data(), (int)output_names.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // update running average
            avg_infer_ms = (avg_infer_ms * frame_count + infer_ms) / (frame_count + 1);
            frame_count++;

            // Convert outputs to vectors of floats
            std::vector<std::vector<float>> outputs_float;
            outputs_float.reserve(output_tensors.size());
            for (auto& ot : output_tensors) {
                size_t n = ot.GetTensorTypeAndShapeInfo().GetElementCount();
                std::vector<float> tmp;
                if (n > 0) {
                    float* data_ptr = ot.GetTensorMutableData<float>();
                    tmp.assign(data_ptr, data_ptr + n);
                }
                outputs_float.push_back(std::move(tmp));
            }

            // Map outputs (robust to missing outputs)
            std::vector<float> boxesArr = outputs_float.size() > 0 ? outputs_float[0] : std::vector<float>();
            std::vector<float> labelsArr = outputs_float.size() > 1 ? outputs_float[1] : std::vector<float>();
            std::vector<float> scoresArr = outputs_float.size() > 2 ? outputs_float[2] : std::vector<float>();
            std::vector<float> keypointsArr = outputs_float.size() > 3 ? outputs_float[3] : std::vector<float>();
            std::vector<float> keypointScoresArr = outputs_float.size() > 4 ? outputs_float[4] : std::vector<float>();

            // Validate required arrays
            std::vector<Det> dets_all;
            std::vector<std::vector<float>> det_keypoints; // per detection flattened K*3
            std::vector<std::vector<float>> det_kpt_scores; // per detection K

            if (!scoresArr.empty() && !boxesArr.empty()) {
                size_t numDetections = scoresArr.size();
                int numKeypoints = 0;
                if (!keypointsArr.empty() && numDetections > 0) {
                    // infer K from keypoints length = N * K * 3
                    numKeypoints = (int)(keypointsArr.size() / (numDetections * 3));
                    if (numKeypoints <= 0) numKeypoints = 0;
                }

                for (size_t i = 0; i < numDetections; ++i) {
                    float score = scoresArr[i];
                    if (score < confThresh) continue;
                    if (boxesArr.size() < (i * 4 + 4)) continue;
                    float x1 = boxesArr[i * 4 + 0];
                    float y1 = boxesArr[i * 4 + 1];
                    float x2 = boxesArr[i * 4 + 2];
                    float y2 = boxesArr[i * 4 + 3];
                    int label = 0;
                    if (labelsArr.size() > i) label = static_cast<int>(std::round(labelsArr[i]));
                    Det d{ x1, y1, x2, y2, score, label };
                    dets_all.push_back(d);

                    // keypoints
                    if (numKeypoints > 0 && keypointsArr.size() >= (i + 1) * (size_t)numKeypoints * 3) {
                        std::vector<float> kps(numKeypoints * 3);
                        size_t base = i * (size_t)numKeypoints * 3;
                        for (int k = 0; k < numKeypoints; ++k) {
                            kps[k * 3 + 0] = keypointsArr[base + k * 3 + 0];
                            kps[k * 3 + 1] = keypointsArr[base + k * 3 + 1];
                            kps[k * 3 + 2] = keypointsArr[base + k * 3 + 2];
                        }
                        det_keypoints.push_back(std::move(kps));
                    }
                    else {
                        det_keypoints.emplace_back(); // empty placeholder
                    }

                    // keypoint scores
                    if (numKeypoints > 0 && keypointScoresArr.size() >= (i + 1) * (size_t)numKeypoints) {
                        std::vector<float> ks(numKeypoints);
                        size_t base = i * (size_t)numKeypoints;
                        for (int k = 0; k < numKeypoints; ++k) ks[k] = keypointScoresArr[base + k];
                        det_kpt_scores.push_back(std::move(ks));
                    }
                    else {
                        det_kpt_scores.emplace_back();
                    }
                }
            }

            // Prepare visualization on original frame
            cv::Mat vis = frame_bgr.clone();

            if (!dets_all.empty()) {
                // NMS
                std::vector<int> keep = NMS_indices(dets_all, iouThresh);
                // scale factors (model coordinates -> original frame coordinates)
                float scaleX = static_cast<float>(origW) / static_cast<float>(inputW);
                float scaleY = static_cast<float>(origH) / static_cast<float>(inputH);

                // draw
                for (int kept_idx : keep) {
                    if (kept_idx < 0 || kept_idx >= (int)dets_all.size()) continue;
                    const Det& d = dets_all[kept_idx];
                    int x1 = (int)std::round(d.x1 * scaleX);
                    int y1 = (int)std::round(d.y1 * scaleY);
                    int x2 = (int)std::round(d.x2 * scaleX);
                    int y2 = (int)std::round(d.y2 * scaleY);
                    int label = d.label;
                    float conf = d.score;

                    cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                    std::stringstream ss; ss << (label >= 0 && label < (int)classes.size() ? classes[label] : "obj") << " " << std::fixed << std::setprecision(2) << conf;
                    cv::putText(vis, ss.str(), cv::Point(x1, std::max(0, y1 - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

                    // draw keypoints if available
                    if (kept_idx < (int)det_keypoints.size() && !det_keypoints[kept_idx].empty()) {
                        const std::vector<float>& kps = det_keypoints[kept_idx];
                        const std::vector<float>& kscores = det_kpt_scores[kept_idx];
                        int numKeypoints = (int)kps.size() / 3;
                        std::vector<cv::Point> scaled_kp(numKeypoints, cv::Point(-1, -1));
                        for (int k = 0; k < numKeypoints; ++k) {
                            float kx = kps[k * 3 + 0];
                            float ky = kps[k * 3 + 1];
                            float kv = kps[k * 3 + 2];
                            float score_k = (kscores.size() == (size_t)numKeypoints) ? kscores[k] : kv;
                            if (score_k > kpt_score_thresh) {
                                int px = (int)std::round(kx * scaleX);
                                int py = (int)std::round(ky * scaleY);
                                scaled_kp[k] = cv::Point(px, py);
                                cv::circle(vis, cv::Point(px, py), 3, cv::Scalar(0, 0, 255), -1);
                            }
                        }
                        // draw skeleton
                        for (const auto& p : skeleton_pairs) {
                            int a = p.first, b = p.second;
                            if (a < 0 || a >= numKeypoints || b < 0 || b >= numKeypoints) continue;
                            if (scaled_kp[a].x >= 0 && scaled_kp[b].x >= 0) {
                                cv::line(vis, scaled_kp[a], scaled_kp[b], cv::Scalar(255, 255, 0), 2);
                            }
                        }
                    }
                }
            }

            // Overlay performance info
            std::stringstream perf;
            perf << "Infer: " << std::fixed << std::setprecision(1) << infer_ms << " ms  Avg: " << std::fixed << std::setprecision(1) << avg_infer_ms << " ms";
            cv::putText(vis, perf.str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::imshow("KeypointRCNN Webcam", vis);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') break; // ESC or q to quit

            // Optional: print more verbose every 120 frames
            if (frame_count % 120 == 0) {
                std::cout << "[INFO] frame=" << frame_count << " infer_ms=" << infer_ms << " avg_infer_ms=" << avg_infer_ms << "\n";
            }
        } // while

        cap.release();
        cv::destroyAllWindows();

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}
