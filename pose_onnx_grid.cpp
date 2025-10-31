#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <numeric>
#include <sstream>
#include <algorithm>

namespace fs = std::filesystem;

// Detection structure (box + score + label)
struct Det {
    float x1, y1, x2, y2;
    float score;
    int label;
};

// IoU helper
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

// Simple NMS (class-agnostic)
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
            float iou = IOU_box(dets[i], dets[j]);
            if (iou <= iou_thresh) remaining.push_back(j);
        }
        idxs = remaining;
    }
    return keep;
}

int main() {
    try {
        // -------------------------
        // Settings (edit as needed)
        // -------------------------
        const std::wstring model_path = L"keypointrcnn_pose_human.onnx"; // set your keypoint rcnn onnx path
        const std::string images_dir = "human-tests";                  // images folder
        const int inputW = 640;
        const int inputH = 480;
        const float confThresh = 0.5f;
        const float iouThresh = 0.5f;
        const int max_images = 8;

        // COCO-like classes (keypoint r-cnn often returns human/person; still keep general)
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

        // Skeleton pairs (COCO 17 keypoints)
        std::vector<std::pair<int, int>> skeleton_pairs = {
            {0,1},{0,2},{1,3},{2,4},
            {0,5},{0,6},{5,6},{5,7},{7,9},
            {6,8},{8,10},{5,11},{6,12},
            {11,12},{11,13},{13,15},{12,14},{14,16}
        };

        // -------------------------
        // ONNX Runtime init
        // -------------------------
        //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "KeypointRCNN");
        //Ort::SessionOptions session_options;
        //session_options.SetIntraOpNumThreads(1);
        //// If you want CUDA, append provider here (requires building ORT with CUDA and linking) similar to your C# code.
        //Ort::Session session(env, model_path.c_str(), session_options);
        //Ort::AllocatorWithDefaultOptions allocator;

        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "KeypointRCNN");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Try CUDA first, fall back to CPU if unavailable
        bool use_cuda = false;
        try {
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            use_cuda = true;
            std::cout << "[INFO] Using ONNX Runtime CUDA Execution Provider\n";
        }
        catch (const std::exception& e) {
            std::cout << "[WARN] CUDA EP unavailable, falling back to CPU: " << e.what() << "\n";
        }

        // Create session
        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::cout << "[INFO] Model loaded successfully ("
            << (use_cuda ? "GPU" : "CPU") << ")\n";

        // -------------------------
        // collect image files
        // -------------------------
        std::vector<std::string> all_images;
        for (const auto& entry : fs::directory_iterator(images_dir)) {
            if (entry.is_regular_file()) all_images.push_back(entry.path().string());
        }
        if (all_images.empty()) { std::cerr << "No images found\n"; return -1; }
        std::shuffle(all_images.begin(), all_images.end(), std::mt19937{ std::random_device{}() });
        if ((int)all_images.size() > max_images) all_images.resize(max_images);

        // prepare output names (we will request all model outputs)
        size_t out_count = session.GetOutputCount();
        std::vector<const char*> output_names;
        std::vector<Ort::AllocatedStringPtr> out_name_ptrs;
        out_name_ptrs.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            out_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(out_name_ptrs.back().get());
        }

        // get input name
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();

        // -------------------------
        // inference loop
        // -------------------------
        std::vector<cv::Mat> vis_images;
        for (const auto& image_path : all_images) {
            cv::Mat img_bgr = cv::imread(image_path);
            if (img_bgr.empty()) { std::cerr << "Failed to read: " << image_path << "\n"; continue; }

            int origW = img_bgr.cols;
            int origH = img_bgr.rows;

            // BGR -> RGB
            cv::Mat img_rgb;
            cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

            // Resize & normalize (stretch to input size, matching your C#)
            cv::Mat resized;
            cv::resize(img_rgb, resized, cv::Size(inputW, inputH));
            resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

            // HWC -> CHW
            std::vector<float> input_tensor_values((size_t)1 * 3 * inputH * inputW);
            size_t idx = 0;
            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < inputH; ++y) {
                    for (int x = 0; x < inputW; ++x) {
                        cv::Vec3f px = resized.at<cv::Vec3f>(y, x);
                        input_tensor_values[idx++] = px[c];
                    }
                }
            }

            std::array<int64_t, 4> input_dims = { 1, 3, inputH, inputW };
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size()
            );

            // Run and get multiple outputs
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, &input_name, &input_tensor, 1, output_names.data(), (int)output_names.size());

            // Convert outputs to float vectors (safe extraction)
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

            // Map outputs: boxes, labels, scores, keypoints, keypoint_scores
            std::vector<float>& boxesArr = outputs_float.size() > 0 ? outputs_float[0] : *(new std::vector<float>());
            std::vector<float>& labelsArr = outputs_float.size() > 1 ? outputs_float[1] : *(new std::vector<float>());
            std::vector<float>& scoresArr = outputs_float.size() > 2 ? outputs_float[2] : *(new std::vector<float>());
            std::vector<float>& keypointsArr = outputs_float.size() > 3 ? outputs_float[3] : *(new std::vector<float>());
            std::vector<float>& keypointScoresArr = outputs_float.size() > 4 ? outputs_float[4] : *(new std::vector<float>());

            // Basic sanity checks
            if (scoresArr.empty() || boxesArr.empty()) {
                std::cerr << "Model returned empty boxes or scores. Skipping image.\n";
                vis_images.push_back(img_bgr);
                continue;
            }

            // number of detections (assuming 1D scores array length = N)
            size_t numDetections = scoresArr.size();

            // infer numKeypoints if keypoints present: keypointsArr size = N * K * 3
            int numKeypoints = 0;
            if (!keypointsArr.empty()) {
                if (keypointsArr.size() % (numDetections * 3) == 0) {
                    numKeypoints = (int)(keypointsArr.size() / (numDetections * 3));
                }
                else {
                    // fallback: try integer division, but warn
                    numKeypoints = (int)(keypointsArr.size() / (numDetections * 3));
                    std::cerr << "Warning: inferred numKeypoints = " << numKeypoints << "\n";
                }
            }

            // Build detection list + per-detection keypoints arrays (only for scored detections)
            std::vector<Det> dets_all; dets_all.reserve(numDetections);
            std::vector<std::vector<float>> det_keypoints; // flattened [K * 3] per detection
            std::vector<std::vector<float>> det_kpt_scores; // [K] per detection

            for (size_t i = 0; i < numDetections; ++i) {
                float score = scoresArr[i];
                if (score < confThresh) {
                    // keep indices aligned? we skip them here.
                    continue;
                }

                // Boxes expected as [N,4] flattened
                if (boxesArr.size() >= (i * 4 + 4)) {
                    float x1 = boxesArr[i * 4 + 0];
                    float y1 = boxesArr[i * 4 + 1];
                    float x2 = boxesArr[i * 4 + 2];
                    float y2 = boxesArr[i * 4 + 3];
                    int label = 0;
                    if (labelsArr.size() > i) label = static_cast<int>(std::round(labelsArr[i]));
                    Det d; d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2; d.score = score; d.label = label;
                    dets_all.push_back(d);
                }
                else {
                    // no proper box, skip
                    continue;
                }

                // keypoints: expected layout [N, K, 3] flattened (x,y,visibility)
                if (!keypointsArr.empty() && numKeypoints > 0 && keypointsArr.size() >= (i + 1) * numKeypoints * 3) {
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
                    det_keypoints.push_back(std::vector<float>()); // empty placeholder
                }

                // keypoint scores: expected layout [N, K]
                if (!keypointScoresArr.empty() && numKeypoints > 0 && keypointScoresArr.size() >= (i + 1) * (size_t)numKeypoints) {
                    std::vector<float> ks(numKeypoints);
                    size_t base = i * (size_t)numKeypoints;
                    for (int k = 0; k < numKeypoints; ++k) ks[k] = keypointScoresArr[base + k];
                    det_kpt_scores.push_back(std::move(ks));
                }
                else {
                    det_kpt_scores.push_back(std::vector<float>()); // empty placeholder
                }
            } // per detection

            // If no detections after filtering
            if (dets_all.empty()) {
                vis_images.push_back(img_bgr);
                continue;
            }

            // Apply NMS indices on dets_all
            std::vector<int> keep = NMS_indices(dets_all, iouThresh);

            // Draw results: scale boxes/keypoints back to original image size
            float scaleX = static_cast<float>(origW) / static_cast<float>(inputW);
            float scaleY = static_cast<float>(origH) / static_cast<float>(inputH);

            cv::Mat vis = img_bgr.clone(); // draw on original BGR mat

            // pens/colors equivalents (BGR)
            cv::Scalar box_color(0, 255, 0);      // green box
            cv::Scalar text_color(0, 255, 255);   // yellow text
            cv::Scalar kpt_color(0, 0, 255);      // red keypoint
            cv::Scalar skel_color(255, 255, 0);   // cyan skeleton

            for (int kept_idx : keep) {
                if (kept_idx < 0 || kept_idx >= (int)dets_all.size()) continue;
                const Det& d = dets_all[kept_idx];

                int x1 = (int)std::round(d.x1 * scaleX);
                int y1 = (int)std::round(d.y1 * scaleY);
                int x2 = (int)std::round(d.x2 * scaleX);
                int y2 = (int)std::round(d.y2 * scaleY);
                int label = d.label;
                float conf = d.score;

                // draw bbox
                cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 4);

                // draw label
                std::stringstream ss; ss << (label >= 0 && label < (int)classes.size() ? classes[label] : "obj") << " " << std::fixed << std::setprecision(2) << conf;
                std::string txt = ss.str();
                int baseline = 0;
                double fontScale = 1.1;
                int thickness = 3;
                cv::Size txtSize = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
                int tx = x1;
                int ty = std::max(0, y1 - 15);
                cv::putText(vis, txt, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, fontScale, text_color, thickness);

                // draw keypoints + skeleton if available
                if (kept_idx < (int)det_keypoints.size() && det_keypoints[kept_idx].size() >= (size_t)numKeypoints * 3) {
                    const std::vector<float>& kps = det_keypoints[kept_idx];
                    const std::vector<float>& kpscores = det_kpt_scores[kept_idx];
                    std::vector<cv::Point> scaled_kp(numKeypoints, cv::Point(-1, -1));
                    // draw circles
                    for (int k = 0; k < numKeypoints; ++k) {
                        float kx = kps[k * 3 + 0];
                        float ky = kps[k * 3 + 1];
                        float kv = kps[k * 3 + 2]; // sometimes visibility
                        float sc = 0.0f;
                        if (!kpscores.empty() && (int)kpscores.size() > k) sc = kpscores[k];
                        // choose threshold: use keypoint score if present otherwise use kv as visibility
                        float score_k = (!kpscores.empty()) ? sc : kv;
                        if (score_k > 0.3f) {
                            int px = (int)std::round(kx * scaleX);
                            int py = (int)std::round(ky * scaleY);
                            scaled_kp[k] = cv::Point(px, py);
                            cv::circle(vis, cv::Point(px, py), 8, kpt_color, -1);
                        }
                    }
                    // draw skeleton pairs
                    for (const auto& p : skeleton_pairs) {
                        int a = p.first, b = p.second;
                        if (a < 0 || a >= numKeypoints || b < 0 || b >= numKeypoints) continue;
                        if (scaled_kp[a].x >= 0 && scaled_kp[b].x >= 0) {
                            cv::line(vis, scaled_kp[a], scaled_kp[b], skel_color, 5);
                        }
                    }
                }
            } // kept detections

            vis_images.push_back(vis);
        } // for each image

        // -------------------------
        // Display grid (2x4 same as prior code)
        // -------------------------
        if (vis_images.empty()) {
            std::cerr << "No images to display\n";
            return -1;
        }

        int rows = 2, cols = 4;
        int cell_w = 256, cell_h = 256;
        cv::Mat grid(rows * cell_h, cols * cell_w, vis_images[0].type(), cv::Scalar(0, 0, 0));

        for (size_t i = 0; i < vis_images.size(); ++i) {
            int r = i / cols;
            int c = i % cols;
            cv::Mat img = vis_images[i];

            float scale = std::min((float)cell_w / img.cols, (float)cell_h / img.rows);
            int new_w = (int)(img.cols * scale);
            int new_h = (int)(img.rows * scale);
            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h));

            cv::Mat canvas(cell_h, cell_w, img.type(), cv::Scalar(0, 0, 0));
            int offset_x = (cell_w - new_w) / 2;
            int offset_y = (cell_h - new_h) / 2;
            resized.copyTo(canvas(cv::Rect(offset_x, offset_y, new_w, new_h)));

            cv::Rect roi(c * cell_w, r * cell_h, cell_w, cell_h);
            canvas.copyTo(grid(roi));
        }

        cv::namedWindow("Keypoint R-CNN ONNX Grid", cv::WINDOW_AUTOSIZE);
        cv::imshow("Keypoint R-CNN ONNX Grid", grid);
        cv::waitKey(0);
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
