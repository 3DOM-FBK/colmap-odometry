// Learned local features with ONNX Runtime (C++ API): ALIKED and XFeat.
//
// Both are cv::Feature2D, like ORB and SIFT, so the odometry uses them the same
// way (detectAndCompute on a grayscale image). Descriptors are float (CV_32F),
// L2-normalized: matched with L2 distance.
//
// The ONNX models (exported by models/export_models.sh) hold the networks; the
// sparse steps around them are here, ported from the reference code:
//   XFeat  (models/xfeat.onnx): XFeat.detectAndCompute of
//          verlab/accelerated_features (modules/xfeat.py): resize to multiples
//          of 32, NMS 5x5 > 0.05, score = heatmap * reliability, top-k,
//          bicubic descriptor sampling.
//   ALIKED (models/aliked-n16_dense.onnx + aliked-n16_desc.onnx): ALIKED in
//          LightGlue (lightglue/aliked.py), top-k mode: pad to multiples of 32,
//          DKD (NMS radius 2, border 2, top-k, sub-pixel soft-argmax over 5x5,
//          bilinear score), SDDH descriptors (second model).
//
// The model runs on the GPU (CUDA execution provider) when available, on the
// CPU otherwise. Compiled only with COLMAP_ODOMETRY_WITH_ONNX (CMakeLists.txt).
//
// Every header used here is included by matching.cpp before this file.

// ------------------------------------------------------------ ONNX Runtime --
inline Ort::Env& ortEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "colmap-odometry");
    return env;
}

// True if the NVIDIA driver is loaded and sees a GPU. Checked before creating a
// CUDA session: ONNX Runtime's CUDA provider crashes (instead of reporting an
// error) when there is no driver, e.g. in a container started without the GPU.
inline bool cudaDevicePresent() {
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) return false;
    using InitFn = int (*)(unsigned int);
    using CountFn = int (*)(int*);
    const auto init = reinterpret_cast<InitFn>(dlsym(driver, "cuInit"));
    const auto count = reinterpret_cast<CountFn>(dlsym(driver, "cuDeviceGetCount"));
    int devices = 0;
    const bool ok = init && count && init(0) == 0 && count(&devices) == 0 && devices > 0;
    dlclose(driver);
    return ok;
}

// One ONNX model: session on CUDA if requested and available, else CPU.
class OrtModel {
public:
    OrtModel(const std::string& path, const std::string& device, bool fixed_input_size) {
        if (device != "cuda" && device != "cpu") {
            throw std::runtime_error("feature device must be cuda or cpu, not " + device);
        }
        if (!fs::exists(path)) {
            throw std::runtime_error("ONNX model not found: " + path +
                                     " (export it with models/export_models.sh)");
        }
        std::string reason;
        if (device == "cuda") {
            const auto providers = Ort::GetAvailableProviders();
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
                providers.end()) {
                reason = "this ONNX Runtime build has no CUDA provider";
            } else if (!cudaDevicePresent()) {
                reason = "no NVIDIA driver or GPU found";
            } else {
                try {
                    session_ = makeSession(path, true, fixed_input_size);
                    device_ = "cuda";
                } catch (const Ort::Exception& e) {
                    reason = e.what();
                }
            }
        }
        if (!session_) {
            if (!reason.empty()) {
                std::cerr << fs::path(path).filename().string() << ": CUDA not available ("
                          << reason << "), using the CPU\n";
            }
            session_ = makeSession(path, false, fixed_input_size);
            device_ = "cpu";
        }
    }

    const std::string& device() const { return device_; }

    // Runs are serialized (see sharedModel): one at a time on the GPU.
    std::vector<Ort::Value> run(const std::vector<const char*>& input_names,
                                std::vector<Ort::Value>& inputs,
                                const std::vector<const char*>& output_names) {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                             inputs.size(), output_names.data(), output_names.size());
    }

    void run(Ort::IoBinding& binding) {
        std::lock_guard<std::mutex> lock(mutex_);
        session_->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
    }

    Ort::Session& session() { return *session_; }

    // Where outputs that stay on the model's device are allocated.
    const Ort::MemoryInfo& deviceMemory() const {
        static const Ort::MemoryInfo cuda("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
        return device_ == "cuda" ? cuda : cpuMemory();
    }

    static const Ort::MemoryInfo& cpuMemory() {
        static const Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return memory;
    }

    // Tensors wrapping host buffers (no copy: the buffer must outlive the run).
    template <typename T>
    static Ort::Value tensor(std::vector<T>& data, const std::vector<int64_t>& shape) {
        return Ort::Value::CreateTensor<T>(cpuMemory(), data.data(), data.size(), shape.data(),
                                           shape.size());
    }

private:
    static std::unique_ptr<Ort::Session> makeSession(const std::string& path, bool cuda,
                                                     bool fixed_input_size) {
        Ort::SessionOptions options;
        options.SetInterOpNumThreads(1);
        // cam0 and cam1 run in parallel, each with its own model: half the cores each.
        options.SetIntraOpNumThreads(std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2));
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cuda) {
            const OrtApi& api = Ort::GetApi();
            OrtCUDAProviderOptionsV2* raw = nullptr;
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
            std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)> cuda_options(
                raw, [](OrtCUDAProviderOptionsV2* p) { Ort::GetApi().ReleaseCUDAProviderOptions(p); });
            // GPU memory is tight (full-resolution 128-channel maps are 190-300
            // MB each, and a laptop GPU may have 4 GB shared with NetVLAD):
            //  - grow the memory pool by what is requested, not to the next
            //    power of two;
            //  - do not let cuDNN take the largest possible workspace for the
            //    fastest convolution algorithm.
            // Fixed image size during a run: search the fastest convolutions
            // once. Inputs that change every call (keypoints): heuristic.
            const char* keys[] = {"device_id", "arena_extend_strategy", "cudnn_conv_use_max_workspace",
                                  "cudnn_conv_algo_search"};
            const char* values[] = {"0", "kSameAsRequested", "0",
                                    fixed_input_size ? "EXHAUSTIVE" : "HEURISTIC"};
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_options.get(), keys, values, 4));
            options.AppendExecutionProvider_CUDA_V2(*cuda_options);
        }
        return std::make_unique<Ort::Session>(ortEnv(), path.c_str(), options);
    }

    std::unique_ptr<Ort::Session> session_;
    std::string device_;
    std::mutex mutex_;
};

// One model per (file, device) in the process, shared by the detectors of
// cam0 and cam1: half the GPU memory, and runs are serialized by the model.
// Released when the last detector using it is destroyed.
inline std::shared_ptr<OrtModel> sharedModel(const std::string& path, const std::string& device,
                                             bool fixed_input_size) {
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<OrtModel>> models;
    std::lock_guard<std::mutex> lock(mutex);
    std::weak_ptr<OrtModel>& slot = models[path + "|" + device];
    if (std::shared_ptr<OrtModel> model = slot.lock()) return model;
    auto model = std::make_shared<OrtModel>(path, device, fixed_input_size);
    slot = model;
    return model;
}

// ------------------------------------------------------------ sampling --
// torch.nn.functional.grid_sample semantics (padding_mode="zeros") on one
// H x W plane; (x, y) are already unnormalized pixel coordinates.
inline float planeAt(const float* p, int w, int h, int x, int y) {
    return (x < 0 || y < 0 || x >= w || y >= h) ? 0.f : p[static_cast<size_t>(y) * w + x];
}

inline float sampleNearest(const float* p, int w, int h, double x, double y) {
    return planeAt(p, w, h, static_cast<int>(std::nearbyint(x)), static_cast<int>(std::nearbyint(y)));
}

inline float sampleBilinear(const float* p, int w, int h, double x, double y) {
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    const double tx = x - x0, ty = y - y0;
    return static_cast<float>(
        (1 - ty) * ((1 - tx) * planeAt(p, w, h, x0, y0) + tx * planeAt(p, w, h, x0 + 1, y0)) +
        ty * ((1 - tx) * planeAt(p, w, h, x0, y0 + 1) + tx * planeAt(p, w, h, x0 + 1, y0 + 1)));
}

// Bicubic weights of the 4 taps at floor(x) - 1 .. + 2 (PyTorch, A = -0.75).
inline std::array<double, 4> cubicWeights(double t) {
    constexpr double A = -0.75;
    auto near = [](double x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };          // |x| <= 1
    auto far = [](double x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };      // 1 < |x| < 2
    return {far(t + 1), near(t), near(1 - t), far(2 - t)};
}

// ------------------------------------------------------------------ XFeat --
class XFeatFeature2D : public cv::Feature2D {
public:
    XFeatFeature2D(const std::string& models_dir, int top_k, const std::string& device)
        : model_(sharedModel((fs::path(models_dir) / "xfeat.onnx").string(), device, true)),
          top_k_(top_k) {}

    const std::string& device() const { return model_->device(); }
    int descriptorSize() const override { return 64; }
    int descriptorType() const override { return CV_32F; }
    int defaultNorm() const override { return cv::NORM_L2; }
    cv::String getDefaultName() const override { return "XFeat"; }

    void detectAndCompute(cv::InputArray image_in, cv::InputArray /*mask*/,
                          std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                          bool /*use_provided_keypoints*/ = false) override {
        cv::Mat gray = image_in.getMat();
        if (gray.channels() == 3) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        // preprocess_tensor: values 0-255, size floored to multiples of 32
        // (bilinear, align_corners=False == cv::INTER_LINEAR).
        cv::Mat image;
        gray.convertTo(image, CV_32F);
        const int H = image.rows, W = image.cols;
        const int H32 = (H / 32) * 32, W32 = (W / 32) * 32;
        if (H32 == 0 || W32 == 0) throw std::runtime_error("XFeat: image smaller than 32 px");
        const double rh = static_cast<double>(H) / H32, rw = static_cast<double>(W) / W32;
        if (H32 != H || W32 != W) cv::resize(image, image, {W32, H32}, 0, 0, cv::INTER_LINEAR);

        std::vector<float> input(image.begin<float>(), image.end<float>());
        std::vector<Ort::Value> inputs;
        inputs.push_back(OrtModel::tensor(input, {1, 1, H32, W32}));
        std::vector<Ort::Value> out = model_->run({"image"}, inputs, {"feats", "heatmap", "reliability"});
        const float* feats = out[0].GetTensorData<float>();        // 64 x h x w
        const float* heatmap = out[1].GetTensorData<float>();      // H32 x W32
        const float* reliability = out[2].GetTensorData<float>();  // h x w
        const int h = H32 / 8, w = W32 / 8;

        // NMS: local maximum in 5x5 (max_pool2d, padding = -inf) and > 0.05.
        const cv::Mat heat(H32, W32, CV_32F, const_cast<float*>(heatmap));
        cv::Mat local_max;
        cv::dilate(heat, local_max, cv::Mat::ones(5, 5, CV_8U));
        // InterpolateSparse2d normalizes positions by (W-1, H-1) of the image
        // but samples with align_corners=False: map to each plane's pixels.
        auto planeX = [&](double x, int plane_w) { return x * plane_w / (W32 - 1) - 0.5; };
        auto planeY = [&](double y, int plane_h) { return y * plane_h / (H32 - 1) - 0.5; };
        struct Candidate { float score; int x, y; };
        std::vector<Candidate> candidates;
        for (int y = 0; y < H32; ++y) {
            const float* row = heat.ptr<float>(y);
            const float* row_max = local_max.ptr<float>(y);
            for (int x = 0; x < W32; ++x) {
                if (row[x] != row_max[x] || !(row[x] > kDetectionThreshold)) continue;
                if (x == 0 && y == 0) continue;  // the reference marks (0, 0) as padding (score -1)
                const float score =
                    sampleNearest(heatmap, W32, H32, planeX(x, W32), planeY(y, H32)) *
                    sampleBilinear(reliability, w, h, planeX(x, w), planeY(y, h));
                if (score > 0) candidates.push_back({score, x, y});
            }
        }
        const size_t n = std::min(candidates.size(), static_cast<size_t>(top_k_));
        std::partial_sort(candidates.begin(), candidates.begin() + n, candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        candidates.resize(n);

        // Bicubic descriptors (grid_sample bicubic, zeros padding), L2-normalized.
        cv::Mat desc(static_cast<int>(n), 64, CV_32F);
        keypoints.clear();
        keypoints.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const Candidate& c = candidates[i];
            const double fx = planeX(c.x, w), fy = planeY(c.y, h);
            const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
            const auto wx = cubicWeights(fx - x0), wy = cubicWeights(fy - y0);
            float* d = desc.ptr<float>(static_cast<int>(i));
            double norm = 0.0;
            for (int ch = 0; ch < 64; ++ch) {
                const float* plane = feats + static_cast<size_t>(ch) * h * w;
                double v = 0.0;
                for (int j = 0; j < 4; ++j) {
                    double row = 0.0;
                    for (int k = 0; k < 4; ++k) row += wx[k] * planeAt(plane, w, h, x0 - 1 + k, y0 - 1 + j);
                    v += wy[j] * row;
                }
                d[ch] = static_cast<float>(v);
                norm += v * v;
            }
            const float scale = static_cast<float>(1.0 / std::max(std::sqrt(norm), 1e-12));
            for (int ch = 0; ch < 64; ++ch) d[ch] *= scale;
            keypoints.emplace_back(cv::Point2f(static_cast<float>(c.x * rw), static_cast<float>(c.y * rh)),
                                   8.f, -1.f, c.score);
        }
        desc.copyTo(descriptors);
    }

private:
    static constexpr float kDetectionThreshold = 0.05f;
    std::shared_ptr<OrtModel> model_;
    int top_k_;
};

// ----------------------------------------------------------------- ALIKED --
class AlikedFeature2D : public cv::Feature2D {
public:
    AlikedFeature2D(const std::string& models_dir, int top_k, const std::string& device)
        : dense_(sharedModel((fs::path(models_dir) / "aliked-n16_dense.onnx").string(), device, true)),
          desc_(sharedModel((fs::path(models_dir) / "aliked-n16_desc.onnx").string(), device, false)),
          top_k_(top_k) {}

    const std::string& device() const { return dense_->device(); }
    int descriptorSize() const override { return 128; }
    int descriptorType() const override { return CV_32F; }
    int defaultNorm() const override { return cv::NORM_L2; }
    cv::String getDefaultName() const override { return "ALIKED"; }

    void detectAndCompute(cv::InputArray image_in, cv::InputArray /*mask*/,
                          std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                          bool /*use_provided_keypoints*/ = false) override {
        cv::Mat gray = image_in.getMat();
        if (gray.channels() == 3) cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        cv::Mat image;
        gray.convertTo(image, CV_32F, 1.0 / 255.0);  // the model repeats it to RGB
        const int H = image.rows, W = image.cols;

        // InputPadder(H, W, 32): edge replication, split between both sides.
        const int pad_h = ((H / 32 + 1) * 32 - H) % 32, pad_w = ((W / 32 + 1) * 32 - W) % 32;
        const int top = pad_h / 2, left = pad_w / 2;
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, top, pad_h - top, left, pad_w - left, cv::BORDER_REPLICATE);
        const int Hp = padded.rows, Wp = padded.cols;

        // Dense maps. The feature map (128 x Hp x Wp floats: 190 MB at 768 x 480,
        // 300 MB at 960 x 608) stays on the model's device (GPU) and goes
        // straight to the descriptor model; only the score map comes back. It
        // is never cropped either: the dense model sets it to zero outside
        // `crop`, the descriptor model reads it shifted.
        // One extraction at a time in the process (cam0 and cam1 share the
        // models), so only one feature map exists at a time: GPU memory.
        static std::mutex extraction;
        std::lock_guard<std::mutex> lock(extraction);
        std::vector<float> input(padded.begin<float>(), padded.end<float>());
        std::vector<int64_t> crop{top, left, H, W};
        Ort::Value image_tensor = OrtModel::tensor(input, {1, 1, Hp, Wp});
        Ort::Value crop_tensor = OrtModel::tensor(crop, {4});
        Ort::IoBinding binding(dense_->session());
        binding.BindInput("image", image_tensor);
        binding.BindInput("crop", crop_tensor);
        binding.BindOutput("feature_map", dense_->deviceMemory());
        binding.BindOutput("score_map", OrtModel::cpuMemory());
        dense_->run(binding);
        std::vector<Ort::Value> dense = binding.GetOutputValues();  // in binding order
        const float* sm = dense[1].GetTensorData<float>();  // Hp x Wp

        // Unpad the score map (crop to the image).
        const cv::Rect roi(left, top, W, H);
        const cv::Mat scores = cv::Mat(Hp, Wp, CV_32F, const_cast<float*>(sm))(roi).clone();

        // DKD (top-k mode). simple_nms with radius 2 (5x5).
        const cv::Mat kernel = cv::Mat::ones(2 * kRadius + 1, 2 * kRadius + 1, CV_8U);
        auto isLocalMax = [&](const cv::Mat& s) {
            cv::Mat m;
            cv::dilate(s, m, kernel);
            return s == m;  // CV_8U 0 / 255
        };
        cv::Mat max_mask = isLocalMax(scores);
        for (int it = 0; it < 2; ++it) {
            cv::Mat supp;
            cv::dilate(max_mask, supp, kernel);  // > 0 where suppressed
            cv::Mat supp_scores = scores.clone();
            supp_scores.setTo(0, supp);
            cv::Mat not_supp, new_max_kept;
            cv::bitwise_not(supp, not_supp);
            cv::bitwise_and(isLocalMax(supp_scores), not_supp, new_max_kept);
            cv::bitwise_or(max_mask, new_max_kept, max_mask);
        }
        cv::Mat nms = cv::Mat::zeros(H, W, CV_32F);
        scores.copyTo(nms, max_mask);
        // Remove the borders (radius).
        nms.rowRange(0, kRadius).setTo(0);
        nms.rowRange(H - kRadius, H).setTo(0);
        nms.colRange(0, kRadius).setTo(0);
        nms.colRange(W - kRadius, W).setTo(0);

        // Top-k of the NMS scores (only positive ones: with fewer peaks than k
        // the reference would also return arbitrary zero-score pixels).
        struct Candidate { float score; int x, y; };
        std::vector<Candidate> candidates;
        for (int y = 0; y < H; ++y) {
            const float* row = nms.ptr<float>(y);
            for (int x = 0; x < W; ++x) if (row[x] > 0) candidates.push_back({row[x], x, y});
        }
        const size_t n = std::min(candidates.size(), static_cast<size_t>(top_k_));
        std::partial_sort(candidates.begin(), candidates.begin() + n, candidates.end(),
                          [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

        // Sub-pixel: soft-argmax (temperature 0.1) of the scores in the 5x5
        // patch, then the score bilinearly interpolated there.
        std::vector<float> kpts_xy(2 * n);
        keypoints.clear();
        keypoints.reserve(n);
        const float* s = scores.ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            const Candidate& c = candidates[i];
            float max_v = -std::numeric_limits<float>::infinity();
            for (int dy = -kRadius; dy <= kRadius; ++dy)
                for (int dx = -kRadius; dx <= kRadius; ++dx)
                    max_v = std::max(max_v, planeAt(s, W, H, c.x + dx, c.y + dy));
            double sum = 0, rx = 0, ry = 0;
            for (int dy = -kRadius; dy <= kRadius; ++dy) {
                for (int dx = -kRadius; dx <= kRadius; ++dx) {
                    const double e = std::exp((planeAt(s, W, H, c.x + dx, c.y + dy) - max_v) / kTemperature);
                    sum += e;
                    rx += e * dx;
                    ry += e * dy;
                }
            }
            const float x = static_cast<float>(c.x + rx / sum), y = static_cast<float>(c.y + ry / sum);
            kpts_xy[2 * i] = x;
            kpts_xy[2 * i + 1] = y;
            keypoints.emplace_back(cv::Point2f(x, y), 8.f, -1.f, sampleBilinear(s, W, H, x, y));
        }

        // SDDH descriptors (second model).
        cv::Mat desc(static_cast<int>(n), kDim, CV_32F);
        if (n > 0) {
            std::vector<Ort::Value> desc_inputs;
            desc_inputs.push_back(std::move(dense[0]));  // padded feature map, on the device
            desc_inputs.push_back(OrtModel::tensor(crop, {4}));
            desc_inputs.push_back(OrtModel::tensor(kpts_xy, {static_cast<int64_t>(n), 2}));
            std::vector<Ort::Value> out =
                desc_->run({"feature_map", "crop", "keypoints"}, desc_inputs, {"descriptors"});
            std::memcpy(desc.data, out[0].GetTensorData<float>(), n * kDim * sizeof(float));
        }
        desc.copyTo(descriptors);
    }

private:
    static constexpr int kDim = 128;
    static constexpr int kRadius = 2;
    static constexpr double kTemperature = 0.1;
    std::shared_ptr<OrtModel> dense_, desc_;
    int top_k_;
};
