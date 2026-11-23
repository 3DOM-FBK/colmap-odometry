// Minimal visual odometry with ORB features (OpenCV + PoseLib), stereo or mono.
//
// Stereo (cam0/ and cam1/ found), for every pair of consecutive epochs (k-1, k):
//   1. Stereo: match ORB features between cam0 and cam1 at epoch k-1 and
//      triangulate them -> 3D points in the cam0 frame of epoch k-1.
//   2. Temporal: match those points' cam0 descriptors to cam0 at epoch k.
//   3. Resection: RANSAC PnP (OpenCV or PoseLib) gives the pose of cam0 at
//      epoch k with respect to cam0 at epoch k-1. Chaining these gives the
//      trajectory.
//
// Mono (only cam0/ found), for every epoch n:
//   1. Match cam0 at n-2 and n-1 and triangulate with their known poses
//      -> 3D points in the world frame.
//   2. Resection of cam0 at n on those points -> pose at n in the world frame.
//   Start-up (epoch 1) and recovery after a failure use the essential matrix
//   between n-1 and n. One camera cannot observe scale: the first translation
//   is set to length 1, so the trajectory is in arbitrary units.
//
// Usage:
//   ./orb_matching <dataset_dir> <camera_calibration.yaml> [camera_rig.yaml] [delay_ms]
//                  [--no-gui] [--scale S] [--sequential | --single-core]
//                  [--pnp opencv|poselib] [--report]
//
//   --report      print a timing table per processing stage at the end
//   --pnp         resection solver: opencv (solvePnPRansac, default) or
//                 poselib (P3P in LO-RANSAC + non-linear refinement)
//   delay_ms      wait between frames in the GUI (default 0 = no wait)
//   --no-gui      no windows, only terminal output (can be placed anywhere)
//   --scale S     resize images by S before ORB, e.g. 0.5 = half size (default 1)
//   --sequential  cam0 and cam1 one after the other, OpenCV internal threads on
//   --single-core cam0 and cam1 one after the other, OpenCV internal threads off
//   (default: cam0 and cam1 in parallel, OpenCV internal threads on)
//   dataset_dir   folder with cam0/ (and optionally cam1/) images with matching filenames
//   calibration   per-camera PINHOLE intrinsics: params = [fx, fy, cx, cy]
//   rig           (stereo only) ROS-style pose of each camera in the rig (cam0 = identity):
//                 p_cam0 = R * p_cam1 + t, quaternion [x, y, z, w]
//
// Keys: q / ESC = quit, space = pause / resume, any key while paused = next frame

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <PoseLib/poselib.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

enum class PnPSolver { OpenCV, PoseLib };

// ---------------------------------------------------------------- settings --
constexpr int kNumFeatures = 2000;
constexpr float kRatioTest = 0.8f;        // Lowe's ratio test
constexpr double kMaxReprojErrorPx = 2.0; // triangulation, PnP and essential matrix threshold
constexpr double kMinDepth = 0.5;         // metres (stereo only)
constexpr double kMaxDepth = 80.0;        // metres (stereo only)
constexpr int kMinInliers = 15;           // for PnP and the essential matrix

// ------------------------------------------------------------------ timing --
// Accumulated wall-clock time per processing stage (printed with --report).
// Thread-safe, because cam0 and cam1 can be processed on two threads.
class Timings {
public:
    // Fix the row order of the report (stages not declared are appended).
    void declare(const std::vector<std::string>& stages) {
        for (const auto& s : stages) {
            if (stats_.find(s) == stats_.end()) { order_.push_back(s); stats_[s]; }
        }
    }

    void add(const std::string& stage, double ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stats_.find(stage) == stats_.end()) order_.push_back(stage);
        stats_[stage].total_ms += ms;
        stats_[stage].calls += 1;
    }

    void print(const std::string& reference_stage) const {
        const double reference_ms = stats_.count(reference_stage)
                                        ? stats_.at(reference_stage).total_ms : 0.0;
        std::cout << "\nTiming report\n"
                  << cv::format("%-30s %7s %11s %11s %8s\n", "stage", "calls", "mean [ms]",
                                "total [s]", "share");
        for (const auto& stage : order_) {
            const Stat& s = stats_.at(stage);
            if (s.calls == 0) continue;
            const double share = reference_ms > 0 ? 100.0 * s.total_ms / reference_ms : 0.0;
            std::cout << cv::format("%-30s %7d %11.2f %11.2f %7.1f%%\n", stage.c_str(), s.calls,
                                    s.total_ms / s.calls, s.total_ms / 1000.0, share);
        }
        std::cout << "share = stage total / '" << reference_stage << "' total. In parallel mode\n"
                  << "the per-image stages of cam0 and cam1 overlap, so shares can add up\n"
                  << "to more than 100%.\n";
    }

private:
    struct Stat { double total_ms = 0.0; int calls = 0; };
    std::map<std::string, Stat> stats_;
    std::vector<std::string> order_;  // stages in the order they first appear
    mutable std::mutex mutex_;
};

// Measures the time from construction to the end of the scope.
class ScopedTimer {
public:
    ScopedTimer(Timings& timings, std::string stage)
        : timings_(timings), stage_(std::move(stage)) { meter_.start(); }
    ~ScopedTimer() { meter_.stop(); timings_.add(stage_, meter_.getTimeMilli()); }

private:
    Timings& timings_;
    std::string stage_;
    cv::TickMeter meter_;
};

// ------------------------------------------------------------- data types --
struct Calibration {
    cv::Matx33d K0, K1;
    cv::Matx34d P0, P1;  // projection matrices, cam0 frame = reference (P1 stereo only)
};

struct Frame {
    std::string name;
    cv::Mat img0, img1;  // img1 is empty in mono mode
    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat desc0, desc1;
};

struct TriangulatedPoints {
    std::vector<cv::Point3d> points;  // in the frame of the projection matrices used
    cv::Mat desc;                     // cam0 descriptor of each point (used for the next epoch)
    std::vector<cv::DMatch> matches;  // matches kept (query = first view, train = second view)
};

// Result of one epoch, plus what to draw.
struct StepResult {
    cv::Matx44d T_world_cam = cv::Matx44d::eye();  // pose of cam0: p_world = T * p_cam
    bool pose_updated = false;  // false = first frame, or failure with the pose kept
    std::string status = "first frame";
    size_t num_points = 0;      // 3D points triangulated in this step
    int num_inliers = 0;        // PnP or essential-matrix inliers
    // Matches between image a (left) and image b (right) for the GUI.
    cv::Mat img_a, img_b;
    std::vector<cv::KeyPoint> kp_a, kp_b;
    std::vector<cv::DMatch> matches;
};

// ------------------------------------------------------------- calibration --
// Intrinsics for images resized by `scale`. Pixel centres are at integer
// coordinates, so the principal point maps as (c + 0.5) * scale - 0.5.
cv::Matx33d readIntrinsics(const cv::FileNode& cam, double scale) {
    if (static_cast<std::string>(cam["model"]) != "PINHOLE") {
        throw std::runtime_error("Only the PINHOLE camera model is supported");
    }
    std::vector<double> p;
    cam["params"] >> p;  // fx, fy, cx, cy
    const double fx = p[0] * scale, fy = p[1] * scale;
    const double cx = (p[2] + 0.5) * scale - 0.5, cy = (p[3] + 0.5) * scale - 0.5;
    return {fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0};
}

cv::Matx33d quaternionToRotation(double x, double y, double z, double w) {
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    x /= n; y /= n; z /= n; w /= n;
    return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
            2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
}

// OpenCV only accepts YAML files whose first line is "%YAML:1.0". To read
// plain YAML files unchanged, prepend that header in memory.
cv::FileStorage openYaml(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    if (content.rfind("%YAML", 0) != 0) content = "%YAML:1.0\n" + content;
    return cv::FileStorage(content, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                                        cv::FileStorage::FORMAT_YAML);
}

// With an empty `rig_file` (mono), only cam0 is loaded.
Calibration loadCalibration(const std::string& calibration_file, const std::string& rig_file,
                            double scale) {
    cv::FileStorage calib = openYaml(calibration_file);
    Calibration c;
    c.K0 = readIntrinsics(calib["cam0"], scale);
    c.P0 = c.K0 * cv::Matx34d::eye();
    if (rig_file.empty()) return c;

    c.K1 = readIntrinsics(calib["cam1"], scale);

    // Rig file stores the pose of cam1 in the cam0 frame (T_cam0_cam1).
    cv::FileStorage rig = openYaml(rig_file);
    std::vector<double> q, t;
    rig["cam1"]["rotation"] >> q;     // x, y, z, w
    rig["cam1"]["translation"] >> t;
    const cv::Matx33d R_0_1 = quaternionToRotation(q[0], q[1], q[2], q[3]);
    const cv::Vec3d t_0_1(t[0], t[1], t[2]);

    // Projection needs the inverse: p_cam1 = R_0_1^T * (p_cam0 - t_0_1).
    const cv::Matx33d R_1_0 = R_0_1.t();
    const cv::Vec3d t_1_0 = -(R_1_0 * t_0_1);

    cv::Matx34d Rt1;
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) Rt1(r, col) = R_1_0(r, col);
        Rt1(r, 3) = t_1_0[r];
    }
    c.P1 = c.K1 * Rt1;

    std::cout << "Baseline: " << cv::norm(t_0_1) << " m\n";
    return c;
}

// ------------------------------------------------------------------ poses --
// 4x4 transform from a rotation and a translation.
cv::Matx44d makeTransform(const cv::Mat& R, const cv::Mat& t) {
    cv::Matx44d T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) T(r, col) = R.at<double>(r, col);
        T(r, 3) = t.at<double>(r);
    }
    return T;
}

// Camera centre in the world frame, from p_cam = T_cam_world * p_world.
cv::Vec3d cameraCenter(const cv::Matx44d& T_cam_world) {
    const cv::Matx44d T = T_cam_world.inv();
    return {T(0, 3), T(1, 3), T(2, 3)};
}

// ---------------------------------------------------------------- features --
std::vector<std::string> listImages(const fs::path& folder) {
    std::vector<std::string> names;
    if (!fs::is_directory(folder)) return names;
    for (const auto& entry : fs::directory_iterator(folder)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            names.push_back(entry.path().filename().string());
        }
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::vector<cv::DMatch> matchDescriptors(const cv::Mat& desc_a, const cv::Mat& desc_b) {
    std::vector<cv::DMatch> good;
    if (desc_a.empty() || desc_b.empty()) return good;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc_a, desc_b, knn, 2);
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < kRatioTest * m[1].distance) {
            good.push_back(m[0]);
        }
    }
    return good;
}

// Load, resize and extract ORB for one image.
void processImage(const fs::path& path, double scale, cv::ORB& orb, Timings& timings,
                  cv::Mat& img, std::vector<cv::KeyPoint>& kp, cv::Mat& desc) {
    {
        ScopedTimer t(timings, "image load + resize (per img)");
        img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (img.empty()) return;
        if (scale != 1.0) cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    ScopedTimer t(timings, "ORB extraction (per img)");
    orb.detectAndCompute(img, cv::noArray(), kp, desc);
}

// In mono mode only cam0 is loaded. Otherwise, with `parallel`, cam1 is
// processed on a second thread while this thread processes cam0. Each thread
// uses its own ORB detector (one detector is not thread-safe). Without it, the
// two images are processed one after the other.
bool loadFrame(const fs::path& dataset, const std::string& name, double scale, bool mono,
               bool parallel, cv::ORB& orb0, cv::ORB& orb1, Timings& timings, Frame& f) {
    ScopedTimer t(timings, "load + ORB (all cameras)");
    f.name = name;
    auto processCam0 = [&] {
        processImage(dataset / "cam0" / name, scale, orb0, timings, f.img0, f.kp0, f.desc0);
    };
    auto processCam1 = [&] {
        processImage(dataset / "cam1" / name, scale, orb1, timings, f.img1, f.kp1, f.desc1);
    };
    if (mono) {
        processCam0();
        return !f.img0.empty();
    }
    if (parallel) {
        auto cam1 = std::async(std::launch::async, processCam1);
        processCam0();
        cam1.get();  // wait for cam1 (also re-throws any error from that thread)
    } else {
        processCam0();
        processCam1();
    }
    return !f.img0.empty() && !f.img1.empty();
}

// ------------------------------------------------------------ triangulation --
double reprojectionError(const cv::Matx34d& P, const cv::Vec4d& X, const cv::Point2d& uv) {
    const cv::Vec3d x = P * X;
    return std::hypot(x[0] / x[2] - uv.x, x[1] / x[2] - uv.y);
}

// Triangulates matches (query = view a, train = view b) seen by the projection
// matrices Pa and Pb (expressed in the same frame). Keeps the points in front of
// both views, with depth in view a within [min_depth, max_depth], and consistent
// with both observations. `kept` receives the matches of the points returned.
std::vector<cv::Point3d> triangulateMatches(const cv::Matx34d& Pa, const cv::Matx34d& Pb,
                                            const std::vector<cv::KeyPoint>& kp_a,
                                            const std::vector<cv::KeyPoint>& kp_b,
                                            const std::vector<cv::DMatch>& matches,
                                            double min_depth, double max_depth,
                                            std::vector<cv::DMatch>& kept) {
    std::vector<cv::Point3d> points;
    if (matches.empty()) return points;

    std::vector<cv::Point2d> uv_a, uv_b;
    for (const auto& m : matches) {
        uv_a.push_back(kp_a[m.queryIdx].pt);
        uv_b.push_back(kp_b[m.trainIdx].pt);
    }
    cv::Mat X4;
    cv::triangulatePoints(cv::Mat(Pa), cv::Mat(Pb), uv_a, uv_b, X4);
    X4.convertTo(X4, CV_64F);

    for (size_t i = 0; i < matches.size(); ++i) {
        const double w = X4.at<double>(3, i);
        if (std::abs(w) < 1e-12) continue;
        const cv::Vec4d X(X4.at<double>(0, i) / w, X4.at<double>(1, i) / w,
                          X4.at<double>(2, i) / w, 1.0);

        // P = K [R | t] and the last row of K is [0 0 1], so (P X)_z is the depth.
        const double depth_a = (Pa * X)[2];
        const double depth_b = (Pb * X)[2];
        if (depth_a < min_depth || depth_a > max_depth || depth_b <= 0) continue;
        if (reprojectionError(Pa, X, uv_a[i]) > kMaxReprojErrorPx) continue;
        if (reprojectionError(Pb, X, uv_b[i]) > kMaxReprojErrorPx) continue;

        points.emplace_back(X[0], X[1], X[2]);
        kept.push_back(matches[i]);
    }
    return points;
}

// Stereo: cam0 and cam1 of one epoch -> points in the cam0 frame.
TriangulatedPoints triangulateStereo(const Frame& f, const Calibration& c, Timings& timings) {
    TriangulatedPoints out;
    std::vector<cv::DMatch> matches;
    {
        ScopedTimer t(timings, "stereo matching");
        matches = matchDescriptors(f.desc0, f.desc1);
    }
    ScopedTimer t(timings, "triangulation + filtering");
    out.points = triangulateMatches(c.P0, c.P1, f.kp0, f.kp1, matches, kMinDepth, kMaxDepth,
                                    out.matches);
    for (const auto& m : out.matches) out.desc.push_back(f.desc0.row(m.queryIdx));
    return out;
}

// Mono: cam0 at two epochs with known poses (p_cam = T_cam_world * p_world)
// -> points in the world frame. Descriptors are taken from view b (the newer).
TriangulatedPoints triangulateTwoViews(const Frame& a, const cv::Matx44d& T_a_world,
                                       const Frame& b, const cv::Matx44d& T_b_world,
                                       const cv::Matx33d& K, Timings& timings) {
    TriangulatedPoints out;
    std::vector<cv::DMatch> matches;
    {
        ScopedTimer t(timings, "two-view matching (mono)");
        matches = matchDescriptors(a.desc0, b.desc0);
    }
    ScopedTimer t(timings, "triangulation + filtering");
    const cv::Matx34d Pa = K * T_a_world.get_minor<3, 4>(0, 0);
    const cv::Matx34d Pb = K * T_b_world.get_minor<3, 4>(0, 0);
    // No metric depth range in mono: the scale is arbitrary.
    out.points = triangulateMatches(Pa, Pb, a.kp0, b.kp0, matches, 0.0,
                                    std::numeric_limits<double>::infinity(), out.matches);
    for (const auto& m : out.matches) out.desc.push_back(b.desc0.row(m.trainIdx));
    return out;
}

// --------------------------------------------------------------- resection --
// Both solvers return T with p_camera = T * p_points (points' frame -> camera).

bool solvePnPOpenCV(const std::vector<cv::Point3d>& object, const std::vector<cv::Point2d>& image,
                    const cv::Matx33d& K, cv::Matx44d& T, int& num_inliers) {
    cv::Mat rvec, tvec, inliers;
    const bool ok = cv::solvePnPRansac(object, image, cv::Mat(K), cv::noArray(), rvec, tvec,
                                       false, 1000, kMaxReprojErrorPx, 0.999, inliers);
    num_inliers = inliers.rows;
    if (!ok) return false;

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    T = makeTransform(R, tvec);
    return true;
}

bool solvePnPPoseLib(const std::vector<cv::Point3d>& object, const std::vector<cv::Point2d>& image,
                     const cv::Matx33d& K, const cv::Size& image_size, cv::Matx44d& T,
                     int& num_inliers) {
    std::vector<Eigen::Vector3d> points3D;
    std::vector<Eigen::Vector2d> points2D;
    points3D.reserve(object.size());
    points2D.reserve(image.size());
    for (size_t i = 0; i < object.size(); ++i) {
        points3D.emplace_back(object[i].x, object[i].y, object[i].z);
        points2D.emplace_back(image[i].x, image[i].y);
    }

    const poselib::Camera camera("PINHOLE", {K(0, 0), K(1, 1), K(0, 2), K(1, 2)},
                                 image_size.width, image_size.height);
    poselib::RansacOptions ransac;
    ransac.max_reproj_error = kMaxReprojErrorPx;  // pixels
    ransac.max_iterations = 1000;
    ransac.success_prob = 0.999;
    poselib::BundleOptions refinement;            // robust non-linear refinement on inliers
    refinement.loss_scale = 0.5 * kMaxReprojErrorPx;

    poselib::CameraPose pose;
    std::vector<char> inliers;
    const poselib::RansacStats stats = poselib::estimate_absolute_pose(
        points2D, points3D, camera, ransac, refinement, &pose, &inliers);
    num_inliers = static_cast<int>(stats.num_inliers);
    if (num_inliers == 0) return false;

    const Eigen::Matrix3d R = pose.R();
    T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) T(r, col) = R(r, col);
        T(r, 3) = pose.t(r);
    }
    return true;
}

// Resection of cam0 of `curr` on the 3D points: p_curr = T * p_points.
// `matches_out` (optional) receives the temporal matches (query = point index,
// train = keypoint of cam0 in `curr`).
bool estimateMotion(const TriangulatedPoints& points, const Frame& curr, const cv::Matx33d& K0,
                    PnPSolver solver, Timings& timings, cv::Matx44d& T, int& num_inliers,
                    std::vector<cv::DMatch>* matches_out = nullptr) {
    std::vector<cv::Point3d> object;
    std::vector<cv::Point2d> image;
    {
        ScopedTimer t(timings, "temporal matching");
        const std::vector<cv::DMatch> matches = matchDescriptors(points.desc, curr.desc0);
        for (const auto& m : matches) {
            object.push_back(points.points[m.queryIdx]);
            image.push_back(curr.kp0[m.trainIdx].pt);
        }
        if (matches_out) *matches_out = matches;
    }
    num_inliers = 0;
    if (static_cast<int>(object.size()) < kMinInliers) return false;

    ScopedTimer t(timings, solver == PnPSolver::PoseLib ? "PnP RANSAC (poselib)"
                                                        : "PnP RANSAC (opencv)");
    const bool ok = (solver == PnPSolver::PoseLib)
        ? solvePnPPoseLib(object, image, K0, curr.img0.size(), T, num_inliers)
        : solvePnPOpenCV(object, image, K0, T, num_inliers);
    return ok && num_inliers >= kMinInliers;
}

// Relative pose of cam0 from frame a to frame b (p_b = T_b_a * p_a) with the
// essential matrix (5-point RANSAC). The translation has unit length.
// `matches` receives the matches used (query = a, train = b).
bool estimateEssential(const Frame& a, const Frame& b, const cv::Matx33d& K, Timings& timings,
                       cv::Matx44d& T_b_a, int& num_inliers, std::vector<cv::DMatch>& matches) {
    {
        ScopedTimer t(timings, "temporal matching");
        matches = matchDescriptors(a.desc0, b.desc0);
    }
    num_inliers = 0;
    if (static_cast<int>(matches.size()) < kMinInliers) return false;

    ScopedTimer t(timings, "essential matrix (mono init)");
    std::vector<cv::Point2d> uv_a, uv_b;
    for (const auto& m : matches) {
        uv_a.push_back(a.kp0[m.queryIdx].pt);
        uv_b.push_back(b.kp0[m.trainIdx].pt);
    }
    cv::Mat mask;
    const cv::Mat E = cv::findEssentialMat(uv_a, uv_b, cv::Mat(K), cv::RANSAC, 0.999,
                                           kMaxReprojErrorPx, mask);
    if (E.rows < 3) return false;  // can hold several 3x3 solutions: use the first

    cv::Mat R, tvec;
    num_inliers = cv::recoverPose(E.rowRange(0, 3), uv_a, uv_b, cv::Mat(K), R, tvec, mask);
    if (num_inliers < kMinInliers) return false;
    T_b_a = makeTransform(R, tvec);
    return true;
}

// --------------------------------------------------------------- odometry --
// Stereo: triangulate cam0 + cam1 at k-1, resection of cam0 at k.
class StereoOdometry {
public:
    StereoOdometry(const Calibration& calib, PnPSolver solver, Timings& timings)
        : calib_(calib), solver_(solver), timings_(timings) {}

    StepResult process(const Frame& f) {
        StepResult r;
        if (prev_points_) {
            cv::Matx44d T_curr_prev;
            if (estimateMotion(*prev_points_, f, calib_.K0, solver_, timings_, T_curr_prev,
                               r.num_inliers)) {
                T_world_cam_ = T_world_cam_ * T_curr_prev.inv();
                r.pose_updated = true;
                r.status = "ok";
            } else {
                r.status = "PnP failed, pose kept";
            }
        }

        // Triangulate this epoch for the next one.
        prev_points_ = triangulateStereo(f, calib_, timings_);

        r.T_world_cam = T_world_cam_;
        r.num_points = prev_points_->points.size();
        r.img_a = f.img0; r.kp_a = f.kp0;
        r.img_b = f.img1; r.kp_b = f.kp1;
        r.matches = prev_points_->matches;
        return r;
    }

private:
    const Calibration& calib_;
    PnPSolver solver_;
    Timings& timings_;
    cv::Matx44d T_world_cam_ = cv::Matx44d::eye();  // world = cam0 at the first epoch
    std::optional<TriangulatedPoints> prev_points_;
};

// Mono: triangulate cam0 at n-2 and n-1, resection of cam0 at n.
class MonoOdometry {
public:
    MonoOdometry(const cv::Matx33d& K, PnPSolver solver, Timings& timings)
        : K_(K), solver_(solver), timings_(timings) {}

    StepResult process(const Frame& f) {
        StepResult r;
        cv::Matx44d T_cam_world = cv::Matx44d::eye();  // world = cam0 at the first epoch
        r.img_a = f.img0; r.kp_a = f.kp0;

        if (prev1_) {
            T_cam_world = prev1_->T_cam_world;  // kept if everything below fails
            r.img_a = prev1_->frame.img0;
            r.kp_a = prev1_->frame.kp0;

            const bool ok = (prev2_ && resection(f, T_cam_world, r)) ||
                            initialise(f, T_cam_world, r);
            if (ok) {
                // Remember the step length to scale a later re-initialisation.
                const double step = cv::norm(cameraCenter(T_cam_world) -
                                             cameraCenter(prev1_->T_cam_world));
                if (step > 0) step_length_ = step;
                r.pose_updated = true;
            } else {
                r.status = "lost, pose kept";
            }
        }

        r.T_world_cam = T_cam_world.inv();
        r.img_b = f.img0; r.kp_b = f.kp0;

        // Shift the history: n-1 -> n-2, n -> n-1.
        prev2_ = std::move(prev1_);
        prev1_ = View{f, T_cam_world};
        return r;
    }

private:
    struct View {
        Frame frame;
        cv::Matx44d T_cam_world;  // p_cam = T_cam_world * p_world
    };

    // Triangulate n-2 and n-1, then resection of n on those world points.
    bool resection(const Frame& f, cv::Matx44d& T_cam_world, StepResult& r) {
        const TriangulatedPoints points = triangulateTwoViews(
            prev2_->frame, prev2_->T_cam_world, prev1_->frame, prev1_->T_cam_world, K_, timings_);
        r.num_points = points.points.size();

        cv::Matx44d T;
        std::vector<cv::DMatch> temporal;
        if (!estimateMotion(points, f, K_, solver_, timings_, T, r.num_inliers, &temporal)) {
            return false;
        }
        T_cam_world = T;
        r.status = "ok";
        // For drawing: keypoint of the point in n-1 -> keypoint in n.
        r.matches.clear();
        for (const auto& m : temporal) {
            r.matches.emplace_back(points.matches[m.queryIdx].trainIdx, m.trainIdx, m.distance);
        }
        return true;
    }

    // Relative pose n-1 -> n from the essential matrix, translation scaled by
    // the last known step length (1 at start-up).
    bool initialise(const Frame& f, cv::Matx44d& T_cam_world, StepResult& r) {
        cv::Matx44d T_curr_prev;
        if (!estimateEssential(prev1_->frame, f, K_, timings_, T_curr_prev, r.num_inliers,
                               r.matches)) {
            return false;
        }
        for (int i = 0; i < 3; ++i) T_curr_prev(i, 3) *= step_length_;
        T_cam_world = T_curr_prev * prev1_->T_cam_world;
        r.status = prev2_ ? "re-initialised (essential matrix)" : "initialised (essential matrix)";
        return true;
    }

    cv::Matx33d K_;
    PnPSolver solver_;
    Timings& timings_;
    std::optional<View> prev1_, prev2_;  // epochs n-1 and n-2
    double step_length_ = 1.0;           // sets the (arbitrary) scale of the trajectory
};

// ------------------------------------------------------------ visualization --
// Top view (x right, z forward) of the cam0 positions, auto-scaled to fit.
cv::Mat drawTrajectory(const std::vector<cv::Point2d>& xz, double fps, int size = 600) {
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(30, 30, 30));
    if (xz.empty()) return canvas;

    double min_x = xz[0].x, max_x = xz[0].x, min_z = xz[0].y, max_z = xz[0].y;
    for (const auto& p : xz) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_z = std::min(min_z, p.y); max_z = std::max(max_z, p.y);
    }
    const double span = std::max({max_x - min_x, max_z - min_z, 1.0});
    const double scale = 0.8 * size / span;
    const cv::Point2d center((min_x + max_x) / 2, (min_z + max_z) / 2);
    auto toPixel = [&](const cv::Point2d& p) {
        return cv::Point(static_cast<int>(size / 2 + (p.x - center.x) * scale),
                         static_cast<int>(size / 2 - (p.y - center.y) * scale));
    };

    for (size_t i = 1; i < xz.size(); ++i) {
        cv::line(canvas, toPixel(xz[i - 1]), toPixel(xz[i]), {0, 200, 0}, 2);
    }
    cv::circle(canvas, toPixel(xz.back()), 5, {0, 0, 255}, -1);
    cv::putText(canvas, "top view (x, z)  span " + cv::format("%.1f", span),
                {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
    cv::putText(canvas, cv::format("%.1f FPS", fps),
                {10, 50}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 1);
    return canvas;
}

// ---------------------------------------------------------------------- main --
int main(int argc, char** argv) {
    // Split the options from the positional arguments.
    bool gui = true;
    bool parallel_cameras = true;  // cam0 and cam1 on two threads
    bool opencv_threads = true;    // OpenCV's own internal multi-threading
    double scale = 1.0;
    std::string pnp = "opencv";
    bool report = false;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-gui") gui = false;
        else if (arg == "--report") report = true;
        else if (arg == "--sequential") parallel_cameras = false;
        else if (arg == "--single-core") parallel_cameras = opencv_threads = false;
        else if (arg == "--scale" && i + 1 < argc) scale = std::stod(argv[++i]);
        else if (arg == "--pnp" && i + 1 < argc) pnp = argv[++i];
        else args.push_back(arg);
    }
    const std::string usage =
        std::string("Usage: ") + argv[0] +
        " <dataset_dir> <camera_calibration.yaml> [camera_rig.yaml] [delay_ms]"
        " [--no-gui] [--scale S] [--sequential | --single-core]"
        " [--pnp opencv|poselib] [--report]\n";
    if (args.size() < 2 || scale <= 0.0 || (pnp != "opencv" && pnp != "poselib")) {
        std::cerr << usage;
        return 1;
    }

    // Remaining positional arguments: the rig file (.yaml / .yml) and delay_ms.
    std::string rig_file;
    int delay_ms = 0;  // 0 = no wait
    for (size_t i = 2; i < args.size(); ++i) {
        const std::string ext = fs::path(args[i]).extension().string();
        if (ext == ".yaml" || ext == ".yml") rig_file = args[i];
        else delay_ms = std::stoi(args[i]);
    }

    // Mono when there are no cam1 images.
    const fs::path dataset = args[0];
    const bool mono = listImages(dataset / "cam1").empty();
    if (!mono && rig_file.empty()) {
        std::cerr << "cam1/ found (stereo): camera_rig.yaml is required.\n" << usage;
        return 1;
    }

    const PnPSolver solver = (pnp == "poselib") ? PnPSolver::PoseLib : PnPSolver::OpenCV;
    if (!opencv_threads) cv::setNumThreads(1);

    const Calibration calib = loadCalibration(args[1], mono ? "" : rig_file, scale);
    const std::string mode = mono ? "mono (cam0 only, arbitrary scale)"
                             : parallel_cameras ? "stereo, parallel"
                             : opencv_threads ? "stereo, sequential"
                                              : "stereo, single-core";
    std::cout << "Image scale: " << scale << ", mode: " << mode << ", PnP: " << pnp << "\n";

    const std::vector<std::string> names = listImages(dataset / "cam0");
    if (names.empty()) {
        std::cerr << "No images found in " << (dataset / "cam0") << "\n";
        return 1;
    }

    // One ORB detector per camera, so both can run in parallel.
    cv::Ptr<cv::ORB> orb0 = cv::ORB::create(kNumFeatures);
    cv::Ptr<cv::ORB> orb1 = cv::ORB::create(kNumFeatures);
    const std::string matches_window = mono ? "temporal matches (cam0 n-1 | cam0 n)"
                                            : "stereo matches (cam0 | cam1)";
    if (gui) {
        cv::namedWindow(matches_window, cv::WINDOW_NORMAL);
        cv::namedWindow("trajectory", cv::WINDOW_AUTOSIZE);
    }

    Timings timings;
    timings.declare({"frame total", "load + ORB (all cameras)", "image load + resize (per img)",
                     "ORB extraction (per img)", "temporal matching", "PnP RANSAC (opencv)",
                     "PnP RANSAC (poselib)", "essential matrix (mono init)", "stereo matching",
                     "two-view matching (mono)", "triangulation + filtering", "visualization"});

    StereoOdometry stereo_vo(calib, solver, timings);
    MonoOdometry mono_vo(calib.K0, solver, timings);

    std::vector<cv::Point2d> trajectory{{0.0, 0.0}};
    bool paused = false;
    double fps = 0.0;  // smoothed processing rate (excludes the waitKey delay)

    for (const auto& name : names) {
        cv::TickMeter timer;
        timer.start();

        Frame frame;
        if (!loadFrame(dataset, name, scale, mono, parallel_cameras, *orb0, *orb1, timings,
                       frame)) {
            std::cerr << "Skipping " << name << " (image missing or unreadable)\n";
            continue;
        }

        const StepResult r = mono ? mono_vo.process(frame) : stereo_vo.process(frame);
        if (r.pose_updated) trajectory.emplace_back(r.T_world_cam(0, 3), r.T_world_cam(2, 3));

        // Draw the matches (GUI only).
        if (gui) {
            ScopedTimer t(timings, "visualization");
            cv::Mat canvas;
            cv::drawMatches(r.img_a, r.kp_a, r.img_b, r.kp_b, r.matches, canvas);
            cv::putText(canvas,
                        name + "  3D pts: " + std::to_string(r.num_points) +
                            "  inliers: " + std::to_string(r.num_inliers) + "  " + r.status,
                        {30, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.5, {0, 255, 255}, 3);
            cv::imshow(matches_window, canvas);
        }

        // Processing time of this frame, smoothed over the last frames.
        timer.stop();
        timings.add("frame total", timer.getTimeMilli());
        const double frame_fps = 1.0 / timer.getTimeSec();
        fps = (fps == 0.0) ? frame_fps : 0.9 * fps + 0.1 * frame_fps;

        const cv::Matx44d& T = r.T_world_cam;
        std::cout << name << ": " << r.num_points << " triangulated points, " << r.num_inliers
                  << " inliers, position [" << T(0, 3) << ", " << T(1, 3) << ", " << T(2, 3)
                  << "] (" << r.status << "), " << cv::format("%.1f FPS", fps) << "\n";

        if (!gui) continue;

        cv::imshow("trajectory", drawTrajectory(trajectory, fps));

        // When paused, any other key steps one frame.
        // delay_ms = 0 uses pollKey(): windows are updated without any wait.
        const int key = paused ? cv::waitKey(0)
                               : (delay_ms > 0 ? cv::waitKey(delay_ms) : cv::pollKey());
        if (key == ' ') paused = !paused;
        if (key == 'q' || key == 27) break;
    }

    if (report) timings.print("frame total");

    if (gui) {
        std::cout << "Done. Press any key in a window to close.\n";
        cv::waitKey(0);
    } else {
        std::cout << "Done.\n";
    }
    return 0;
}
