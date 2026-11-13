// Minimal stereo visual odometry with ORB features (OpenCV only).
//
// For every pair of consecutive epochs (k-1, k):
//   1. Stereo: match ORB features between cam0 and cam1 at epoch k-1 and
//      triangulate them -> 3D points in the cam0 frame of epoch k-1.
//   2. Temporal: match those points' cam0 descriptors to cam0 at epoch k.
//   3. Resection: solvePnPRansac gives the pose of cam0 at epoch k with
//      respect to cam0 at epoch k-1. Chaining these gives the trajectory.
//
// Usage:
//   ./orb_matching <dataset_dir> <camera_calibration.yaml> <camera_rig.yaml> [delay_ms]
//
//   dataset_dir   folder with cam0/ and cam1/ images with matching filenames
//   calibration   per-camera PINHOLE intrinsics: params = [fx, fy, cx, cy]
//   rig           ROS-style pose of each camera in the rig (cam0 = identity):
//                 p_cam0 = R * p_cam1 + t, quaternion [x, y, z, w]
//
// Keys: q / ESC = quit, space = pause / resume, any key while paused = next frame

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ---------------------------------------------------------------- settings --
constexpr int kNumFeatures = 2000;
constexpr float kRatioTest = 0.8f;        // Lowe's ratio test
constexpr double kMaxReprojErrorPx = 2.0; // triangulation and PnP threshold
constexpr double kMinDepth = 0.5;         // metres
constexpr double kMaxDepth = 80.0;        // metres
constexpr int kMinPnPInliers = 15;

// ------------------------------------------------------------- data types --
struct StereoCalibration {
    cv::Matx33d K0, K1;
    cv::Matx34d P0, P1;  // projection matrices, cam0 frame = reference
};

struct Frame {
    std::string name;
    cv::Mat img0, img1;
    std::vector<cv::KeyPoint> kp0, kp1;
    cv::Mat desc0, desc1;
};

struct StereoPoints {
    std::vector<cv::Point3d> points;  // in the cam0 frame
    cv::Mat desc0;                    // cam0 descriptor of each point
    std::vector<cv::DMatch> matches;  // stereo matches kept (for drawing)
};

// ------------------------------------------------------------- calibration --
cv::Matx33d readIntrinsics(const cv::FileNode& cam) {
    if (static_cast<std::string>(cam["model"]) != "PINHOLE") {
        throw std::runtime_error("Only the PINHOLE camera model is supported");
    }
    std::vector<double> p;
    cam["params"] >> p;  // fx, fy, cx, cy
    return {p[0], 0.0, p[2],
            0.0, p[1], p[3],
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

StereoCalibration loadCalibration(const std::string& calibration_file,
                                  const std::string& rig_file) {
    cv::FileStorage calib = openYaml(calibration_file);
    cv::FileStorage rig = openYaml(rig_file);

    StereoCalibration c;
    c.K0 = readIntrinsics(calib["cam0"]);
    c.K1 = readIntrinsics(calib["cam1"]);

    // Rig file stores the pose of cam1 in the cam0 frame (T_cam0_cam1).
    std::vector<double> q, t;
    rig["cam1"]["rotation"] >> q;     // x, y, z, w
    rig["cam1"]["translation"] >> t;
    const cv::Matx33d R_0_1 = quaternionToRotation(q[0], q[1], q[2], q[3]);
    const cv::Vec3d t_0_1(t[0], t[1], t[2]);

    // Projection needs the inverse: p_cam1 = R_0_1^T * (p_cam0 - t_0_1).
    const cv::Matx33d R_1_0 = R_0_1.t();
    const cv::Vec3d t_1_0 = -(R_1_0 * t_0_1);

    c.P0 = c.K0 * cv::Matx34d::eye();
    cv::Matx34d Rt1;
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) Rt1(r, col) = R_1_0(r, col);
        Rt1(r, 3) = t_1_0[r];
    }
    c.P1 = c.K1 * Rt1;

    std::cout << "Baseline: " << cv::norm(t_0_1) << " m\n";
    return c;
}

// ---------------------------------------------------------------- features --
std::vector<std::string> listImages(const fs::path& folder) {
    std::vector<std::string> names;
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

bool loadFrame(const fs::path& dataset, const std::string& name, cv::ORB& orb, Frame& f) {
    f.name = name;
    f.img0 = cv::imread((dataset / "cam0" / name).string(), cv::IMREAD_GRAYSCALE);
    f.img1 = cv::imread((dataset / "cam1" / name).string(), cv::IMREAD_GRAYSCALE);
    if (f.img0.empty() || f.img1.empty()) return false;
    orb.detectAndCompute(f.img0, cv::noArray(), f.kp0, f.desc0);
    orb.detectAndCompute(f.img1, cv::noArray(), f.kp1, f.desc1);
    return true;
}

// ------------------------------------------------------------ triangulation --
double reprojectionError(const cv::Matx34d& P, const cv::Vec4d& X, const cv::Point2f& uv) {
    const cv::Vec3d x = P * X;
    return std::hypot(x[0] / x[2] - uv.x, x[1] / x[2] - uv.y);
}

StereoPoints triangulateStereo(const Frame& f, const StereoCalibration& c) {
    StereoPoints out;
    const std::vector<cv::DMatch> matches = matchDescriptors(f.desc0, f.desc1);
    if (matches.empty()) return out;

    std::vector<cv::Point2d> uv0, uv1;
    for (const auto& m : matches) {
        uv0.push_back(f.kp0[m.queryIdx].pt);
        uv1.push_back(f.kp1[m.trainIdx].pt);
    }
    cv::Mat X4;
    cv::triangulatePoints(cv::Mat(c.P0), cv::Mat(c.P1), uv0, uv1, X4);
    X4.convertTo(X4, CV_64F);

    for (size_t i = 0; i < matches.size(); ++i) {
        const double w = X4.at<double>(3, i);
        if (std::abs(w) < 1e-12) continue;
        const cv::Vec4d X(X4.at<double>(0, i) / w, X4.at<double>(1, i) / w,
                          X4.at<double>(2, i) / w, 1.0);

        // Keep points in front of both cameras, within the depth range,
        // and consistent with both observations.
        const double depth0 = X[2];
        const double depth1 = (c.P1 * X)[2];  // positive scale of cam1 depth
        if (depth0 < kMinDepth || depth0 > kMaxDepth || depth1 <= 0) continue;
        if (reprojectionError(c.P0, X, f.kp0[matches[i].queryIdx].pt) > kMaxReprojErrorPx) continue;
        if (reprojectionError(c.P1, X, f.kp1[matches[i].trainIdx].pt) > kMaxReprojErrorPx) continue;

        out.points.emplace_back(X[0], X[1], X[2]);
        out.desc0.push_back(f.desc0.row(matches[i].queryIdx));
        out.matches.push_back(matches[i]);
    }
    return out;
}

// --------------------------------------------------------------- resection --
// Pose of cam0 at epoch k w.r.t. cam0 at epoch k-1 (p_k = T * p_{k-1}).
bool estimateMotion(const StereoPoints& prev, const Frame& curr, const cv::Matx33d& K0,
                    cv::Matx44d& T_curr_prev, int& num_inliers) {
    const std::vector<cv::DMatch> matches = matchDescriptors(prev.desc0, curr.desc0);
    std::vector<cv::Point3d> object;
    std::vector<cv::Point2d> image;
    for (const auto& m : matches) {
        object.push_back(prev.points[m.queryIdx]);
        image.push_back(curr.kp0[m.trainIdx].pt);
    }
    num_inliers = 0;
    if (static_cast<int>(object.size()) < kMinPnPInliers) return false;

    cv::Mat rvec, tvec, inliers;
    const bool ok = cv::solvePnPRansac(object, image, cv::Mat(K0), cv::noArray(), rvec, tvec,
                                       false, 1000, kMaxReprojErrorPx, 0.999, inliers);
    num_inliers = inliers.rows;
    if (!ok || num_inliers < kMinPnPInliers) return false;

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    T_curr_prev = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) T_curr_prev(r, col) = R.at<double>(r, col);
        T_curr_prev(r, 3) = tvec.at<double>(r);
    }
    return true;
}

// ------------------------------------------------------------ visualization --
// Top view (x right, z forward) of the cam0 positions, auto-scaled to fit.
cv::Mat drawTrajectory(const std::vector<cv::Point2d>& xz, int size = 600) {
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
    cv::putText(canvas, "top view (x, z)  span " + cv::format("%.1f m", span),
                {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
    return canvas;
}

// ---------------------------------------------------------------------- main --
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_dir> <camera_calibration.yaml> <camera_rig.yaml> [delay_ms]\n";
        return 1;
    }
    const fs::path dataset = argv[1];
    const StereoCalibration calib = loadCalibration(argv[2], argv[3]);
    const int delay_ms = (argc > 4) ? std::stoi(argv[4]) : 30;

    const std::vector<std::string> names = listImages(dataset / "cam0");
    if (names.empty()) {
        std::cerr << "No images found in " << (dataset / "cam0") << "\n";
        return 1;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(kNumFeatures);
    cv::namedWindow("stereo matches (cam0 | cam1)", cv::WINDOW_NORMAL);
    cv::namedWindow("trajectory", cv::WINDOW_AUTOSIZE);

    cv::Matx44d T_world_cam = cv::Matx44d::eye();  // world = cam0 at the first epoch
    std::vector<cv::Point2d> trajectory{{0.0, 0.0}};
    StereoPoints prev_points;
    bool has_prev = false;
    bool paused = false;

    for (const auto& name : names) {
        Frame frame;
        if (!loadFrame(dataset, name, *orb, frame)) {
            std::cerr << "Skipping " << name << " (missing cam0 or cam1 image)\n";
            continue;
        }

        // Resection of cam0 at this epoch on the points of the previous epoch.
        int inliers = 0;
        std::string status = "first frame";
        if (has_prev) {
            cv::Matx44d T_curr_prev;
            if (estimateMotion(prev_points, frame, calib.K0, T_curr_prev, inliers)) {
                T_world_cam = T_world_cam * T_curr_prev.inv();
                trajectory.emplace_back(T_world_cam(0, 3), T_world_cam(2, 3));
                status = "ok";
            } else {
                status = "PnP failed, pose kept";
            }
        }

        // Triangulate this epoch for the next one.
        prev_points = triangulateStereo(frame, calib);
        has_prev = true;

        const cv::Vec3d pos(T_world_cam(0, 3), T_world_cam(1, 3), T_world_cam(2, 3));
        std::cout << name << ": " << prev_points.points.size() << " stereo points, "
                  << inliers << " PnP inliers, position [" << pos[0] << ", " << pos[1]
                  << ", " << pos[2] << "] (" << status << ")\n";

        // Visualize stereo matches and trajectory.
        cv::Mat canvas;
        cv::drawMatches(frame.img0, frame.kp0, frame.img1, frame.kp1, prev_points.matches, canvas);
        cv::putText(canvas,
                    name + "  stereo pts: " + std::to_string(prev_points.points.size()) +
                        "  PnP inliers: " + std::to_string(inliers) + "  " + status,
                    {30, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.5, {0, 255, 255}, 3);
        cv::imshow("stereo matches (cam0 | cam1)", canvas);
        cv::imshow("trajectory", drawTrajectory(trajectory));

        // When paused, any other key steps one frame.
        const int key = cv::waitKey(paused ? 0 : delay_ms);
        if (key == ' ') paused = !paused;
        if (key == 'q' || key == 27) break;
    }

    std::cout << "Done. Press any key in a window to close.\n";
    cv::waitKey(0);
    return 0;
}
