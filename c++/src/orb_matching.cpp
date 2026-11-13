// Minimal stereo ORB matching viewer.
//
// Reads image pairs with the same filename from <dataset>/cam0 and
// <dataset>/cam1, detects ORB features, matches them and shows the matches.
//
// Usage:   ./orb_matching <dataset_dir> [delay_ms]
// Keys:    q / ESC = quit, space = pause / resume, any key while paused = next frame

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Sorted list of image filenames (not full paths) inside a folder.
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

// Match descriptors and keep only distinctive matches (Lowe's ratio test).
std::vector<cv::DMatch> matchDescriptors(const cv::Mat& desc0, const cv::Mat& desc1,
                                         float ratio = 0.8f) {
    std::vector<cv::DMatch> good;
    if (desc0.empty() || desc1.empty()) return good;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc0, desc1, knn, 2);

    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < ratio * m[1].distance) {
            good.push_back(m[0]);
        }
    }
    return good;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_dir> [delay_ms]\n";
        return 1;
    }
    const fs::path dataset = argv[1];
    const int delay_ms = (argc > 2) ? std::stoi(argv[2]) : 30;

    const std::vector<std::string> names = listImages(dataset / "cam0");
    if (names.empty()) {
        std::cerr << "No images found in " << (dataset / "cam0") << "\n";
        return 1;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    const std::string window = "ORB matches (cam0 | cam1)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);

    bool paused = false;
    for (const auto& name : names) {
        cv::Mat img0 = cv::imread((dataset / "cam0" / name).string(), cv::IMREAD_GRAYSCALE);
        cv::Mat img1 = cv::imread((dataset / "cam1" / name).string(), cv::IMREAD_GRAYSCALE);
        if (img0.empty() || img1.empty()) {
            std::cerr << "Skipping " << name << " (missing cam0 or cam1 image)\n";
            continue;
        }

        // 1. Detect keypoints and compute descriptors
        std::vector<cv::KeyPoint> kp0, kp1;
        cv::Mat desc0, desc1;
        orb->detectAndCompute(img0, cv::noArray(), kp0, desc0);
        orb->detectAndCompute(img1, cv::noArray(), kp1, desc1);

        // 2. Match
        const std::vector<cv::DMatch> matches = matchDescriptors(desc0, desc1);
        std::cout << name << ": " << kp0.size() << " / " << kp1.size()
                  << " keypoints, " << matches.size() << " matches\n";

        // 3. Visualize
        cv::Mat canvas;
        cv::drawMatches(img0, kp0, img1, kp1, matches, canvas);
        cv::putText(canvas, name + "  matches: " + std::to_string(matches.size()),
                    {30, 60}, cv::FONT_HERSHEY_SIMPLEX, 1.5, {0, 255, 255}, 3);
        cv::imshow(window, canvas);

        // 4. Handle keys. When paused, any other key steps one frame.
        const int key = cv::waitKey(paused ? 0 : delay_ms);
        if (key == ' ') paused = !paused;
        if (key == 'q' || key == 27) break;
    }

    std::cout << "Done. Press any key in the window to close.\n";
    cv::waitKey(0);
    return 0;
}
