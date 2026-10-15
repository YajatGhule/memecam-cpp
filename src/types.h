// Core tracking types shared by every module (mirrors memecam/tracking.py: Hand, Face, Frame).
#pragma once

#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace memecam {

using Vec3 = cv::Vec3f;          // pixel x, y and relative z
using Box = cv::Rect;            // x0, y0, w, h  (Python bbox = (x0, y0, x1, y1) -> Box(x0, y0, x1-x0, y1-y0))

inline float dist2d(const Vec3& a, const Vec3& b) { return std::hypot(a[0] - b[0], a[1] - b[1]); }
inline cv::Point2f xy(const Vec3& p) { return {p[0], p[1]}; }

Box bbox_of(const std::vector<Vec3>& pts);

struct Hand {
    std::array<Vec3, 21> pts{};
    std::string label;           // "Left" / "Right" (user's hand, frame is mirrored)
    std::string gesture = "None";// MediaPipe canned gesture, e.g. "Victory", "Open_Palm"
    float gesture_score = 0.f;

    Box bbox() const;
    float size() const { return dist2d(pts[9], pts[0]) + 1e-6f; }   // wrist -> middle MCP
};

struct Face {
    std::vector<Vec3> pts;       // 478 face-mesh points, pixels
    Box bbox() const { return bbox_of(pts); }
    float width() const { return dist2d(pts[234], pts[454]) + 1e-6f; }
};

struct PoseLm { float x, y, z, visibility; };
using Pose = std::array<PoseLm, 33>;   // pixels + visibility (0..1)

constexpr int MOTION_W = 160, MOTION_H = 90;

// One person's tracking result. The top-level Frame returned by the tracker mirrors its primary
// (largest) person in hands/face/pose and lists everyone in `people`.
struct Frame {
    std::vector<Hand> hands;
    std::optional<Face> face;
    std::shared_ptr<const Pose> pose;          // shared: the async pose result is reused across frames
    std::shared_ptr<const cv::Mat> motion;     // CV_8U MOTION_H x MOTION_W, nonzero = changed (shared)
    int width = 0, height = 0;
    int pid = 0;                               // persistent person id; 0 = not assigned
    std::vector<Frame> people;                 // top-level frame only
    std::map<int, int> renamed;                // ids renumbered this frame, old -> new (top-level only)
};

}  // namespace memecam
