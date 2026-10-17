// MediaPipe Tasks VIDEO-mode graphs, one class per task: GestureRecognizer (palm detector -> hand landmarks ->
// gesture embedder/classifier), FaceLandmarker (BlazeFace -> face mesh), PoseLandmarker (pose detector ->
// pose landmarks). Each *Tracker holds the per-stream state (ROIs from the previous frame, smoothing filters)
// and borrows the shared, immutable-after-load models, so two trackers with different instance counts can
// share one set of interpreters (as long as they are not run at the same time).
#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision/geometry.h"
#include "vision/smoothing.h"
#include "vision/tflite_model.h"

namespace memecam::vision {

// Accumulated wall time per stage (ms) and call counts.
struct StageStats {
    std::atomic<double> det_ms{0}, lm_ms{0}, extra_ms{0}, total_ms{0};
    std::atomic<long> det_runs{0}, lm_runs{0}, calls{0};
    void add(std::atomic<double>& a, double v) { a.fetch_add(v, std::memory_order_relaxed); }
};

// ---- hands ---------------------------------------------------------------------------------------
struct HandModels {
    HandModels(const std::string& dir, int threads);
    TfliteModel palm, landmarks, embedder, classifier;
    std::vector<Anchor> anchors;
    std::vector<std::string> labels;
    StageStats stats;   // det = palm, lm = landmarks, extra = gesture
};

struct HandResult {
    Landmarks lms;            // 21, normalized image coords (z relative, scaled like x)
    Landmarks world;          // 21, metres, rotated into the image frame
    float handedness = 0;     // raw model score (P(right))
    std::string label;        // "Right" / "Left"
    std::string gesture = "None";
    float gesture_score = 0;
};

class HandTracker {
public:
    HandTracker(HandModels& m, int num_hands) : m_(m), num_(num_hands) {}
    std::vector<HandResult> process(const cv::Mat& rgb, int64_t ts_ms);
    int num_hands() const { return num_; }
private:
    std::vector<NormRect> detect(const cv::Mat& rgb);
    void classify(HandResult& h, int W, int H);
    HandModels& m_;
    int num_;
    std::vector<NormRect> prev_;
};

// ---- face ----------------------------------------------------------------------------------------
struct FaceModels {
    FaceModels(const std::string& dir, int threads);
    TfliteModel detector, landmarks;
    std::vector<Anchor> anchors;
    StageStats stats;
};

// rescan_every > 0 (single instance only): every that many tracked frames, run the detector anyway and switch
// to a clearly bigger (closer) instance than the tracked one. Detections are then ranked by size, not score.
class FaceTracker {
public:
    FaceTracker(FaceModels& m, int num_faces, int rescan_every = 0);
    std::vector<Landmarks> process(const cv::Mat& rgb, int64_t ts_ms);   // 478 normalized landmarks each
    int num_faces() const { return num_; }
private:
    std::vector<NormRect> detect(const cv::Mat& rgb, int limit);   // limit < 0: all
    FaceModels& m_;
    int num_, rescan_;
    long frame_ = 0;
    std::vector<NormRect> prev_;
    std::optional<LandmarksSmoother> smoother_;   // only with num_faces == 1 (MediaPipe VIDEO mode)
};

// ---- pose ----------------------------------------------------------------------------------------
struct PoseModels {
    PoseModels(const std::string& dir, int threads);
    TfliteModel detector, landmarks;
    std::vector<Anchor> anchors;
    StageStats stats;
};

class PoseTracker {
public:
    PoseTracker(PoseModels& m, int num_poses, int rescan_every = 0);   // rescan_every: as FaceTracker
    std::vector<Landmarks> process(const cv::Mat& rgb, int64_t ts_ms);   // 33 normalized landmarks (+visibility)
    int num_poses() const { return num_; }
private:
    std::vector<NormRect> detect(const cv::Mat& rgb);
    PoseModels& m_;
    int num_, rescan_;
    long frame_ = 0;
    std::vector<NormRect> prev_;
    std::optional<LandmarksSmoother> smoother_;
    std::optional<VisibilitySmoother> vis_smoother_;
};

}  // namespace memecam::vision
