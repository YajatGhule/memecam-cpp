// Tracker: MediaPipe-equivalent hand / face / pose pipelines on LiteRT, plus grouping into people.
// Mirrors memecam/tracking.py Tracker.process(): returns a top-level Frame whose `people` lists everyone.
#pragma once

#include <memory>
#include <string>

#include "config.h"
#include "types.h"

namespace memecam {

class Tracker {
public:
    Tracker(const Config& cfg, bool use_pose = true, bool pose_async = true);
    ~Tracker();
    Frame process(const cv::Mat& rgb, int64_t ts_ms);   // rgb: CV_8UC3 RGB, mirrored
    std::string device() const;                          // "cpu" / "gpu"
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace memecam
