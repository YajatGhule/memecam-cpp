// People grouping, identities and helpers — 1:1 port of memecam/tracking.py (no model inference here).
#pragma once

#include <optional>
#include <vector>

#include "types.h"

namespace memecam {

constexpr float EAR_TO_FACE = 1.35f, POSE_FACE_MATCH = 1.5f, HAND_WRIST_MATCH = 1.2f, HAND_FACE_MATCH = 3.5f,
                WRIST_STRAY = 2.5f, GHOST_DIST = 2.0f, ID_MAX_JUMP = 1.5f, ID_FORGET = 1.0f, CROWD_LINGER = 1.5f;
constexpr int ID_CONFIRM = 2, SCAN_EVERY = 10;

struct HeadRef { cv::Point2f nose; float width; };

float pose_face_width(const Pose& pose);
std::optional<HeadRef> head_of(const Frame& person);
std::optional<Box> head_box(const Frame& person);
std::optional<Box> hands_box(const Frame& person);

struct Seen {                          // a recently seen person (Identities::recent)
    int pid = 0;
    std::optional<cv::Point2f> head;
    float width = 1.f, last = 0.f;
    std::vector<cv::Point2f> wrists;   // last pose wrists, only if recent
    int hits = 0;
};

// Group raw detections into people, largest first (ids assigned later by Identities).
std::vector<Frame> group_people(const std::vector<Face>& faces, const std::vector<std::shared_ptr<const Pose>>& poses,
                                const std::vector<Hand>& hands, int width, int max_people = 3,
                                float min_face_width = 0.08f, const std::vector<Seen>& recent = {});

class Identities {
public:
    explicit Identities(float max_jump = ID_MAX_JUMP, float forget = ID_FORGET);
    std::vector<Seen> recent(float now) const;
    void assign(std::vector<Frame>& people, float now);   // sets pid, fills `renamed`
    std::map<int, int> renamed;
private:
    float max_jump_, forget_;
    std::map<int, Seen> tracks_;
};

// 160x90 blurred grayscale for frame differencing (input RGB or BGR, any size).
cv::Mat motion_gray(const cv::Mat& img);

}  // namespace memecam
