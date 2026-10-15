// Gesture detectors and the trigger state machine — 1:1 port of memecam/gestures.py.
#pragma once

#include <map>
#include <optional>
#include <string>

#include "config.h"
#include "tracking.h"

namespace memecam {

struct Detection {
    bool raw = false;
    float score = 0.f;              // for the debug overlay
    std::optional<Box> anchor;      // bbox the overlay should sit next to
};

std::map<std::string, bool> finger_states(const Hand& hand, float ratio);
std::optional<HeadRef> head_ref(const Frame& fr);   // gestures.py head_ref (face mesh, else pose)

class Detectors {
public:
    explicit Detectors(std::map<std::string, double> thresholds);   // merged with Python DEFAULTS
    void calibrate();
    std::map<std::string, Detection> run(const Frame& fr, float now);
    double cfg(const std::string& key) const;
    struct State;                    // all EMA / baseline / history state (defined in gestures.cpp)
    ~Detectors();
    Detectors(Detectors&&) noexcept;
    Detectors& operator=(Detectors&&) noexcept;
private:
    std::map<std::string, double> cfg_;
    std::unique_ptr<State> st_;
};

class Trigger {   // idle --(raw held `hold`)--> active --(raw gone, min_show & linger elapsed)--> cooldown
public:
    Trigger(double hold = 0.3, double min_show = 0.0, double linger = 0.3, double cooldown = 0.0);
    bool update(bool raw, double now);
    bool active() const { return started_.has_value(); }
    double hold, min_show, linger, cooldown;
private:
    std::optional<double> raw_since_, started_;
    double last_raw_ = 0, cool_until_ = 0;
};

}  // namespace memecam
