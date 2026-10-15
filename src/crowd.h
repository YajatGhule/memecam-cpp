// Per-person meme state (Crowd / Person / Shown in memecam/main.py), without any rendering.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "config.h"
#include "gestures.h"

namespace memecam {

struct MemeState {                 // Python Meme minus the media
    const MemeSpec* spec = nullptr;
    Trigger trigger;
    std::optional<double> started; // when it switched on (gif restarts from frame 0)
    std::optional<Box> anchor;
    bool update(const Detection& det, double now);
};

struct Shown {                      // what the display thread draws for one person (a consistent copy)
    int pid;
    std::optional<Box> head;
    std::map<std::string, MemeState> memes;
};

class Crowd {
public:
    explicit Crowd(const Config& cfg);
    void calibrate();
    // Run every present person's detectors + triggers. Returns detections per person id.
    std::map<int, std::map<std::string, Detection>> update(const Frame& fr, double now,
                                                           const std::vector<std::string>& force = {});
    std::vector<Shown> snapshot() const;   // only people present in the last update
private:
    struct Person;
    const Config& cfg_;
    std::map<int, std::unique_ptr<Person>> people_;
    std::vector<int> present_;
};

}  // namespace memecam
