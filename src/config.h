// Config: config/config.toml (+ config/config_v2.toml layered on top for memecamv2). Mirrors the Python dicts.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace memecam {

struct MemeSpec {
    std::string name, asset, title, placement = "beside";
    double height = 0.35, hold = 0.3, min_show = 0.0, linger = 0.3, cooldown = 0.0;
    bool key_black = false;
    std::optional<std::pair<int, int>> frames;    // trim [first, last]
    std::optional<int> loop_from;
};

struct Config {
    // [camera]
    int camera_index = 0, width = 1280, height = 720;
    bool mirror = true;
    // [tracking]
    std::string device = "auto";
    // [people]
    int max_people = 3;
    double min_face_width = 0.08, forget = 1.0;
    // [display]
    bool meme_on_face = false;
    // [thresholds]: free-form numbers/bools (bools stored as 0/1), read via thr(key, default)
    std::map<std::string, double> thresholds;
    // [memes.*] in file order
    std::vector<MemeSpec> memes;

    double thr(const std::string& key, double def) const {
        auto it = thresholds.find(key);
        return it == thresholds.end() ? def : it->second;
    }
};

// Load `base` and, if given, merge `overlay` on top (tables merged key by key, like Python merge()).
Config load_config(const std::string& base, const std::string& overlay = "");

std::string project_path(const std::string& rel);   // MEMECAM_ROOT / rel

}  // namespace memecam
