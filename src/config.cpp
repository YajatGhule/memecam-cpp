#include "config.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <toml++/toml.hpp>

#ifndef MEMECAM_ROOT
#define MEMECAM_ROOT "."
#endif

namespace memecam {

namespace fs = std::filesystem;

std::string project_path(const std::string& rel) {
    fs::path p(rel);
    if (p.is_absolute()) return rel;
    return (fs::path(MEMECAM_ROOT) / p).string();
}

namespace {

std::string resolve(const std::string& path) {
    if (fs::exists(path) || fs::path(path).is_absolute()) return path;
    return project_path(path);
}

toml::table parse(const std::string& path) {
    std::string p = resolve(path);
    if (!fs::exists(p)) throw std::runtime_error("config not found: " + p);
    try {
        return toml::parse_file(p);
    } catch (const toml::parse_error& e) {
        throw std::runtime_error("config " + p + ": " + std::string(e.description()));
    }
}

template <typename T>
void get(const toml::table* t, const char* key, T& out) {
    if (!t) return;
    if (auto v = (*t)[key].value<T>()) out = *v;
}

void get_num(const toml::table* t, const char* key, double& out) {
    if (!t) return;
    const toml::node* n = t->get(key);
    if (!n) return;
    if (auto v = n->value<double>()) out = *v;   // value<double>() also converts integers
}

// Apply one file's settings on top of `cfg` (Python merge(): tables merged key by key, the rest replaced).
void apply(Config& cfg, const toml::table& root) {
    get(root["camera"].as_table(), "index", cfg.camera_index);
    get(root["camera"].as_table(), "width", cfg.width);
    get(root["camera"].as_table(), "height", cfg.height);
    get(root["camera"].as_table(), "mirror", cfg.mirror);
    get(root["tracking"].as_table(), "device", cfg.device);
    get(root["people"].as_table(), "max_people", cfg.max_people);
    get_num(root["people"].as_table(), "min_face_width", cfg.min_face_width);
    get_num(root["people"].as_table(), "forget", cfg.forget);
    get(root["display"].as_table(), "meme_on_face", cfg.meme_on_face);

    if (const toml::table* thr = root["thresholds"].as_table()) {
        for (const auto& [k, v] : *thr) {
            if (auto b = v.value_exact<bool>()) cfg.thresholds[std::string(k.str())] = *b ? 1.0 : 0.0;
            else if (auto d = v.value<double>()) cfg.thresholds[std::string(k.str())] = *d;
        }
    }

    if (const toml::table* memes = root["memes"].as_table()) {
        // toml++ tables are sorted by key: restore file order from the source positions.
        std::vector<std::pair<std::string, const toml::table*>> order;
        for (const auto& [k, v] : *memes)
            if (const toml::table* t = v.as_table()) order.emplace_back(std::string(k.str()), t);
        std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
            const auto &sa = a.second->source().begin, &sb = b.second->source().begin;
            return sa.line != sb.line ? sa.line < sb.line : sa.column < sb.column;
        });
        for (const auto& [name, t] : order) {
            auto it = std::find_if(cfg.memes.begin(), cfg.memes.end(), [&](const MemeSpec& m) { return m.name == name; });
            if (it == cfg.memes.end()) {
                cfg.memes.push_back(MemeSpec{});
                cfg.memes.back().name = name;
                it = cfg.memes.end() - 1;
            }
            MemeSpec& m = *it;
            get(t, "asset", m.asset);
            get(t, "title", m.title);
            get(t, "placement", m.placement);
            get_num(t, "height", m.height);
            get_num(t, "hold", m.hold);
            get_num(t, "min_show", m.min_show);
            get_num(t, "linger", m.linger);
            get_num(t, "cooldown", m.cooldown);
            get(t, "key_black", m.key_black);
            if (const toml::array* fr = (*t)["frames"].as_array(); fr && fr->size() == 2) {
                auto a = (*fr)[0].value<int64_t>(), b = (*fr)[1].value<int64_t>();
                if (a && b) m.frames = std::make_pair(int(*a), int(*b));
            }
            if (auto lf = (*t)["loop_from"].value<int64_t>()) m.loop_from = int(*lf);
        }
    }
}

}  // namespace

Config load_config(const std::string& base, const std::string& overlay) {
    Config cfg;
    apply(cfg, parse(base));
    if (!overlay.empty()) apply(cfg, parse(overlay));
    return cfg;
}

}  // namespace memecam
