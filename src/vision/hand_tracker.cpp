// GestureRecognizer graph (VIDEO mode): HandDetectorGraph + HandLandmarksDetectorGraph +
// HandLandmarksDeduplication + HandGestureRecognizerGraph. Options as set by the MediaPipe Tasks sources
// (hand_detector_graph.cc, hand_landmarks_detector_graph.cc, hand_landmarker_graph.cc,
// hand_gesture_recognizer_graph.cc) with min detection / presence / tracking confidence 0.5.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

#include "vision/pipelines.h"

namespace memecam::vision {

namespace {
constexpr float kMinDetection = 0.5f, kMinPresence = 0.5f, kMinTracking = 0.5f;
constexpr int kPalmSize = 192, kLmSize = 224;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// HandLandmarksToRectCalculator (note MediaPipe's numeric_limits<float>::min() initial maxima, kept as is).
// Given all 21 landmarks it keeps the partial set {0,1,2,3,5,6,9,10,13,14,17,18} (no fingertips), so its
// "PIP" indices 4/6/8 are really the index / middle / ring MCPs.
NormRect hand_landmarks_to_rect(const Landmarks& all, int W, int H) {
    static constexpr int kPartial[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
    Landmarks l;
    l.reserve(12);
    for (int i : kPartial) l.push_back(all[i]);
    constexpr int kWrist = 0, kIndexPip = 4, kMiddlePip = 6, kRingPip = 8;
    const float x0 = l[kWrist].x * W, y0 = l[kWrist].y * H;
    float x1 = (l[kIndexPip].x + l[kRingPip].x) / 2.f, y1 = (l[kIndexPip].y + l[kRingPip].y) / 2.f;
    x1 = (x1 + l[kMiddlePip].x) / 2.f * W;
    y1 = (y1 + l[kMiddlePip].y) / 2.f * H;
    const float rotation = normalize_radians(static_cast<float>(M_PI) * 0.5f - std::atan2(-(y1 - y0), x1 - x0));
    const float reverse = normalize_radians(-rotation);
    float max_x = std::numeric_limits<float>::min(), max_y = max_x;
    float min_x = std::numeric_limits<float>::max(), min_y = min_x;
    for (auto& p : l) {
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
    }
    const float acx = (max_x + min_x) / 2.f, acy = (max_y + min_y) / 2.f;
    max_x = max_y = std::numeric_limits<float>::min();
    min_x = min_y = std::numeric_limits<float>::max();
    for (auto& p : l) {
        const float ox = (p.x - acx) * W, oy = (p.y - acy) * H;
        const float px = ox * std::cos(reverse) - oy * std::sin(reverse);
        const float py = ox * std::sin(reverse) + oy * std::cos(reverse);
        max_x = std::max(max_x, px);
        max_y = std::max(max_y, py);
        min_x = std::min(min_x, px);
        min_y = std::min(min_y, py);
    }
    const float pcx = (max_x + min_x) / 2.f, pcy = (max_y + min_y) / 2.f;
    const float cx = pcx * std::cos(rotation) - pcy * std::sin(rotation) + W * acx;
    const float cy = pcx * std::sin(rotation) + pcy * std::cos(rotation) + H * acy;
    return {cx / W, cy / H, (max_x - min_x) / W, (max_y - min_y) / H, rotation};
}

// HandDuplicatesFinder (hand_landmarks_deduplication_calculator.cc).
float px_dist(const NormLandmark& a, const NormLandmark& b, int W, int H) {
    return std::sqrt(std::pow((a.x - b.x) * W, 2.f) + std::pow((a.y - b.y) * H, 2.f));
}
std::set<int> find_duplicates(const std::vector<HandResult>& hands, int W, int H) {
    struct B { float l, t, r, b; };
    const int n = static_cast<int>(hands.size());
    std::vector<float> base(n);
    std::vector<B> bounds(n);
    for (int i = 0; i < n; ++i) {
        const auto& l = hands[i].lms;
        float d = px_dist(l[0], l[5], W, H);
        d = std::max(d, px_dist(l[5], l[17], W, H));
        d = std::max(d, px_dist(l[17], l[0], W, H));
        base[i] = d;
        B bb{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
             std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        for (auto& p : l) {
            bb.l = std::min(bb.l, p.x);
            bb.t = std::min(bb.t, p.y);
            bb.r = std::max(bb.r, p.x);
            bb.b = std::max(bb.b, p.y);
        }
        bounds[i] = bb;
    }
    auto area = [](const B& b) { return (b.r - b.l) * (b.b - b.t); };
    auto iou = [&](const B& a, const B& b) {
        const float aa = area(a), ab = area(b);
        if (aa <= 0 || ab <= 0) return 0.f;
        const float iw = std::max(0.f, std::min(a.r, b.r) - std::max(a.l, b.l));
        const float ih = std::max(0.f, std::min(a.b, b.b) - std::max(a.t, b.t));
        const float inter = iw * ih;
        return inter / (aa + ab - inter);
    };
    std::vector<int> retained;
    std::set<int> suppressed;
    for (int i = 0; i < n; ++i) {
        bool sup = false;
        for (int j : retained) {
            const float thr = std::max(base[i], base[j]) * 0.2f;
            int matched = 0;
            for (int k = 0; k < 21; ++k)
                if (px_dist(hands[i].lms[k], hands[j].lms[k], W, H) < thr) ++matched;
            if (matched >= 10 && iou(bounds[i], bounds[j]) > 0.2f) {
                sup = true;
                break;
            }
        }
        if (sup) suppressed.insert(i);
        else retained.push_back(i);
    }
    return suppressed;
}

// LandmarksToMatrixCalculator::NormalizeObject (origin landmark 0).
void normalize_object(std::vector<cv::Vec3f>& p) {
    const cv::Vec3f o = p[0];
    float min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::min();
    float min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::min();
    for (auto& v : p) {
        v -= o;
        min_x = std::min(min_x, v[0]);
        max_x = std::max(max_x, v[0]);
        min_y = std::min(min_y, v[1]);
        max_y = std::max(max_y, v[1]);
    }
    const float scale = std::max(max_x - min_x, max_y - min_y) + 1e-5f;
    for (auto& v : p) v /= scale;
}
}  // namespace

HandModels::HandModels(const std::string& dir, int threads)
    : palm(dir + "/hand_detector.tflite", threads),
      landmarks(dir + "/hand_landmarks_detector.tflite", threads),
      embedder(dir + "/gesture_embedder.tflite", 1),
      classifier(dir + "/canned_gesture_classifier.tflite", 1),
      labels{"None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"} {
    AnchorOptions a;
    a.input_w = a.input_h = kPalmSize;
    a.min_scale = 0.1484375f;
    a.max_scale = 0.75f;
    a.strides = {8, 16, 16, 16};
    anchors = ssd_anchors(a);
}

std::vector<NormRect> HandTracker::detect(const cv::Mat& rgb) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    TensorInput ti;
    image_to_tensor(rgb, nullptr, kPalmSize, kPalmSize, true, true, 0.f, 1.f, m_.palm.input(0), &ti);
    m_.palm.invoke();
    const int bi = m_.palm.output_size(0) > m_.palm.output_size(1) ? 0 : 1;
    DecodeOptions o{2016, 18, 0, 4, 7, 2, 192.f, 192.f, 192.f, 192.f, 100.f, kMinDetection, true};
    auto dets = weighted_nms(decode_detections(m_.palm.output(bi), m_.palm.output(1 - bi), m_.anchors, o), 0.3f);
    std::vector<NormRect> rects;
    for (auto& d : dets) {
        project_detection(d, ti.matrix);
        // hand_detector_graph.cc sets rotation_vector_target_angle(90), a field in *radians*: MediaPipe really
        // rotates the first crop by 90 rad (~116.6 deg); the landmark model copes, and parity needs the same.
        NormRect r = detection_to_rect(d, W, H, 0, 2, 90.f);
        rects.push_back(transform_rect(r, W, H, 2.6f, 2.6f, 0.f, -0.5f, true));
        if (static_cast<int>(rects.size()) == num_) break;   // ClipNormalizedRectVectorSize(num_hands)
    }
    m_.stats.add(m_.stats.det_ms, ms_since(t0));
    m_.stats.det_runs.fetch_add(1);
    return rects;
}

void HandTracker::classify(HandResult& h, int W, int H) {
    const auto t0 = Clock::now();
    // Landmarks: NormalizeLandmarkAspectRatio, RotateLandmarks (input NORM_RECT rotation 0), NormalizeObject.
    const float max_dim = static_cast<float>(std::max(W, H));
    const float ws = W / max_dim, hs = H / max_dim;
    std::vector<cv::Vec3f> lm(21), world(21);
    for (int i = 0; i < 21; ++i) {
        lm[i] = {(h.lms[i].x - 0.5f) * ws + 0.5f, (h.lms[i].y - 0.5f) * hs + 0.5f, h.lms[i].z};
        world[i] = {h.world[i].x, h.world[i].y, h.world[i].z};
    }
    normalize_object(lm);
    normalize_object(world);
    float* in0 = m_.embedder.input(0);
    float* in1 = m_.embedder.input(1);
    float* in2 = m_.embedder.input(2);
    for (int i = 0; i < 21; ++i)
        for (int c = 0; c < 3; ++c) {
            in0[i * 3 + c] = lm[i][c];
            in2[i * 3 + c] = world[i][c];
        }
    in1[0] = h.handedness;   // HandednessToMatrix: score of "Right"
    m_.embedder.invoke();
    std::copy_n(m_.embedder.output(0), 128, m_.classifier.input(0));
    m_.classifier.invoke();
    const float* p = m_.classifier.output(0);
    const int n = std::min<int>(m_.classifier.output_size(0), static_cast<int>(m_.labels.size()));
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (p[i] > p[best]) best = i;
    h.gesture = m_.labels[best];
    h.gesture_score = p[best];
    m_.stats.add(m_.stats.extra_ms, ms_since(t0));
}

std::vector<HandResult> HandTracker::process(const cv::Mat& rgb, int64_t) {
    const auto t0 = Clock::now();
    const int W = rgb.cols, H = rgb.rows;
    std::vector<NormRect> rects = prev_;
    if (static_cast<int>(prev_.size()) < num_) rects = hand_association(prev_, detect(rgb), kMinTracking);
    if (static_cast<int>(rects.size()) > num_) rects.resize(num_);

    std::vector<HandResult> hands;
    std::vector<NormRect> next;
    const auto t1 = Clock::now();
    for (const auto& rect : rects) {
        if (!image_to_tensor(rgb, &rect, kLmSize, kLmSize, false, false, 0.f, 1.f, m_.landmarks.input(0), nullptr))
            continue;
        m_.landmarks.invoke();
        m_.stats.lm_runs.fetch_add(1);
        const float presence = m_.landmarks.output(1)[0];
        if (!(presence > kMinPresence)) continue;
        HandResult h;
        const float* raw = m_.landmarks.output(0);
        const float* wr = m_.landmarks.output(3);
        h.lms.resize(21);
        h.world.resize(21);
        const float c = std::cos(rect.rot), s = std::sin(rect.rot);
        for (int i = 0; i < 21; ++i) {
            h.lms[i] = {raw[i * 3] / kLmSize, raw[i * 3 + 1] / kLmSize, raw[i * 3 + 2] / kLmSize / 0.4f, 0, 0};
            const float x = wr[i * 3], y = wr[i * 3 + 1];
            h.world[i] = {c * x - s * y, s * x + c * y, wr[i * 3 + 2], 0, 0};
        }
        project_landmarks_rect(h.lms, rect);
        h.handedness = m_.landmarks.output(2)[0];
        h.label = h.handedness >= 1.f - h.handedness ? "Right" : "Left";
        next.push_back(transform_rect(hand_landmarks_to_rect(h.lms, W, H), W, H, 2.f, 2.f, 0.f, -0.1f, true));
        hands.push_back(std::move(h));
    }
    m_.stats.add(m_.stats.lm_ms, ms_since(t1));

    const auto dup = find_duplicates(hands, W, H);
    std::vector<HandResult> kept;
    prev_.clear();
    for (int i = 0; i < static_cast<int>(hands.size()); ++i)
        if (!dup.count(i)) {
            kept.push_back(std::move(hands[i]));
            prev_.push_back(next[i]);
        }
    for (auto& h : kept) classify(h, W, H);
    m_.stats.add(m_.stats.total_ms, ms_since(t0));
    m_.stats.calls.fetch_add(1);
    return kept;
}

}  // namespace memecam::vision
