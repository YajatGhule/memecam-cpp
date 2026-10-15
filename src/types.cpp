#include "types.h"

#include <algorithm>

namespace memecam {

Box bbox_of(const std::vector<Vec3>& pts) {
    if (pts.empty()) return {};
    float x0 = pts[0][0], y0 = pts[0][1], x1 = x0, y1 = y0;
    for (const auto& p : pts) {
        x0 = std::min(x0, p[0]); y0 = std::min(y0, p[1]);
        x1 = std::max(x1, p[0]); y1 = std::max(y1, p[1]);
    }
    return {int(x0), int(y0), int(x1) - int(x0), int(y1) - int(y0)};
}

Box Hand::bbox() const { return bbox_of(std::vector<Vec3>(pts.begin(), pts.end())); }

}  // namespace memecam
