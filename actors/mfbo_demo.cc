// Standalone driver for mfbo.h, mirroring the Python reference's
// "Running It" section so the two can be diffed against each other.
#include <cstdio>
#include "mfbo.h"

int main() {
    mfbo::MFBO bo(mfbo::low_fn, mfbo::high_fn, {0.0, 6.0});
    auto [x_min, y_min] = bo.run(30);
    std::printf("minimum: %.15g %.15g\n", x_min, y_min);
    return 0;
}
