// Driver for mfbo.h.
//
//   ./mfbo_demo              -> multi-fidelity BO over the beta-VAE MMD objective
//                               (low fidelity = few epochs, high = full run)
//   ./mfbo_demo --synthetic  -> the original analytic low_fn/high_fn test, kept
//                               as a regression check on the GP / acquisition math
//                               (diff it against the Python reference)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "mfbo.h"

namespace {

// ----- beta objective -------------------------------------------------------

// BO runs in normalized space u in [0,1]; beta = denorm(u).
constexpr double BETA_LO = 0.05, BETA_HI = 2.5;
double denorm(double u) { return BETA_LO + (BETA_HI - BETA_LO) * u; }

// One training run of latentflow.py at the given beta.
//   fidelity 0 = cheap probe (few epochs), 1 = full run.
// Returns log10(MMD) -- monotonic in MMD, keeps the argmin, and gives the GP
// a sane dynamic range (raw MMD is ~1e-2..1e-5). Memoized by (beta, fidelity)
// so repeated acquisition picks don't re-train.
double run_latentflow(double beta, int fidelity) {
    static std::map<std::pair<long, int>, double> cache;
    auto key = std::make_pair(std::lround(beta * 1e4), fidelity);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    const char* epochs    = fidelity ? "5"   : "1";
    const char* fm_epochs  = fidelity ? "5"   : "1";
    const char* batch_size = fidelity ? "512" : "256";
    const char* suffix     = fidelity ? "_hi" : "_lo";

    std::ostringstream cmd;
    cmd << "../.venv/bin/python3 ./latentflow1.py --quiet"
        << " --beta "       << beta
        << " --epochs "     << epochs
        << " --fm-epochs "  << fm_epochs
        << " --batch-size " << batch_size
        << " --tag-suffix " << suffix;

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { std::perror("popen"); std::exit(1); }
    char buf[256];
    std::string out;
    while (std::fgets(buf, sizeof buf, pipe)) out += buf;
    int rc = pclose(pipe);
    if (rc != 0) {
        std::fprintf(stderr, "latentflow.py failed (rc=%d) for beta=%.4f f=%d\n",
                     rc, beta, fidelity);
        std::exit(1);
    }

    double mmd = std::stod(out);
    double y = std::log10(mmd);
    cache[key] = y;
    std::fprintf(stderr, "  [f=%d] beta=%.4f  MMD=%.3e  log10=%.4f\n",
                 fidelity, beta, mmd, y);
    return y;
}

int run_beta() {
    auto low  = [](double u) { return run_latentflow(denorm(u), 0); };
    auto high = [](double u) { return run_latentflow(denorm(u), 1); };

    mfbo::MFBO bo(low, high, {0.0, 1.0});

    // Every step here is a full training run -- keep n small until the
    // plumbing and the log-MMD values look sane, then widen.
    auto [u_best, y_best] = bo.run(/*n=*/12);

    std::printf("\nbest beta = %.4f   MMD = %.3e   (log10 = %.4f)\n",
                denorm(u_best), std::pow(10.0, y_best), y_best);

    // Model's predicted high-fidelity MMD curve over the whole range.
    mfbo::Vec grid(50), mean, sd;
    for (int i = 0; i < 50; ++i) grid[i] = i / 49.0;
    bo.predict_high(grid, mean, sd);
    std::printf("\n# beta      pred_MMD    log10_sd\n");
    for (int i = 0; i < 50; ++i)
        std::printf("%.4f   %.3e   %.3f\n",
                    denorm(grid[i]), std::pow(10.0, mean[i]), sd[i]);
    return 0;
}

// ----- original analytic test --------------------------------------------

int run_synthetic() {
    mfbo::MFBO bo(mfbo::low_fn, mfbo::high_fn, {0.0, 6.0});
    auto [x_min, y_min] = bo.run(30);
    std::printf("minimum: %.15g %.15g\n", x_min, y_min);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--synthetic") == 0) return run_synthetic();
    return run_beta();
}
