// Driver for mfbo.h.
//
//   ./mfbo              -> sequential (no actors) high-fidelity-only BO over
//                           log10(beta), mirroring mfbo_actors.cc's search
//                           space/fidelity strategy exactly so the two are
//                           directly comparable -- the only difference is
//                           mfbo_actors dispatches each round's batch to
//                           parallel CAF worker actors while this driver
//                           evaluates the same batch one at a time
//                           in-process.
//   ./mfbo --synthetic -> the original analytic low_fn/high_fn test, kept
//                           as a regression check on the GP / acquisition
//                           math (diff it against the Python reference)

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "mfbo.h"

namespace {

// "YYYYmmdd_HHMMSS" local-time stamp, used to tag this run's output files so
// successive runs don't overwrite each other. Same scheme as mfbo_actors.cc.
std::string run_stamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

// ----- beta objective -------------------------------------------------------

// Same log10(beta) search space as mfbo_actors.cc: the optimizer's u in [0,1]
// maps to log10(beta) in [LOG10_BETA_LO, LOG10_BETA_HI], i.e. beta in
// [1e-8, 1e1]. Kept identical to mfbo_actors so the two runs' graphs line up.
constexpr double LOG10_BETA_LO = -8.0;
constexpr double LOG10_BETA_HI =  1.0;

double log10_beta(double u) { return LOG10_BETA_LO + (LOG10_BETA_HI - LOG10_BETA_LO) * u; }
double denorm(double u)     { return std::pow(10.0, log10_beta(u)); }

// One full-fidelity training run of latentflow1.py at the given beta (no
// --epochs/--fm-epochs/--batch-size overrides -- latentflow1.py's own
// defaults, same as mfbo_actors.cc's worker). Returns raw MMD. Memoized by beta
// so repeated acquisition picks don't re-train. "_demo" tag suffix keeps
// this driver's output files from colliding with mfbo_actors's "_hf" ones.
double run_latentflow(double beta) {
    static std::map<long, double> cache;
    long key = std::lround(beta * 1e8);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    std::ostringstream cmd;
    cmd << "../.venv/bin/python3 ./latentflow1.py --quiet"
        << " --beta "       << beta
        << " --tag-suffix " << "_demo";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { std::perror("popen"); std::exit(1); }
    char buf[256];
    std::string out;
    while (std::fgets(buf, sizeof buf, pipe)) out += buf;
    int rc = pclose(pipe);
    if (rc != 0) {
        std::fprintf(stderr, "latentflow.py failed (rc=%d) for beta=%.4e\n", rc, beta);
        std::exit(1);
    }

    double mmd = std::stod(out);
    cache[key] = mmd;
    return mmd;
}

// One completed run, kept so the driver can dump a CSV for plotting -- same
// shape as mfbo_actors.cc's run_point.
struct run_point {
    double u;
    double mmd;
    double pred_log10_before;
};

// Evaluate one point at full fidelity, fold it into the GP, and append it to
// the trace. `approx` is the GP's prediction for this point just before the
// real result is recorded -- purely informational, mirrors mfbo_actors.cc.
void run_one(mfbo::MFBO& bo, double u, std::vector<run_point>& trace) {
    double mmd = run_latentflow(denorm(u));
    // Defensive floor for exact zero / round-off; latentflow1.py's
    // compute_mmd() already floors MMD at 0.0. Same rationale as mfbo_actors.cc.
    double y = std::log10(std::max(mmd, 1e-6));

    double approx = std::numeric_limits<double>::quiet_NaN();
    if (!bo.high_X().empty()) {
        mfbo::Vec m, s;
        bo.predict_high({u}, m, s);
        approx = m[0];
    }

    bo.record(u, /*fidelity=*/1, y);
    trace.push_back({u, mmd, approx});
    std::fprintf(stderr, "log10_beta=%.4f  beta=%.4e  MMD=%.4e  log10_mmd=%.4f  (GP predicted log10_mmd~%.4f)\n",
                 log10_beta(u), denorm(u), mmd, y, approx);
}

int run_beta() {
    const std::string ts = run_stamp();
    const auto wall_start = std::chrono::steady_clock::now();
    std::fprintf(stderr, "mfbo run %s started\n", ts.c_str());

    // Same batch/rounds as mfbo_actors.cc's defaults (workers=3 -> batch=3,
    // rounds=10), so both drivers make the same number of evaluations over
    // the same acquisition schedule -- only the parallelism differs.
    const int batch  = 3;
    const int rounds = 10;

    // The constructor still wants low_()/high_() functors, but this driver
    // never calls evaluate() -- every observation comes in through record().
    auto unused = [](double) -> double {
        throw std::logic_error("mfbo: low_()/high_() are never called directly; "
                                "high-fidelity results come from record(), low "
                                "fidelity is predict_high()");
    };
    mfbo::MFBO bo(unused, unused, {0.0, 1.0});
    bo.length = 0.1; // same RBF lengthscale as mfbo_actors.cc

    std::vector<run_point> trace;

    // Seed round: a spread of betas across the range, all at full fidelity.
    for (int i = 0; i < batch; ++i)
        run_one(bo, batch > 1 ? i / double(batch - 1) : 0.5, trace);

    // Acquisition rounds: propose a batch, evaluate every point in it
    // sequentially (no actors -- this is the point of the comparison).
    for (int r = 0; r < rounds; ++r) {
        std::vector<double> xs = bo.propose_batch(batch);
        std::fprintf(stderr, "--- round %d : %zu full runs ---\n", r, xs.size());
        for (double x : xs) run_one(bo, x, trace);
    }

    const mfbo::Vec& hx = bo.high_X();
    const mfbo::Vec& hy = bo.high_y();
    size_t best_i = std::min_element(hy.begin(), hy.end()) - hy.begin();
    std::printf("\nbest  log10_beta = %.4f  beta = %.4e  MMD = %.4e\n",
                log10_beta(hx[best_i]), denorm(hx[best_i]), std::pow(10.0, hy[best_i]));

    // Predicted full-fidelity MMD curve, even in log10_beta.
    mfbo::Vec grid(50), mean, sd;
    for (int i = 0; i < 50; ++i) grid[i] = i / 49.0;
    bo.predict_high(grid, mean, sd);
    std::printf("\n# log10_beta   beta          pred_MMD      log10_mmd_sd\n");
    for (int i = 0; i < 50; ++i)
        std::printf("%.4f   %.4e   %.4e   %.4f\n",
                    log10_beta(grid[i]), denorm(grid[i]), std::pow(10.0, mean[i]), sd[i]);

    // ---- CSV dump for plot_mfbo.py, same schema as mfbo_actors.cc ----------
    std::ostringstream runs_csv;
    runs_csv << "order,log10_beta,beta,mmd,log10_mmd,gp_pred_log10_mmd_before\n";
    for (size_t i = 0; i < trace.size(); ++i) {
        const run_point& p = trace[i];
        runs_csv << i << ',' << log10_beta(p.u) << ',' << denorm(p.u) << ','
                 << p.mmd << ',' << std::log10(std::max(p.mmd, 1e-6)) << ','
                 << p.pred_log10_before << '\n';
    }
    std::ostringstream pred_csv;
    pred_csv << "log10_beta,beta,pred_mmd,pred_log10_mmd,log10_mmd_sd\n";
    for (int i = 0; i < 50; ++i)
        pred_csv << log10_beta(grid[i]) << ',' << denorm(grid[i]) << ','
                 << std::pow(10.0, mean[i]) << ',' << mean[i] << ',' << sd[i] << '\n';

    const std::string runs_path = "mfbo_runs_" + ts + ".csv";
    const std::string pred_path = "mfbo_pred_" + ts + ".csv";
    for (const std::string& p : {runs_path, std::string("mfbo_runs.csv")})
        std::ofstream(p) << runs_csv.str();
    for (const std::string& p : {pred_path, std::string("mfbo_pred.csv")})
        std::ofstream(p) << pred_csv.str();
    std::printf("\nwrote %s, %s (and mfbo_runs.csv / mfbo_pred.csv)\n",
                runs_path.c_str(), pred_path.c_str());

    // Render the PNG straight away, same as mfbo_actors.cc. Best-effort: if
    // matplotlib/the venv isn't there, the CSVs are still on disk.
    const std::string png_path = "mfbo_" + ts + ".png";
    std::string cmd = "../.venv/bin/python3 ./plot_mfbo.py " + runs_path + " " + pred_path
                     + " " + png_path + " mfbo && cp " + png_path + " mfbo.png";
    int rc = std::system(cmd.c_str());
    if (rc == 0)
        std::printf("wrote %s (and mfbo.png)\n", png_path.c_str());
    else
        std::printf("plot_mfbo.py exited with %d -- CSVs are still there, plot them manually\n", rc);

    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - wall_start).count();
    std::printf("mfbo run %s finished in %lld s (%zu full runs)\n",
                ts.c_str(), static_cast<long long>(secs), trace.size());
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
