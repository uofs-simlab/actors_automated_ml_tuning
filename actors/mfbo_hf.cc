// Single-fidelity parallel BO over beta. Same worker-actor pattern as
// mfbo_caf.cc, but there is only ONE fidelity here: every evaluation is a
// full latentflow1.py run at the script's own default settings (EPOCHS /
// EPOCHS_FM / BATCH_SIZE from the top of latentflow1.py) -- no --epochs /
// --batch-size overrides, no cheap truncated "low" runs.
//
// "Low fidelity" in this driver never launches anything. It is just the
// GP's own prediction of what a full run at that beta *would* return
// (mfbo::MFBO::predict_high -> posterior mean), which costs nothing and is
// never fed back into the model as an observation. So:
//
//   high fidelity  -> a real latentflow1.py run, recorded into the GP
//   low  fidelity  -> bo.predict_high(beta), free, for logging / the
//                     acquisition surface only
//
// That makes this ordinary batch Bayesian optimization: the only cost is
// the high-fidelity runs; the GP interpolates everywhere you did not run.
//
// Build:  make mfbo_hf   (needs the same CAF_PREFIX as test_actor)
// Run:    ./mfbo_hf --workers 4 --gpus 2

#include "config.h" // CAF, job registration helpers, the `config` CLI class
#include "mfbo.h"
#include <chrono>
#include <cmath>
#include <ctime>

namespace {
// "YYYYmmdd_HHMMSS" local-time stamp, used to tag this run's output files
// so successive runs don't overwrite each other.
std::string run_stamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}
} // namespace

// One job = one beta to train + evaluate at full fidelity. `u` is the
// *normalized* value in [0,1] that mfbo::MFBO works in -- see denorm()
// below -- the worker converts it to a real beta before calling Python.
struct job_t {
    double u;

    template <class Inspector>
    friend bool inspect(Inspector& f, job_t& j) {
        return f.object(j).fields(f.field("u", j.u));
    }
};

CAF_BEGIN_TYPE_ID_BLOCK(mfbo_hf_project, caf::first_custom_type_id)
CAF_ADD_TYPE_ID(mfbo_hf_project, (job_t))
CAF_END_TYPE_ID_BLOCK(mfbo_hf_project)

namespace {
// beta is searched on a LOG10 scale: the optimizer's u in [0,1] maps to
// log10(beta) in [LOG10_BETA_LO, LOG10_BETA_HI], i.e. beta in
// [1e-8, 1e1]. Log spacing gives the GP even resolution across all nine
// decades; linear spacing would put almost every sample above beta=1 and
// never probe the small-beta regime. Set LOG10_BETA_HI = 0.0 for a 1e0
// (beta = 1) ceiling instead.
constexpr double LOG10_BETA_LO = -8.0;
constexpr double LOG10_BETA_HI =  1.0;

double log10_beta(double u) { return LOG10_BETA_LO + (LOG10_BETA_HI - LOG10_BETA_LO) * u; }
double denorm(double u)     { return std::pow(10.0, log10_beta(u)); }
} // namespace

// ---- worker: one full-fidelity latentflow1.py run per job ----------------

struct worker_state {
    actor manager_actor;
};

behavior worker_actor(stateful_actor<worker_state>* self, actor manager, int num_gpus) {
    self->state().manager_actor = manager;

    return {
        [=](job_t job) {
            actor self_hdl{self};
            actor mgr = self->state().manager_actor;

            static std::atomic<int> next_gpu{0};
            int gpu_id = num_gpus > 0 ? (next_gpu.fetch_add(1) % num_gpus) : -1;

            std::thread([self_hdl, mgr, job, gpu_id]() mutable {
                double beta = denorm(job.u);

                // No --epochs / --fm-epochs / --batch-size: latentflow1.py
                // falls back to its own module-level defaults, i.e. a full run.
                std::ostringstream command;
                if (gpu_id >= 0) command << "CUDA_VISIBLE_DEVICES=" << gpu_id << " ";
                command << "../.venv/bin/python3 ./latentflow1.py --quiet"
                        << " --beta "       << beta
                        << " --tag-suffix " << "_hf";

                double result = 0.0;
                FILE* pipe = popen(command.str().c_str(), "r");
                if (!pipe) {
                    std::cerr << "Failed to execute Python script for beta=" << beta << "\n";
                    anon_mail("result", job, result, self_hdl).send(mgr);
                    return;
                }

                char buffer[128];
                std::string output;
                while (fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;

                int return_code = pclose(pipe);
                if (return_code == 0) {
                    try {
                        result = std::stod(output);
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing result for beta=" << beta << ": " << e.what() << "\n";
                    }
                } else {
                    std::cerr << "Python script failed for beta=" << beta
                              << " with return code: " << return_code << "\n";
                }

                anon_mail("result", job, result, self_hdl).send(mgr);
            }).detach();
        },
        [=](std::string message) {
            if (message == "quit") self->quit();
        },
    };
}

// One completed high-fidelity run, kept so the driver can dump a CSV for
// plotting: u (normalized), the measured MMD, and what the GP had predicted
// (log10 MMD) for that beta just before this result was folded in.
struct run_point {
    double u;
    double mmd;
    double pred_log10_before;
};

// ---- one round: spawn `jobs.size()` workers, hand each one job, block ----
// ---- until every result is back, fold each into the GP.               ----

void run_round(caf::scoped_actor& self, actor_system& system, int num_gpus,
                mfbo::MFBO& bo, const std::vector<job_t>& jobs,
                std::vector<run_point>& trace) {
    int pending = static_cast<int>(jobs.size());
    std::vector<actor> workers;
    workers.reserve(jobs.size());

    //Assign worker per job 
    for (const job_t& job : jobs) {
        actor worker = system.spawn(worker_actor, actor{self}, num_gpus);
        workers.push_back(worker);
        anon_mail(job).send(worker);
    }

    while (pending > 0) {
        self->receive(
            [&](std::string msg, job_t job, double mmd, actor /*worker*/) {
                if (msg != "result") return;
                // Defensive floor for exact zero / round-off; latentflow1.py's
                // compute_mmd() already floors MMD at 0.0. 1e-6 keeps log10 near
                // the range of real readings instead of an outlier that would
                // dominate the GP's nugget-variance estimate.
                double y = std::log10(std::max(mmd, 1e-6));

                // "low fidelity" = what the GP predicted this run would give,
                // BEFORE recording the real result. Purely informational.
                double approx = std::numeric_limits<double>::quiet_NaN();
                if (!bo.high_X().empty()) {
                    mfbo::Vec m, s;
                    bo.predict_high({job.u}, m, s);
                    approx = m[0];
                }

                bo.record(job.u, /*fidelity=*/1, y); // only high fidelity is ever recorded
                trace.push_back({job.u, mmd, approx});
                self->println("log10_beta={}  beta={}  MMD={}  log10_mmd={}  (GP predicted log10_mmd~{})",
                               log10_beta(job.u), denorm(job.u), mmd, y, approx);
                --pending;
            });
    }
    for (actor& w : workers) anon_mail("quit").send(w);
}

// ---- driver ---------------------------------------------------------------

void caf_main(actor_system& system, const config& cfg) {
    caf::scoped_actor self{system};

    const std::string ts = run_stamp();
    const auto wall_start = std::chrono::steady_clock::now();
    self->println("mfbo_hf run {} started", ts);

    const int num_gpus = cfg.gpus;
    const int batch    = std::max(2, cfg.workers); // full runs in parallel per round
    const int rounds   = 10;                         // acquisition rounds after the seed

    // The constructor still wants low_()/high_() functors, but this driver
    // never calls evaluate()/run() -- every observation comes in through
    // record(). Throwing placeholders make a stray call loud instead of silent.
    auto unused = [](double) -> double {
        throw std::logic_error("mfbo_hf: low_()/high_() are never called directly; "
                                "high-fidelity results come from record(), low "
                                "fidelity is predict_high()");
    };
    mfbo::MFBO bo(unused, unused, {0.0, 1.0});

    // GP kernel lengthscale, in normalized u units (u in [0,1] spans the
    // whole log10(beta) range, i.e. 9 decades). ~0.1 => one correlation
    // length is roughly one decade of beta. Raise it for a smoother/stiffer
    // fit, lower it if the GP mean isn't tracking the sampled points.
    bo.length = 0.1;

    std::vector<run_point> trace; // every completed high-fidelity run, for the CSV

    // Seed round: a spread of betas across the range, all run at full fidelity
    // (there is no cheap tier to seed with here).
    std::vector<job_t> seed;
    for (int i = 0; i < batch; ++i)
        seed.push_back({batch > 1 ? i / double(batch - 1) : 0.5});
    run_round(self, system, num_gpus, bo, seed, trace);

    //Runs the proposed batch after seeding
    for (int r = 0; r < rounds; ++r) {
        std::vector<double> xs = bo.propose_batch(batch);
        std::vector<job_t> jobs;
        jobs.reserve(xs.size());
        for (double x : xs) jobs.push_back({x});
        self->println("--- round {} : {} full runs ---", r, jobs.size());
        run_round(self, system, num_gpus, bo, jobs, trace);
    }

    const mfbo::Vec& hx = bo.high_X();
    const mfbo::Vec& hy = bo.high_y();
    size_t best_i = std::min_element(hy.begin(), hy.end()) - hy.begin();
    self->println("\nbest  log10_beta = {}  beta = {}  MMD = {}",
                   log10_beta(hx[best_i]), denorm(hx[best_i]), std::pow(10.0, hy[best_i]));

    // Predicted full-fidelity MMD curve -- the "low fidelity" approximation
    // for every beta, including all the ones never actually run. Grid is even
    // in log10_beta (that is the space the optimizer works in).
    mfbo::Vec grid(50), mean, sd;
    for (int i = 0; i < 50; ++i) grid[i] = i / 49.0;
    bo.predict_high(grid, mean, sd);
    self->println("\n# log10_beta   beta          pred_MMD      log10_mmd_sd");
    for (int i = 0; i < 50; ++i)
        self->println("{}   {}   {}   {}",
                       log10_beta(grid[i]), denorm(grid[i]), std::pow(10.0, mean[i]), sd[i]);

    // ---- CSV dump for plot_mfbo.py -------------------------------------
    // Build each CSV once, then write it to a timestamped name (kept per run)
    // and to a stable "latest" name (what plot_mfbo.py reads by default).

    // runs: one row per actual high-fidelity run, in the order they finished.
    std::ostringstream runs_csv;
    runs_csv << "order,log10_beta,beta,mmd,log10_mmd,gp_pred_log10_mmd_before\n";
    for (size_t i = 0; i < trace.size(); ++i) {
        const run_point& p = trace[i];
        runs_csv << i << ',' << log10_beta(p.u) << ',' << denorm(p.u) << ','
                 << p.mmd << ',' << std::log10(std::max(p.mmd, 1e-6)) << ','
                 << p.pred_log10_before << '\n';
    }
    // pred: the GP's low-fidelity prediction curve + 1-sigma band (in log10 MMD).
    std::ostringstream pred_csv;
    pred_csv << "log10_beta,beta,pred_mmd,pred_log10_mmd,log10_mmd_sd\n";
    for (int i = 0; i < 50; ++i)
        pred_csv << log10_beta(grid[i]) << ',' << denorm(grid[i]) << ','
                 << std::pow(10.0, mean[i]) << ',' << mean[i] << ',' << sd[i] << '\n';

    const std::string runs_path = "mfbo_hf_runs_" + ts + ".csv";
    const std::string pred_path = "mfbo_hf_pred_" + ts + ".csv";
    for (const std::string& p : {runs_path, std::string("mfbo_hf_runs.csv")})
        std::ofstream(p) << runs_csv.str();
    for (const std::string& p : {pred_path, std::string("mfbo_hf_pred.csv")})
        std::ofstream(p) << pred_csv.str();
    self->println("\nwrote {}, {} (and mfbo_hf_runs.csv / mfbo_hf_pred.csv)", runs_path, pred_path);

    // Render the PNG straight away so a run always leaves a fresh graph
    // behind -- no need to invoke plot_mfbo.py by hand. Best-effort: if
    // matplotlib/the venv isn't there, the CSVs are still on disk.
    const std::string png_path = "mfbo_hf_" + ts + ".png";
    std::string cmd = "../.venv/bin/python3 ./plot_mfbo.py " + runs_path + " " + pred_path
                    + " " + png_path + " && cp " + png_path + " mfbo_hf.png";
    int rc = std::system(cmd.c_str());
    if (rc == 0)
        self->println("wrote {} (and mfbo_hf.png)", png_path);
    else
        self->println("plot_mfbo.py exited with {} -- CSVs are still there, plot them manually", rc);

    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::steady_clock::now() - wall_start).count();
    self->println("mfbo_hf run {} finished in {} s ({} full runs)", ts, secs, trace.size());
    // self is a scoped_actor (blocking) -- returning from caf_main ends the program.
}
CAF_MAIN(io::middleman, caf::id_block::mfbo_hf_project)
