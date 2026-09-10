// Parallel multi-fidelity BO over beta, using the same worker-actor pattern
// as test.cc (one popen() per job, handed off to its own std::thread so the
// actor mailbox stays responsive) instead of test.cc's fixed --betas list.
//
// mfbo::MFBO::run() is inherently sequential: pick one point -> evaluate it
// -> refit the GP -> pick the next. That loop can't be parallelized as-is.
// What *can* be parallelized is each round of it: propose a whole batch of
// candidate betas at once (mfbo::MFBO::propose_batch), hand all of them to
// worker actors that train concurrently across --gpus GPUs, wait for the
// whole batch to finish, fold every result into the GP (mfbo::MFBO::record),
// then propose the next batch. So parallelism is within a round; rounds
// themselves are still sequential, same as sequential BO would be.
//
// Build:  make mfbo_caf   (needs the same CAF_PREFIX as test_actor)
// Run:    ./mfbo_caf --workers 4 --gpus 2

#include "config.h" // CAF, job registration helpers, the `config` CLI class
#include "mfbo.h"
#include <cmath>

// One job = one (beta, fidelity) pair to train + evaluate. `beta` here is
// the *normalized* u in [0,1] that mfbo::MFBO works in -- see denorm()
// below -- not a real beta value; the worker converts it before it ever
// touches latentflow1.py.
struct job_t {
    double u;
    int fidelity; // 0 = low (cheap probe), 1 = high (accurate)

    template <class Inspector>
    friend bool inspect(Inspector& f, job_t& j) {
        return f.object(j).fields(f.field("u", j.u), f.field("fidelity", j.fidelity));
    }
};

CAF_BEGIN_TYPE_ID_BLOCK(mfbo_caf_project, caf::first_custom_type_id)
CAF_ADD_TYPE_ID(mfbo_caf_project, (job_t))
CAF_END_TYPE_ID_BLOCK(mfbo_caf_project)

namespace {
constexpr double BETA_LO = 0.05, BETA_HI = 2.5;
double denorm(double u) { return BETA_LO + (BETA_HI - BETA_LO) * u; }
} // namespace

// ---- worker: unchanged shape from test.cc, just building the fidelity- ----
// ---- aware command line mfbo_demo.cc's run_latentflow() uses.         ----

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
                const char* epochs     = job.fidelity ? "5"   : "1";
                const char* fm_epochs  = job.fidelity ? "5"   : "1";
                const char* batch_size = job.fidelity ? "512" : "256";
                const char* suffix     = job.fidelity ? "_hi" : "_lo";

                std::ostringstream command;
                if (gpu_id >= 0) command << "CUDA_VISIBLE_DEVICES=" << gpu_id << " ";
                command << "../.venv/bin/python3 ./latentflow1.py --quiet"
                        << " --beta "       << beta
                        << " --epochs "     << epochs
                        << " --fm-epochs "  << fm_epochs
                        << " --batch-size " << batch_size
                        << " --tag-suffix " << suffix;

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

// ---- one round: spawn `jobs.size()` workers, hand each one job, block ----
// ---- until every result is back, fold each into the GP.               ----

void run_round(caf::scoped_actor& self, actor_system& system, int num_gpus,
                mfbo::MFBO& bo, const std::vector<job_t>& jobs) {
    int pending = static_cast<int>(jobs.size());
    std::vector<actor> workers;
    workers.reserve(jobs.size());

    for (const job_t& job : jobs) {
        actor worker = system.spawn(worker_actor, actor{self}, num_gpus);
        workers.push_back(worker);
        anon_mail(job).send(worker);
    }

    while (pending > 0) {
        self->receive(
            [&](std::string msg, job_t job, double mmd, actor /*worker*/) {
                if (msg != "result") return;
                // See the matching comment in mfbo_demo.cc's run_latentflow(): this is
                // a defensive floor for exact zero / round-off now that latentflow1.py's
                // compute_mmd() itself floors at 0.0, not a workaround for the old
                // shared-bandwidth bug that used to produce real negative MMD values.
                double y = std::log10(std::max(mmd, 1e-6));
                bo.record(job.u, job.fidelity, y);
                self->println("beta={} fidelity={} MMD={} (log10={})",
                               denorm(job.u), job.fidelity, mmd, y);
                --pending;
            });
    }
    for (actor& w : workers) anon_mail("quit").send(w);
}

// ---- driver ---------------------------------------------------------------

void caf_main(actor_system& system, const config& cfg) {
    caf::scoped_actor self{system};

    const int num_gpus = cfg.gpus;
    const int batch    = std::max(2, cfg.workers); // jobs run in parallel per round
    const int rounds    = 6;                        // acquisition rounds after the seed

    // mfbo::MFBO still wants low_()/high_() functors for its constructor,
    // but this driver never calls evaluate()/run() -- every observation
    // comes back through record() once a CAF worker finishes. Throwing
    // placeholders make it loud and obvious if that assumption is ever
    // broken by future edits.
    auto unused = [](double) -> double {
        throw std::logic_error("mfbo_caf: low_()/high_() should never be called directly; "
                                "results come from record() instead");
    };
    mfbo::MFBO bo(unused, unused, {0.0, 1.0});

    // Seed round: same shape as MFBO::run()'s warm-up (spread of cheap
    // probes + the two bounds at high fidelity), just dispatched together.
    std::vector<job_t> seed;
    for (int i = 0; i < batch; ++i)
        seed.push_back({batch > 1 ? i / double(batch - 1) : 0.5, /*fidelity=*/0});
    seed.push_back({0.0, /*fidelity=*/1});
    seed.push_back({1.0, /*fidelity=*/1});
    run_round(self, system, num_gpus, bo, seed);

    for (int r = 0; r < rounds; ++r) {
        std::vector<double> xs = bo.propose_batch(batch);
        std::vector<job_t> jobs;
        jobs.reserve(xs.size());
        for (size_t i = 0; i < xs.size(); ++i)
            jobs.push_back({xs[i], (i % 4 == 3) ? 1 : 0}); // 3 low : 1 high, same split as MFBO::run()
        self->println("--- round {} : {} jobs ---", r, jobs.size());
        run_round(self, system, num_gpus, bo, jobs);
    }

    const mfbo::Vec& hx = bo.high_X();
    const mfbo::Vec& hy = bo.high_y();
    size_t best_i = std::min_element(hy.begin(), hy.end()) - hy.begin();
    self->println("\nbest beta = {}  MMD = {}", denorm(hx[best_i]), std::pow(10.0, hy[best_i]));
    // self is a scoped_actor (blocking, not event-based) -- it has no
    // quit() of its own; caf_main returning is what ends the program.
}
CAF_MAIN(io::middleman, caf::id_block::mfbo_caf_project)
