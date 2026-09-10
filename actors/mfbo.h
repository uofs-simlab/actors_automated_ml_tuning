#ifndef MFBO_H
#define MFBO_H

// Multi-fidelity Bayesian optimization (MFBO): C++ port of the Python
// reference implementation. Ports test functions y/low/high, the GP
// posterior, and the acquisition loop 1:1; the only real deviation is the
// linear algebra, since numpy's np.linalg.solve(K, ...) becomes a Cholesky
// factorization here (K is symmetric positive definite once the 1e-6
// jitter is added, so Cholesky is enough and avoids adding an Eigen/BLAS
// dependency this project doesn't otherwise have).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfbo {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

// ---- Test functions ----

inline double y_fn(double x) {
    return std::exp(-((x - 2.0) * (x - 2.0) - 2.0))
         * std::cos((1.1 * x - 1.0) * (x - 2.2) * (0.8 * x - 4.0) * (1.15 * x - 5.0));
}

inline double low_fn(double x) {
    return y_fn(x) + 5.0 * std::sin(3.0 * x);
}

inline double high_fn(double x) {
    return y_fn(x);
}

// ---- Small linear algebra (Cholesky solve for SPD systems) ----

// In-place-free Cholesky decomposition: A = L.L^T, A must be symmetric
// positive definite (true here since K = kernel + 1e-6*I).
inline Mat cholesky(const Mat& A) {
    size_t n = A.size();
    Mat L(n, Vec(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (size_t k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) {
                if (sum <= 0.0) throw std::runtime_error("mfbo: kernel matrix not positive definite");
                L[i][j] = std::sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return L;
}

// Solve A x = b given A's Cholesky factor L.
inline Vec cholesky_solve(const Mat& L, const Vec& b) {
    size_t n = L.size();
    Vec z(n), x(n);
    for (size_t i = 0; i < n; ++i) {                 // forward: L z = b
        double sum = b[i];
        for (size_t k = 0; k < i; ++k) sum -= L[i][k] * z[k];
        z[i] = sum / L[i][i];
    }
    for (size_t ii = 0; ii < n; ++ii) {               // back: L^T x = z
        size_t i = n - 1 - ii;
        double sum = z[i];
        for (size_t k = i + 1; k < n; ++k) sum -= L[k][i] * x[k];
        x[i] = sum / L[i][i];
    }
    return x;
}

// Solve A X = B (B given as n x m, columns are right-hand sides), reusing
// one factorization -- this is what backs `v = np.linalg.solve(K, Kx.T)`.
inline Mat cholesky_solve_mat(const Mat& L, const Mat& B) {
    size_t n = L.size();
    size_t m = B.empty() ? 0 : B[0].size();
    Mat X(n, Vec(m, 0.0));
    for (size_t col = 0; col < m; ++col) {
        Vec b(n);
        for (size_t i = 0; i < n; ++i) b[i] = B[i][col];
        Vec x = cholesky_solve(L, b);
        for (size_t i = 0; i < n; ++i) X[i][col] = x[i];
    }
    return X;
}

// ---- MFBO ----

class MFBO {
public:
    using Fn = std::function<double(double)>;

    MFBO(Fn low, Fn high, std::pair<double, double> bounds)
        : rho(1.0), low_(low), high_(high), bounds_(bounds) {}

    double evaluate(double x, int fidelity) {
        double val;
        if (fidelity == 0) {
            val = low_(x);
            low_X_.push_back(x);
            low_y_.push_back(val);
        } else {
            val = high_(x);
            high_X_.push_back(x);
            high_y_.push_back(val);
        }
        return val;
    }

    static Mat kernel(const Vec& x1, const Vec& x2, double length = 1.0) {
        Mat K(x1.size(), Vec(x2.size()));
        for (size_t i = 0; i < x1.size(); ++i)
            for (size_t j = 0; j < x2.size(); ++j) {
                double d = x1[i] - x2[j];
                K[i][j] = std::exp(-0.5 * d * d / (length * length));
            }
        return K;
    }

    // GP posterior mean/std of observations (X, y) evaluated at `query`.
    static void gp(const Vec& X, const Vec& y, const Vec& query, Vec& mean_out, Vec& std_out) {
        size_t n = X.size();
        Mat K = kernel(X, X);
        // Diagonal nugget: 1e-6 for numerical stability plus a noise term
        // scaled to the spread of y, since a few-epoch / small-batch training
        // run is a genuinely noisy observation, not a near-exact one.
        double ybar = 0.0;
        for (double v : y) ybar += v;
        ybar /= (n ? double(n) : 1.0);
        double vy = 0.0;
        for (double v : y) vy += (v - ybar) * (v - ybar);
        vy /= (n > 1 ? double(n - 1) : 1.0);
        double nugget = 1e-6 + 1e-3 * vy;
        for (size_t i = 0; i < n; ++i) K[i][i] += nugget;

        Mat Kx = kernel(query, X); // m x n
        Mat L = cholesky(K);

        Vec alpha = cholesky_solve(L, y); // K @ alpha = y

        size_t m = query.size();
        mean_out.assign(m, 0.0);
        for (size_t q = 0; q < m; ++q) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += Kx[q][i] * alpha[i];
            mean_out[q] = s;
        }

        Mat KxT(n, Vec(m));
        for (size_t i = 0; i < n; ++i)
            for (size_t q = 0; q < m; ++q) KxT[i][q] = Kx[q][i];
        Mat v = cholesky_solve_mat(L, KxT); // n x m

        std_out.assign(m, 0.0);
        for (size_t q = 0; q < m; ++q) {
            double s = 0.0;
            for (size_t i = 0; i < n; ++i) s += Kx[q][i] * v[i][q];
            double variance = std::max(1.0 - s, 0.0);
            std_out[q] = std::sqrt(variance);
        }
    }

    void predict_high(const Vec& xs, Vec& mean_out, Vec& std_out) {
        size_t m = xs.size();
        Vec low_mean, low_std;

        // -----------------------
        // Single-fidelity shortcut
        // -----------------------
        // With no low-fidelity data at all (e.g. the mfbo_hf driver, which
        // only ever runs full-fidelity), the multi-fidelity path below would
        // synthesize low_std = 1 and fold it into every prediction, so
        // std_out could never drop below 1 even right on top of an
        // observation. That makes the acquisition unable to tell explored
        // from unexplored beta. Just return the plain GP posterior over the
        // high-fidelity points instead.
        if (low_X_.empty()) {
            if (high_X_.empty()) {
                mean_out.assign(m, 0.0);
                std_out.assign(m, 1.0);
            } else {
                gp(high_X_, high_y_, xs, mean_out, std_out);
            }
            return;
        }

        // -----------------------
        // Low-fidelity GP
        // -----------------------
        gp(low_X_, low_y_, xs, low_mean, low_std);

        // -----------------------
        // High-fidelity correction
        // -----------------------
        if (!high_X_.empty()) {
            // Low fidelity at the high-fidelity locations: use the low GP's
            // posterior mean (AR1 correction) instead of re-running low_(),
            // which would launch a full training run on every acquisition step.
            Vec low_at_X, low_at_X_std;
            gp(low_X_, low_y_, high_X_, low_at_X, low_at_X_std);

            // High - rho * Low
            Vec discrepancy(high_X_.size());
            for (size_t i = 0; i < high_X_.size(); ++i)
                discrepancy[i] = high_y_[i] - rho * low_at_X[i];

            Vec correction, correction_std;
            gp(high_X_, discrepancy, xs, correction, correction_std);

            mean_out.assign(m, 0.0);
            std_out.assign(m, 0.0);
            for (size_t q = 0; q < m; ++q) {
                mean_out[q] = rho * low_mean[q] + correction[q];
                std_out[q] = std::sqrt(rho * rho * low_std[q] * low_std[q]
                                        + correction_std[q] * correction_std[q]);
            }
        } else {
            mean_out.assign(m, 0.0);
            std_out.assign(m, 0.0);
            for (size_t q = 0; q < m; ++q) {
                mean_out[q] = rho * low_mean[q];
                std_out[q] = std::fabs(rho) * low_std[q];
            }
        }
    }

    //Select the x point which has best score (lowest mean and high std)
    double step() {
        const int N = 200;
        Vec xs(N);
        for (int i = 0; i < N; ++i)
            //Lay 200 evenly spaced out points
            xs[i] = bounds_.first + (bounds_.second - bounds_.first) * i / double(N - 1);

        //create resulting mean and std for all those 200 points
        Vec mean, std;
        predict_high(xs, mean, std);

        //Find the lowest value 
        double best = !high_y_.empty()
            ? *std::min_element(high_y_.begin(), high_y_.end())
            : *std::min_element(low_y_.begin(), low_y_.end());

        // Lower predicted value is good. Higher uncertainty is useful.
        size_t best_i = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < N; ++i) {
            double score = best - mean[i] + std[i];
            if (score > best_score) {
                best_score = score;
                best_i = static_cast<size_t>(i);
            }
        }
        return xs[best_i];
    }

    // Proposes up to `k` distinct candidate x's from the same
    // exploration/exploitation surface `step()` scores, without evaluating
    // any of them. This is what makes parallel workers possible: `step()`
    // alone only ever hands you one point at a time (the single best-scoring
    // one), which is fine for a sequential loop but leaves every worker but
    // one idle. Here we take the ranked list of candidates instead of just
    // the top one, and greedily keep points that are at least `min_gap`
    // apart -- from each other AND from every x already evaluated
    // (low_X_ / high_X_) -- so a batch doesn't collapse onto one spot the
    // model currently likes, and a later round doesn't re-run a point an
    // earlier round already measured. `min_gap` is in x units (here the
    // normalized [0,1] the optimizer works in). If the range is so
    // saturated that fewer than k points clear the history filter, a
    // second pass tops the batch up ignoring history, so a round always
    // has k jobs. Feed results back with record().
    std::vector<double> propose_batch(int k, double min_gap = 0.02) {
        const int N = 200;
        Vec xs(N);
        for (int i = 0; i < N; ++i)
            xs[i] = bounds_.first + (bounds_.second - bounds_.first) * i / double(N - 1);

        Vec mean, std;
        predict_high(xs, mean, std);

        double best = !high_y_.empty()
            ? *std::min_element(high_y_.begin(), high_y_.end())
            : *std::min_element(low_y_.begin(), low_y_.end());

        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        //Sort the indexes xs accroding to their score
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return (best - mean[a] + std[a]) > (best - mean[b] + std[b]);
        });

        std::vector<double> batch;

        auto far_enough = [&](double x, bool check_history) {
            for (double picked : batch)
                if (std::fabs(picked - x) < min_gap) return false;
            if (check_history) {
                for (double seen : high_X_)
                    if (std::fabs(seen - x) < min_gap) return false;
                for (double seen : low_X_)
                    if (std::fabs(seen - x) < min_gap) return false;
            }
            return true;
        };

        // Pass 1: space new points from each other and from all past evals.
        for (int idx : order) {
            if (far_enough(xs[idx], /*check_history=*/true)) batch.push_back(xs[idx]);
            if (static_cast<int>(batch.size()) == k) return batch;
        }
        // Pass 2: range nearly saturated -- top up ignoring history so the
        // round still gets k jobs (they will just sit closer to past points).
        for (int idx : order) {
            if (far_enough(xs[idx], /*check_history=*/false)) batch.push_back(xs[idx]);
            if (static_cast<int>(batch.size()) == k) break;
        }
        return batch;
    }

    // Records an observation obtained some other way (e.g. a CAF worker
    // actor's training run) without calling low_()/high_() -- use this
    // instead of evaluate() when the caller already has y.
    void record(double x, int fidelity, double y) {
        if (fidelity == 0) { low_X_.push_back(x); low_y_.push_back(y); }
        else                { high_X_.push_back(x); high_y_.push_back(y); }
    }

    // Runs the sweep, printing each evaluation the way the Python version
    // does; returns the best (x, y) found at the high fidelity.
    std::pair<double, double> run(int n = 30, bool verbose = true) {
        // Initial cheap exploration
        for (int i = 0; i < 8; ++i) {
            double x = bounds_.first + (bounds_.second - bounds_.first) * i / 7.0;
            evaluate(x, /*fidelity=*/0);
        }

        // Initial expensive evaluations
        evaluate(bounds_.first, /*fidelity=*/1);
        evaluate(bounds_.second, /*fidelity=*/1);

        // Bayesian optimization
        for (int i = 0; i < n; ++i) {
            double x = step();

            // Simple fidelity strategy: 3 cheap evaluations followed by 1 expensive.
            int fidelity = (i % 4 == 3) ? 1 : 0;

            double val = evaluate(x, fidelity);
            if (verbose) {
                std::printf("%2d: x=%.4f, fidelity=%d, y=%.4f\n", i, x, fidelity, val);
            }
        }

        size_t best_i = static_cast<size_t>(
            std::min_element(high_y_.begin(), high_y_.end()) - high_y_.begin());
        return {high_X_[best_i], high_y_[best_i]};
    }

    const Vec& low_X() const { return low_X_; }
    const Vec& low_y() const { return low_y_; }
    const Vec& high_X() const { return high_X_; }
    const Vec& high_y() const { return high_y_; }

    double rho;

private:
    Fn low_;
    Fn high_;
    std::pair<double, double> bounds_;

    Vec low_X_, low_y_;
    Vec high_X_, high_y_;
};

} // namespace mfbo

#endif // MFBO_H
