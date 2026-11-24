// SPDX-License-Identifier: GPL-3.0-or-later
// Gaussian sampler adapted from Tao Zhang's Rust TVM, src/rng.rs at 1676eb89.
#ifndef TVM_RANDOM_H
#define TVM_RANDOM_H

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <random>
#include <string>

class Random {
public:
    void seed(std::mt19937_64::result_type seed) {
        engine_.seed(seed);
        hasSaved_ = false;
        saved_ = 0.;
    }

    std::mt19937_64& engine() { return engine_; }

    double normal() {
        if (hasSaved_) {
            hasSaved_ = false;
            return saved_;
        }
        for (;;) {
            double x = 2. * uniform() - 1.;
            double y = 2. * uniform() - 1.;
            double r2 = x*x + y*y;
            if (r2 > 0. && r2 <= 1.) {
                double factor = std::sqrt(-2. * std::log(r2) / r2);
                saved_ = x * factor;
                hasSaved_ = true;
                return y * factor;
            }
        }
    }

    bool save(std::ostream& out) const {
        return bool(out << "tvm_rng_1\n" << engine_ << '\n' << hasSaved_ << ' '
                        << std::setprecision(std::numeric_limits<double>::max_digits10) << saved_ << '\n');
    }

    bool load(std::istream& in) {
        std::string version;
        return bool(in >> version) && version == "tvm_rng_1" &&
               bool(in >> engine_ >> hasSaved_ >> saved_) && std::isfinite(saved_);
    }

private:
    double uniform() {
        return std::min(static_cast<double>(engine_()) / 18446744073709551616.0,
                        std::nextafter(1., 0.));
    }

    std::mt19937_64 engine_;
    bool hasSaved_ = false;
    double saved_ = 0.;
};

#endif
