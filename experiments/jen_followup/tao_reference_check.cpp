// Compare the production C++ initialization and forces against Tao's Rust fixture.
#include "../../Run/Run.h"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int InitializeAll(std::string, std::string, std::string, Run*);

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Pass the path to Tao's reference.tsv");
        Run run;
        InitializeAll("sample.topo", "ECM.topo", "cellParameters.input", &run);
        run.volume_->updateForces();
        run.interface_->updateForces();
        run.volume_->updateEnergy();
        run.interface_->updateEnergy();
        Random rng;
        rng.seed(20261028);

        std::ifstream reference(argv[1]);
        if (!reference) throw std::runtime_error("Could not open reference fixture");
        std::unordered_map<long, const Vertex*> vertices;
        for (const auto* vertex : run.vertices_) vertices.emplace(vertex->id_, vertex);
        std::unordered_map<long, const Cell*> cells;
        for (const auto* cell : run.cells_) cells.emplace(cell->id_, cell);
        std::array<size_t, 4> counts{};
        double maxError = 0.;
        std::string kind;
        long id = -1;
        auto check = [&](double actual) {
            double expected;
            if (!(reference >> expected)) throw std::runtime_error("Incomplete reference row");
            double error = std::abs(actual - expected);
            if (!std::isfinite(error) || error >= 2e-10 * std::max(1., std::abs(expected))) {
                std::cerr << kind << " ID " << id << ": actual " << std::setprecision(17) << actual
                          << ", expected " << expected << '\n';
                throw std::runtime_error("Reference mismatch");
            }
            maxError = std::max(maxError, error);
        };
        while (reference >> kind) {
            if (kind == "energy") {
                ++counts[0];
                check(run.volume_->energy_);
                check(run.interface_->energy_);
                check(run.volume_->totalVolume_);
            } else if (kind == "vertex") {
                ++counts[1];
                reference >> id;
                const auto* vertex = vertices.at(id);
                for (double value : vertex->volumeForce_) check(value);
                for (double value : vertex->interfaceForce_) check(value);
            } else if (kind == "cell") {
                ++counts[2];
                reference >> id;
                check(cells.at(id)->volume_);
            } else if (kind == "normal") {
                ++counts[3];
                check(rng.normal());
            } else {
                throw std::runtime_error("Unexpected reference row " + kind);
            }
        }
        if (counts != std::array<size_t, 4>{1, 445, 64, 12})
            throw std::runtime_error("Unexpected reference row counts");
        std::cout << "Tao periodic64 reference passed (maximum absolute error " << maxError << ").\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
