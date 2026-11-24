// Finite differences of the C++ energies on a warped periodic cell.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "../../Run/Run.h"

int main() {
    Run run;
    Box box(&run);
    run.box_ = &box;
    for (int m = 0; m < 3; ++m) {
        box.size_[m] = 8;
        box.boundaryCondition_[m] = true;
    }
    const double positions[][3] = {
        {3.7, 0, 0}, {-3.4, -0.1, 0.03}, {-3.1, 0.6, 0}, {-3.6, 1.1, -0.02}, {3.8, 0.86, 0.01},
        {3.6, -0.03, 1}, {-3.3, 0, 1.2}, {-3.05, 0.73, 0.73}, {-3.52, 1.17, 1.12}, {3.83, 0.74, 1.27},
    };
    const std::vector<std::vector<int>> loops = {
        {0, 4, 3, 2, 1}, {5, 6, 7, 8, 9}, {0, 1, 6, 5}, {1, 2, 7, 6},
        {2, 3, 8, 7}, {3, 4, 9, 8}, {4, 0, 5, 9},
    };
    std::vector<std::unique_ptr<Vertex>> vertices;
    std::map<std::pair<int, int>, std::unique_ptr<Edge>> edges;
    std::vector<std::unique_ptr<Polygon>> faces;
    Cell cell(&run, 1);
    cell.type_ = 1;
    cell.kv_ = 10;
    cell.v0_ = 0.83;
    cell.s0_ = 4.7;
    run.cells_ = {&cell};
    Volume volume(&run);
    Interface interface(&run);
    for (const auto& position : positions) {
        vertices.emplace_back(new Vertex(&run, vertices.size()));
        std::copy(position, position + 3, vertices.back()->position_);
        run.vertices_.push_back(vertices.back().get());
    }
    for (const auto& loop : loops) {
        faces.emplace_back(new Polygon(&run, faces.size()));
        Polygon* face = faces.back().get();
        face->type_ = 1;
        face->cells_ = {&cell};
        run.polygons_.push_back(face);
        cell.polygons_.push_back(face);
        cell.polygonDirections_[face->id_] = true;
        for (size_t i = 0; i < loop.size(); ++i) {
            int a = loop[i], b = loop[(i + 1) % loop.size()];
            face->vertices_.push_back(run.vertices_[a]);
            auto& edge = edges[std::minmax(a, b)];
            if (!edge) {
                edge.reset(new Edge(&run, edges.size()));
                edge->vertices_ = {run.vertices_[a], run.vertices_[b]};
                run.edges_.push_back(edge.get());
            }
            face->edges_.push_back(edge.get());
        }
    }

    auto update = [&] {
        for (auto* edge : run.edges_) edge->update();
        for (auto* face : run.polygons_) face->updateCenter();
        volume.updateForces();
        interface.updateForces();
        volume.updateEnergy();
        interface.updateEnergy();
        return std::array<double, 2>{volume.energy_, interface.energy_};
    };

    const double h = 1e-6;
    for (bool exact : {false, true}) {
        run.exactFaceCenterForces_ = exact;
        std::array<double, 2> max_error = {0, 0};
        for (auto* vertex : run.vertices_) {
            for (int m = 0; m < 3; ++m) {
                update();
                std::array<double, 2> force = {vertex->volumeForce_[m], vertex->interfaceForce_[m]};
                double original = vertex->position_[m];
                vertex->position_[m] = original + h;
                auto plus = update();
                vertex->position_[m] = original - h;
                auto minus = update();
                vertex->position_[m] = original;
                for (int k = 0; k < 2; ++k) {
                    double diff = std::abs(force[k] + (plus[k] - minus[k]) / (2 * h));
                    if (!std::isfinite(diff)) throw std::runtime_error("nonfinite force error");
                    max_error[k] = std::max(max_error[k], diff);
                }
            }
        }
        std::cout << (exact ? "Exact" : "Legacy") << " maximum force error: volume "
                  << max_error[0] << ", area " << max_error[1] << '\n';
        if (exact && (max_error[0] > 2e-7 || max_error[1] > 2e-7))
            throw std::runtime_error("exact face-center forces disagree with the energy gradient");
        if (!exact && (max_error[0] < 1e-4 || max_error[1] < 1e-4))
            throw std::runtime_error("fixture does not distinguish the legacy approximation");
    }
}
