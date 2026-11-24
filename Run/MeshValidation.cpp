// SPDX-License-Identifier: GPL-3.0-or-later
// Boundary-aware checks adapted from Tao Zhang's Rust TVM, src/model.rs at 1676eb89.
#include "MeshValidation.h"
#include "Run.h"

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace {
using EdgeKey = std::pair<long, long>;

EdgeKey key(const Vertex* a, const Vertex* b) {
    return std::minmax(a->id_, b->id_);
}

void require(bool condition, const std::string& message, long id) {
    if (!condition) throw std::runtime_error(message + " (ID " + std::to_string(id) + ")");
}

struct EdgeUse {
    size_t face;
    bool forward;
};

// The shell is connected, orientable and closed, including for boundary cells.
std::vector<int> winding(const Cell& cell, const std::vector<Polygon>& faces,
                         size_t anchor, int anchorDirection) {
    std::map<EdgeKey, std::vector<EdgeUse>> uses;
    std::set<long> vertices;
    for (size_t i = 0; i < faces.size(); ++i) {
        const auto& loop = faces[i].vertices_;
        for (size_t j = 0; j < loop.size(); ++j) {
            const Vertex* a = loop[j];
            const Vertex* b = loop[(j + 1) % loop.size()];
            vertices.insert(a->id_);
            uses[key(a, b)].push_back({i, a->id_ < b->id_});
        }
    }
    const long euler = static_cast<long>(vertices.size()) - static_cast<long>(uses.size()) +
                       static_cast<long>(faces.size());
    require(euler == 2, "Cell shell Euler relation differs from 2", cell.id_);
    std::vector<std::vector<std::pair<size_t, int>>> neighbors(faces.size());
    for (const auto& pair : uses) {
        const auto& incidence = pair.second;
        require(incidence.size() == 2, "Cell shell edge does not join exactly two faces", cell.id_);
        const auto& a = incidence[0];
        const auto& b = incidence[1];
        int sign = a.forward == b.forward ? -1 : 1;
        neighbors[a.face].push_back({b.face, sign});
        neighbors[b.face].push_back({a.face, sign});
    }
    std::vector<int> signs(faces.size(), 0);
    std::vector<size_t> pending = {anchor};
    signs[anchor] = anchorDirection;
    size_t visited = 0;
    while (!pending.empty()) {
        size_t i = pending.back();
        pending.pop_back();
        ++visited;
        for (const auto& neighbor : neighbors[i]) {
            int expected = signs[i] * neighbor.second;
            int& sign = signs[neighbor.first];
            if (!sign) {
                sign = expected;
                pending.push_back(neighbor.first);
            }
            require(sign == expected, "Cell shell winding is inconsistent", cell.id_);
        }
    }
    require(visited == faces.size(), "Cell shell is disconnected", cell.id_);
    return signs;
}

std::vector<Polygon> checkedFaces(const Cell& cell, Run* run = nullptr) {
    require(cell.polygons_.size() >= 4, "Cell has fewer than four faces", cell.id_);
    std::unordered_set<long> ids;
    std::vector<Polygon> faces;
    faces.reserve(cell.polygons_.size());
    for (auto* face : cell.polygons_) {
        require(face && ids.insert(face->id_).second, "Cell has a missing or repeated face", cell.id_);
        faces.emplace_back(run, face->id_);
        faces.back().edges_ = face->edges_;
        faces.back().vertices_ = MeshValidation::faceLoop(*face);
    }
    return faces;
}
}

std::vector<Vertex*> MeshValidation::faceLoop(const Polygon& face) {
    require(face.edges_.size() >= 3, "Face has fewer than three edges", face.id_);
    std::unordered_map<Vertex*, std::vector<Vertex*>> adjacency;
    std::set<EdgeKey> pairs;
    for (auto* edge : face.edges_) {
        require(edge && !edge->markToDelete_ && edge->vertices_.size() == 2 &&
                edge->vertices_[0] && edge->vertices_[1] && edge->vertices_[0] != edge->vertices_[1],
                "Face references an invalid edge", face.id_);
        auto* a = edge->vertices_[0];
        auto* b = edge->vertices_[1];
        require(pairs.insert(key(a, b)).second, "Face repeats an edge", face.id_);
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }
    for (const auto& entry : adjacency)
        require(entry.second.size() == 2, "Face is not a closed loop", face.id_);

    // Start from the same ordered edge as Polygon::updateVertices().
    Vertex* previous = face.edges_.back()->vertices_[0];
    Vertex* current = face.edges_.back()->vertices_[1];
    std::vector<Vertex*> loop = {previous};
    while (current != loop.front()) {
        require(loop.size() < face.edges_.size(), "Face loop fails to close", face.id_);
        loop.push_back(current);
        const auto& next = adjacency.at(current);
        Vertex* following = next[0] == previous ? next[1] : next[0];
        previous = current;
        current = following;
    }
    require(loop.size() == face.edges_.size(), "Face contains multiple loops", face.id_);
    return loop;
}

void MeshValidation::topology(const Run& run) {
    require(!run.vertices_.empty() && !run.edges_.empty() && !run.polygons_.empty(),
            "Tissue topology is empty", 0);
    std::unordered_set<Vertex*> vertices;
    std::unordered_set<long> ids;
    for (auto* v : run.vertices_) {
        require(v->id_ >= 0 && ids.insert(v->id_).second, "Invalid or duplicate vertex ID", v->id_);
        for (double x : v->position_)
            require(std::isfinite(x), "Nonfinite vertex coordinate", v->id_);
        vertices.insert(v);
    }
    ids.clear();
    std::set<EdgeKey> endpoints;
    std::unordered_map<Edge*, size_t> faceIncidence;
    for (auto* edge : run.edges_) {
        require(edge->id_ >= 0 && ids.insert(edge->id_).second, "Invalid or duplicate edge ID", edge->id_);
        require(edge->vertices_.size() == 2 && vertices.count(edge->vertices_[0]) &&
                vertices.count(edge->vertices_[1]) && edge->vertices_[0] != edge->vertices_[1] &&
                endpoints.insert(key(edge->vertices_[0], edge->vertices_[1])).second,
                "Edge has missing, identical or duplicate endpoints", edge->id_);
        faceIncidence.emplace(edge, 0);
    }
    ids.clear();
    std::unordered_map<Polygon*, std::vector<long>> owners;
    for (auto* face : run.polygons_) {
        require(face->id_ >= 0 && ids.insert(face->id_).second, "Invalid or duplicate face ID", face->id_);
        for (auto* edge : face->edges_) {
            auto it = faceIncidence.find(edge);
            require(it != faceIncidence.end(), "Face references an absent edge", face->id_);
            ++it->second;
        }
        faceLoop(*face);
        owners.emplace(face, std::vector<long>{});
    }
    for (const auto& entry : faceIncidence)
        require(entry.second == 2 || entry.second == 3, "Edge does not join two or three faces", entry.first->id_);

    ids.clear();
    for (auto* cell : run.cells_) {
        if (cell->type_ < 0) continue;
        require(cell->id_ >= 0 && ids.insert(cell->id_).second, "Invalid or duplicate cell ID", cell->id_);
        for (auto* face : cell->polygons_) {
            auto it = owners.find(face);
            require(it != owners.end(), "Cell references an absent face", cell->id_);
            it->second.push_back(cell->id_);
        }
        winding(*cell, checkedFaces(*cell), 0, 1);
    }
    require(!ids.empty(), "Tissue has no material cells", 0);
    std::set<std::pair<long, long>> pairs;
    for (const auto& entry : owners) {
        const auto& cells = entry.second;
        // Open boundary faces belong to one material cell plus the implicit exterior.
        require(cells.size() == 1 || cells.size() == 2, "Face does not have one or two material owners",
                entry.first->id_);
        if (cells.size() == 2)
            require(pairs.insert(std::minmax(cells[0], cells[1])).second,
                    "Two material cells share multiple faces", entry.first->id_);
    }
}

void MeshValidation::positiveVolumes(const Run& run) {
    double sum = 0.;
    for (const auto* cell : run.cells_)
        if (cell->type_ >= 0) {
            require(std::isfinite(cell->volume_) && cell->volume_ > 0.,
                    "Nonpositive or nonfinite cell volume", cell->id_);
            sum += cell->volume_;
        }
    if (run.emptySpace_->polygons_.empty() &&
            run.box_->boundaryCondition_[0] && run.box_->boundaryCondition_[1] && run.box_->boundaryCondition_[2]) {
        const double boxVolume = run.box_->size_[0] * run.box_->size_[1] * run.box_->size_[2];
        require(std::isfinite(sum) && std::abs(sum / boxVolume - 1.) <= 1e-8,
                "Periodic tissue volume differs from the box volume", 0);
    }
}

double MeshValidation::signedVolume(const Cell& cell, Run& run) {
    Cell geometry(&run, cell.id_);
    geometry.polygons_ = cell.polygons_;
    geometry.polygonDirections_ = cell.polygonDirections_;
    geometry.updateVolume();
    return geometry.volume_;
}

MeshValidation::CellGeometry MeshValidation::proposedCell(const Cell& cell, Run& run) {
    auto faces = checkedFaces(cell, &run);
    int anchorDirection = 0;
    size_t anchor = 0;
    for (; anchor < faces.size() && !anchorDirection; ++anchor) {
        auto* previous = cell.polygons_[anchor];
        auto direction = cell.polygonDirections_.find(previous->id_);
        if (direction == cell.polygonDirections_.end()) continue;
        const auto& oldLoop = previous->vertices_;
        const auto& newLoop = faces[anchor].vertices_;
        for (size_t j = 0; j < oldLoop.size() && !anchorDirection; ++j) {
            Vertex* a = oldLoop[j];
            Vertex* b = oldLoop[(j + 1) % oldLoop.size()];
            for (size_t k = 0; k < newLoop.size(); ++k) {
                Vertex* x = newLoop[k];
                Vertex* y = newLoop[(k + 1) % newLoop.size()];
                if (a == x && b == y) anchorDirection = direction->second ? 1 : -1;
                if (a == y && b == x) anchorDirection = direction->second ? -1 : 1;
                if (anchorDirection) break;
            }
        }
    }
    require(anchorDirection != 0, "Cannot retain the old cell winding in proposal", cell.id_);
    auto signs = winding(cell, faces, anchor - 1, anchorDirection);
    Cell proposal(&run, cell.id_);
    for (size_t i = 0; i < faces.size(); ++i) {
        auto& face = faces[i];
        for (const auto* edge : face.edges_)
            require(std::isfinite(edge->length_) && edge->length_ > 0.,
                    "Proposed face has a degenerate edge", face.id_);
        face.updateCenter();
        face.updateArea();
        require(std::isfinite(face.area_) && face.area_ > 0., "Proposed face has invalid area", face.id_);
        proposal.polygons_.push_back(&face);
        proposal.polygonDirections_.emplace(face.id_, signs[i] > 0);
    }
    proposal.updateVolume();
    require(std::isfinite(proposal.volume_) && proposal.volume_ > 0.,
            "Proposed cell has nonpositive or nonfinite volume", cell.id_);
    return {std::move(proposal.polygonDirections_), proposal.volume_};
}
