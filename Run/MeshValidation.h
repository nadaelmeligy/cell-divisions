// SPDX-License-Identifier: GPL-3.0-or-later
// CHANGED 2026-11-24 (new file, adapted from Tao Zhang's Rust TVM model.rs at commit 1676eb89):
// mesh checks used only with 'validate_topology on': every face one closed loop, every cell a closed shell,
// positive cell volumes, and (periodic boxes) cell volumes adding up to the box volume.
#ifndef TVM_MESH_VALIDATION_H
#define TVM_MESH_VALIDATION_H

#include <unordered_map>
#include <vector>

class Box;
class Cell;
class Polygon;
class Run;
class Vertex;

namespace MeshValidation {
struct CellGeometry {
    std::unordered_map<long, bool> directions;
    double volume;
};
std::vector<Vertex*> faceLoop(const Polygon& face);
void topology(const Run& run);
void positiveVolumes(const Run& run);
double signedVolume(const Cell& cell, Run& run);
CellGeometry proposedCell(const Cell& cell, Run& run);
}

#endif
