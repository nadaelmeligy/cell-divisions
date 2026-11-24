/* ---------------------------------------------------------------------------------
 * Copyright 2021-2023 Tao Zhang
 *
 * This file is part of TVM.
 *
 * TVM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation,
 * either version 3 of the License,
 * or (at your option) any later version.
 *
 * TVM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TVM. If not, see <https://www.gnu.org/licenses/>.
 *
 * Author: Tao Zhang @ Shanghai Jiao Tong University, zhangtao.scholar@sjtu.edu.cn
 * Coauthor: Jennifer Schwarz @ Syracuse University, jschwarz@physics.syr.edu
 * ---------------------------------------------------------------------------------
 */

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <unordered_map>

#include "Reconnection.h"
#include "../Run/MeshValidation.h"
#include <set>
#include <stdexcept>

// CHANGED 2026-11-24 (Tao Zhang's Rust TVM, src at commit 1676eb89): a proposed neighbour swap is made reversible. With
// 'validate_topology on' the edited cells are checked (closed shells, positive volumes, volume kept) before
// the old mesh is deleted; a failing swap is undone and counted as rejected. Off (default): no change.
namespace {
// Preserve the existing local rewrites, but make the proposed edits reversible
// until all affected cells pass the checks used for Tao's Rust proposals.
class ProposedSwap {
public:
    ProposedSwap(Run& run, std::vector<Cell*> cells, std::vector<Polygon*> faces)
        : run_(run), vertices_(run.vertices_.size()), edges_(run.edges_.size()),
          faces_(run.polygons_.size()), vertexId_(run.count_vertices_),
          edgeId_(run.count_edges_), faceId_(run.count_polygons_) {
        if (!run_.validateTopology_) return;
        for (auto* cell : cells) {
            oldCells_.emplace_back(cell, cell->polygons_);
            if (cell->type_ < 0)
                includesExterior_ = true;
            else
                oldVolume_ += MeshValidation::signedVolume(*cell, run_);
        }
        for (auto* face : faces) {
            oldFaces_.emplace_back(face, face->edges_);
            for (auto* edge : face->edges_) triangleCounts_.emplace(edge, edge->triangle_count_);
        }
    }

    ~ProposedSwap() {
        if (accepted_ || !run_.validateTopology_) return;
        for (auto& item : oldCells_) item.first->polygons_.swap(item.second);
        for (auto& item : oldFaces_) item.first->edges_.swap(item.second);
        for (const auto& item : triangleCounts_) item.first->triangle_count_ = item.second;
        while (run_.polygons_.size() > faces_) {
            delete run_.polygons_.back();
            run_.polygons_.pop_back();
        }
        while (run_.edges_.size() > edges_) {
            delete run_.edges_.back();
            run_.edges_.pop_back();
        }
        while (run_.vertices_.size() > vertices_) {
            delete run_.vertices_.back();
            run_.vertices_.pop_back();
        }
        run_.count_vertices_ = vertexId_;
        run_.count_edges_ = edgeId_;
        run_.count_polygons_ = faceId_;
    }

    bool accept(const char* transition, long id) {
        if (run_.validateTopology_) {
            try {
                std::set<std::pair<long, long>> pairs;
                for (size_t i = edges_; i < run_.edges_.size(); ++i) {
                    const Edge& edge = *run_.edges_[i];
                    auto endpoints = std::minmax(edge.vertices_[0]->id_, edge.vertices_[1]->id_);
                    if (!pairs.insert(endpoints).second || !(edge.length_ > 0.) || !std::isfinite(edge.length_))
                        throw std::runtime_error("duplicate or degenerate proposed edge");
                }

                double volume = 0.;
                std::vector<std::pair<Cell*, MeshValidation::CellGeometry>> cells;
                for (const auto& item : oldCells_) {
                    if (item.first->type_ < 0) continue;
                    auto geometry = MeshValidation::proposedCell(*item.first, run_);
                    volume += geometry.volume;
                    cells.emplace_back(item.first, std::move(geometry));
                }
                if (!includesExterior_ && std::abs(volume - oldVolume_) > 1e-8 * std::max(1., std::abs(oldVolume_)))
                    throw std::runtime_error("proposed cells do not conserve volume");
                std::vector<std::pair<Polygon*, std::vector<Vertex*>>> loops;
                for (const auto& item : oldFaces_)
                    loops.emplace_back(item.first, MeshValidation::faceLoop(*item.first));
                for (size_t i = faces_; i < run_.polygons_.size(); ++i) {
                    auto* face = run_.polygons_[i];
                    loops.emplace_back(face, MeshValidation::faceLoop(*face));
                }
                for (auto& item : loops) {
                    item.first->vertices_.swap(item.second);
                    item.first->updateCenter();
                }
                for (auto& item : cells) item.first->polygonDirections_.swap(item.second.directions);
            } catch (const std::runtime_error& error) {
                ++run_.reconnection_->rejectedProposals_;
                run_.verboseReconnection_ << "Rejected " << transition << " at " << run_.simulation_time_
                    << ", ID " << id << ": " << error.what() << '\n';
                return false;
            }
        }
        accepted_ = true;
        return true;
    }

    ProposedSwap(const ProposedSwap&) = delete;
    ProposedSwap& operator=(const ProposedSwap&) = delete;

private:
    Run& run_;
    size_t vertices_, edges_, faces_;
    long vertexId_, edgeId_, faceId_;
    bool accepted_ = false;
    bool includesExterior_ = false;
    double oldVolume_ = 0.;
    std::vector<std::pair<Cell*, std::vector<Polygon*>>> oldCells_;
    std::vector<std::pair<Polygon*, std::vector<Edge*>>> oldFaces_;
    std::unordered_map<Edge*, int> triangleCounts_;
};
}

using namespace std;

Reconnection::Reconnection(Run * run) {
    run_ = run;
    Lth_ = 1.0e-3;
//    Lth_ = 0.04;
    count_IH_ = 0;
    count_HI_ = 0;
    verbose_ = false;
}

int     Reconnection::start() {
    run_->updateGeoinfo();
    // set edge candidate
    for (auto edge : run_->edges_) {
        if (edge->length_ < Lth_) {
            edge->candidate_ = true;
        } else {
            edge->candidate_ = false;
        }
        edge->triangle_count_ = 0;
    }
    // set edge triangle_count_
    for (auto polygon : run_->polygons_) {
        if (polygon->edges_.size() == 3) {
            for (auto edge : polygon->edges_) {
                edge->triangle_count_ = edge->triangle_count_ + 1;
            }
        }
    }
    // look for H type triangle candidates
    std::vector<Polygon *> triangleCandidates;
    for (auto polygon : run_->polygons_) {
        if (polygon->checkH()) {
            triangleCandidates.push_back(polygon);
        }
    }

    // I -> H reconnection
    std::vector<Edge *> tmp_edges = run_->edges_;
    for (auto edge : tmp_edges) {
        if (edge->markToDelete_) {
            continue;
        }
        if (!edge->checkI()) {
            continue;
        }
        I_H(edge);
    }

    // H -> I reconnection
    for (auto polygon : triangleCandidates) {
        if (!polygon->checkH()) {
            continue;
        }
        H_I(polygon);
    }

    // delete marked edges
//    printf("%ld %ld ", tmp_edges.size(), run_->edges_.size());
    tmp_edges.clear();
    tmp_edges = run_->edges_;
    run_->edges_.clear();
    for (auto edge: tmp_edges) {
        if (edge->markToDelete_) {
            delete edge;
        } else {
            run_->edges_.push_back(edge);
        }
    }
//    printf("%ld\n", run_->edges_.size());

    // update geometry and topology information
    run_->updateGeoinfo();
    run_->updateVertexCells();
    run_->updateEdgeCells();
    if (run_->validateTopology_) {
        // Accepted proposals already set the affected face loops and cell winding.
        run_->updatePolygonCells();
        run_->volume_->updateExteriorDirections();
    } else {
        run_->volume_->updatePolygonDirections();
    }
    run_->updateEmptyCells();

    return 0;
}

int Reconnection::I_H(Edge * edge) {
    // locate two vertices: 10 and 11
    Vertex * v10 = edge->vertices_[0];
    Vertex * v11 = edge->vertices_[1];

    // locate five cells: 10-123, 11-456, 1011-1245, 1011-2356, 1011-1346
    // if using finite boundary condition, cellTop_ and cellBottom_ will be the same,
    // and this part should be modified
    if (v10->cells_.size() != 4) {
        if (verbose_) {
            printf("Topology Warning: vertex %ld has %ld neighboring cells", v10->id_, v10->cells_.size());
            for (auto tCell : v10->cells_) {
                printf(" %ld", tCell->id_);
            }
            printf("\n");
        }
//        for (auto cell : v10->cells_) {
//            printf(" %ld", cell->id_);
//        }
//        printf("\n");
        return 1;
    }
    if (v11->cells_.size() != 4) {
        if (verbose_) {
            printf("Topology Warning: vertex %ld has %ld neighboring cells", v11->id_, v11->cells_.size());
            for (auto tCell : v11->cells_) {
                printf(" %ld", tCell->id_);
            }
            printf("\n");
        }
        return 1;
    }
    Cell * c123 = NULL;
    Cell * c456 = NULL;
    Cell * c1245 = NULL;
    Cell * c2356 = NULL;
    Cell * c1346 = NULL;
    std::vector<Cell *> sideCells;
    for (auto cell10 : v10->cells_) {
        if (std::find(v11->cells_.begin(), v11->cells_.end(), cell10) != v11->cells_.end()) {
            sideCells.push_back(cell10);
        } else {
            c123 = cell10;
        }
    }
    if (sideCells.size() != 3) {
        if (verbose_) {
            printf("Topology Warning: edge %ld has %ld neighboring side cells\n", edge->id_, sideCells.size());
        }
        return 1;
    }
    c1245 = sideCells[0];
    c2356 = sideCells[1];
    c1346 = sideCells[2];
    for (auto cell11 : v11->cells_) {
        if (std::find(v10->cells_.begin(), v10->cells_.end(), cell11) == v10->cells_.end()) {
            c456 = cell11;
            break;
        }
    }
    if (c456 == NULL) {
        printf("Topology Error: edge %ld has 0 neighboring bottom cells\n", edge->id_);
        exit(1);
    }

    // check if top/bottom pair of cells already have common polygon
    if (commonPolygon(c123, c456) != NULL) {
//        c123->logPolygons("c123");
//        c456->logPolygons("c456");
        if (verbose_) {
            printf("Topology warning: cell %ld and %ld already have common polygon before I->H reconnection\n",
                   c123->id_, c456->id_);
        }
        return 1;
    }

    std::vector<Cell *> tmpCells = {c123, c456, c1245, c2356, c1346};
    if (verbose_) {
        dumpCells(true, true, tmpCells);
    }

    if (!checkCells(tmpCells)) {
        return 1;
    }

    // locate three side polygons: 1-1011-4, 2-1011-5, 3-1011-6
    Polygon * p14 = commonPolygon(c1245, c1346);
    Polygon * p25 = commonPolygon(c2356, c1245);
    Polygon * p36 = commonPolygon(c1346, c2356);
    // locate three polygons: 10-12, 10-23, 10-13
    Polygon * p12 = commonPolygon(c1245, c123);
    Polygon * p23 = commonPolygon(c2356, c123);
    Polygon * p13 = commonPolygon(c1346, c123);
    // locate three polygons: 11-45, 11-56, 11-46
    Polygon * p45 = commonPolygon(c1245, c456);
    Polygon * p56 = commonPolygon(c2356, c456);
    Polygon * p46 = commonPolygon(c1346, c456);
    if (p14 == NULL || p25 == NULL || p36 == NULL ||
        p12 == NULL || p23 == NULL || p13 == NULL ||
        p45 == NULL || p56 == NULL || p46 == NULL) {
        if (verbose_) {
            printf("Topology Warning: no common polygon or more than one common polygon\n");
        }
        return 1;
    }
    // check if top/bottom pair of polygons already have common edge
    if (commonEdge(p12, p45) != NULL) {
//        p12->logEdges("p12");
//        p45->logEdges("p45");
        if (verbose_) {
            printf("Topology Warning: polygon %ld and %ld already have common edge before I->H reconnection\n",
                   p12->id_, p45->id_);
        }
        return 1;
    }
    if (commonEdge(p23, p56) != NULL) {
//        p23->logEdges("p23");
//        p56->logEdges("p56");
        if (verbose_) {
            printf("Topology Warning: polygon %ld and %ld already have common edge before I->H reconnection\n",
                   p23->id_, p56->id_);
        }
        return 1;
    }
    if (commonEdge(p13, p46) != NULL) {
//        p13->logEdges("p13");
//        p46->logEdges("p46");
        if (verbose_) {
            printf("Topology Warning: polygon %ld and %ld already have common edge before I->H reconnection\n",
                   p13->id_, p46->id_);
        }
        return 1;
    }
    // locate six edges: 10-1, 10-2, 10-3, 11-4, 11-5, 11-6
    Edge * e1 = commonEdge(p12, p13);
    Edge * e2 = commonEdge(p23, p12);
    Edge * e3 = commonEdge(p13, p23);
    Edge * e4 = commonEdge(p45, p46);
    Edge * e5 = commonEdge(p56, p45);
    Edge * e6 = commonEdge(p46, p56);
    if (e1 == NULL || e2 == NULL || e3 == NULL ||
        e4 == NULL || e5 == NULL || e6 == NULL) {
        if (verbose_) {
            printf("Topology Warning: no common edge or more than one common edge\n");
        }
//        run_->updateVertexEdges();
//        v10->logEdges("v10");
//        v11->logEdges("v11");
//        v10->logCells("v10");
//        v11->logCells("v11");
//        c123->logPolygons("c123");
//        c456->logPolygons("c456");
//        c1245->logPolygons("c1245");
//        c1346->logPolygons("c1346");
//        c2356->logPolygons("c2356");
//        printf("edge %ld\n",edge->id_);
//        printf("e1 %ld\n",e1->id_);
//        printf("e2 %ld\n",e2->id_);
//        printf("e3 %ld\n",e3->id_);
//        printf("e4 %ld\n",e4->id_);
//        printf("e5 %ld\n",e5->id_);
//        printf("e6 %ld\n",e6->id_);
//        p14->logEdges("p14");
//        p25->logEdges("p25");
//        p36->logEdges("p36");
//        p12->logEdges("p12");
//        p23->logEdges("p23");
//        p13->logEdges("p13");
//        p45->logEdges("p45");
//        p56->logEdges("p56");
//        p46->logEdges("p46");
//        run_->updatePolygonVertices();
        return 1;
    }
    // locate six vertices: 1, 2, 3, 4, 5, 6
    Vertex * v1 = e1->otherVertex(v10);
    Vertex * v2 = e2->otherVertex(v10);
    Vertex * v3 = e3->otherVertex(v10);
    Vertex * v4 = e4->otherVertex(v11);
    Vertex * v5 = e5->otherVertex(v11);
    Vertex * v6 = e6->otherVertex(v11);
    if (v1 == NULL || v2 == NULL || v3 == NULL ||
        v4 == NULL || v5 == NULL || v6 == NULL) {
        printf("Topology Error: no common vertex\n");
        exit(1);
    }

    ProposedSwap proposal(*run_, {c123, c456, c1245, c2356, c1346},
                          {p14, p25, p36, p12, p23, p13, p45, p56, p46});
    // create vertices 7, 8, 9
    Vertex * v7 = new Vertex(run_, run_->count_vertices_);
    run_->count_vertices_ = run_->count_vertices_ + 1;
    run_->vertices_.push_back(v7);
    Vertex * v8 = new Vertex(run_, run_->count_vertices_);
    run_->count_vertices_ = run_->count_vertices_ + 1;
    run_->vertices_.push_back(v8);
    Vertex * v9 = new Vertex(run_, run_->count_vertices_);
    run_->count_vertices_ = run_->count_vertices_ + 1;
    run_->vertices_.push_back(v9);
    ////////// compute positions of vertices 7, 8, 9 ///////////////
    // r0: midpoint position of edge
    // uT: unit direction vector of edge
    double r0[3];
    double uT[3];
    for (int m = 0; m < 3; m++) {
        r0[m] = edge->center_[m];
        uT[m] = edge->vv_[m];
    }
    double uTL = sqrt(uT[0]*uT[0] + uT[1]*uT[1] + uT[2]*uT[2]);
    for (int m = 0; m < 3; m++) {
        uT[m] = uT[m]/uTL;
    }
    // compute vectors indicating orientation of neighboring polygons
    double w7[3];
    double w8[3];
    double w9[3];
    double r01[3];
    double r02[3];
    double r03[3];
    double r04[3];
    double r05[3];
    double r06[3];
    computeDirection(r0, v1->position_, r01);
    computeDirection(r0, v2->position_, r02);
    computeDirection(r0, v3->position_, r03);
    computeDirection(r0, v4->position_, r04);
    computeDirection(r0, v5->position_, r05);
    computeDirection(r0, v6->position_, r06);
    for (int m = 0; m < 3; m++) {
        w7[m] = 0.5*(r01[m] + r04[m]);
        w8[m] = 0.5*(r02[m] + r05[m]);
        w9[m] = 0.5*(r03[m] + r06[m]);
    }
    // compute vectors indicating orientation of vertices 7, 8, 9
    double wv7[3];
    double wv8[3];
    double wv9[3];
    double dP_w7_uT = 0.;
    double dP_w8_uT = 0.;
    double dP_w9_uT = 0.;
    for (int m = 0; m < 3; m++) {
        dP_w7_uT += w7[m]*uT[m];
        dP_w8_uT += w8[m]*uT[m];
        dP_w9_uT += w9[m]*uT[m];
    }
    for (int m = 0; m < 3; m++) {
        wv7[m] = w7[m] - dP_w7_uT*uT[m];
        wv8[m] = w8[m] - dP_w8_uT*uT[m];
        wv9[m] = w9[m] - dP_w9_uT*uT[m];
    }
    double Lmax = 0.;
    std::vector<double *> tmp_wv = {wv7, wv8, wv9};
    for (auto wv : tmp_wv) {
        double wvL = sqrt(wv[0]*wv[0] + wv[1]*wv[1] + wv[2]*wv[2]);
        if (wvL > Lmax) {
            Lmax = wvL;
        }
    }
    // compute position of vertices 7, 8, 9
    for (int m = 0; m < 3; m++) {
        v7->position_[m] = r0[m] + Lth_/Lmax*wv7[m];
        v8->position_[m] = r0[m] + Lth_/Lmax*wv8[m];
        v9->position_[m] = r0[m] + Lth_/Lmax*wv9[m];
    }
    run_->box_->resetPosition(v7->position_);
    run_->box_->resetPosition(v8->position_);
    run_->box_->resetPosition(v9->position_);
    ////////// compute positions of vertices 7, 8, 9 done  /////////
    if (verbose_) {
        std::vector<Vertex *> tmpVertices = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11};
        dumpVertices(tmpVertices);
    }
    // associate cells to vertices 7,8,9
    v7->cells_.push_back(c1245);
    v7->cells_.push_back(c1346);
    v7->cells_.push_back(c123);
    v7->cells_.push_back(c456);
    v8->cells_.push_back(c1245);
    v8->cells_.push_back(c2356);
    v8->cells_.push_back(c123);
    v8->cells_.push_back(c456);
    v9->cells_.push_back(c2356);
    v9->cells_.push_back(c1346);
    v9->cells_.push_back(c123);
    v9->cells_.push_back(c456);
    // create edges 78, 79, 89
    Edge * e78 = run_->addEdge(v7, v8);
    Edge * e79 = run_->addEdge(v7, v9);
    Edge * e89 = run_->addEdge(v8, v9);
    // create polygon 789
    Polygon * p789 = new Polygon(run_, run_->count_polygons_);
    run_->count_polygons_ = run_->count_polygons_ + 1;
    run_->polygons_.push_back(p789);
    p789->edges_.push_back(e78);
    p789->edges_.push_back(e79);
    p789->edges_.push_back(e89);
    // associate p789 to c123 and c456
    c123->polygons_.push_back(p789);
    c456->polygons_.push_back(p789);
    // create edges: 71,82,93,74,85,96
    Edge * e71 = run_->addEdge(v7, v1);
    Edge * e82 = run_->addEdge(v8, v2);
    Edge * e93 = run_->addEdge(v9, v3);
    Edge * e74 = run_->addEdge(v7, v4);
    Edge * e85 = run_->addEdge(v8, v5);
    Edge * e96 = run_->addEdge(v9, v6);
    // update side polygon 1-1011-4
    p14->shrink(edge);
    p14->shrink(e1);
    p14->shrink(e4);
    p14->expand(e71);
    p14->expand(e74);
    // update side polygon 2-1011-5
    p25->shrink(edge);
    p25->shrink(e2);
    p25->shrink(e5);
    p25->expand(e82);
    p25->expand(e85);
    // update side polygon 3-1011-6
    p36->shrink(edge);
    p36->shrink(e3);
    p36->shrink(e6);
    p36->expand(e93);
    p36->expand(e96);
    // update polygon 10-12
    p12->shrink(e1);
    p12->shrink(e2);
    p12->expand(e78);
    p12->expand(e71);
    p12->expand(e82);
    // update polygon 10-23
    p23->shrink(e2);
    p23->shrink(e3);
    p23->expand(e89);
    p23->expand(e82);
    p23->expand(e93);
    // update polygon 10-13
    p13->shrink(e1);
    p13->shrink(e3);
    p13->expand(e79);
    p13->expand(e71);
    p13->expand(e93);
    // update polygon 11-45
    p45->shrink(e4);
    p45->shrink(e5);
    p45->expand(e78);
    p45->expand(e74);
    p45->expand(e85);
    // update polygon 11-56
    p56->shrink(e5);
    p56->shrink(e6);
    p56->expand(e89);
    p56->expand(e85);
    p56->expand(e96);
    // update polygon 11-46
    p46->shrink(e4);
    p46->shrink(e6);
    p46->expand(e79);
    p46->expand(e74);
    p46->expand(e96);

    if (!proposal.accept("I->H", edge->id_)) return 1;

    // delete vertices 10 and 11
    run_->deleteVertex(v10);
    run_->deleteVertex(v11);
    // delete edges
    run_->deleteEdge(edge);
    run_->deleteEdge(e1);
    run_->deleteEdge(e2);
    run_->deleteEdge(e3);
    run_->deleteEdge(e4);
    run_->deleteEdge(e5);
    run_->deleteEdge(e6);

//    // debug: turn off reconnection for all related edges
//    std::vector<Polygon *> tmp_polygons = {p14, p25, p36, p12, p23, p13, p45, p56, p46, p789};
//    for (auto polygon : tmp_polygons) {
//        for (auto edge : polygon->edges_) {
//            edge->candidate_ = false;
//        }
//    }

    if (verbose_) {
        std::vector<Cell *> tmpCells = {c123, c456, c1245, c2356, c1346};
        dumpCells(false, true, tmpCells);
    }

    count_IH_ += 1;
    ++acceptedProposals_;

    return 0;
}

int Reconnection::H_I(Polygon * polygon) {
    // locate two vertices: 7, 8, 9
    polygon->updateVertices();
    Vertex * v7 = polygon->vertices_[0];
    Vertex * v8 = polygon->vertices_[1];
    Vertex * v9 = polygon->vertices_[2];

    // locate five cells: 10-123, 11-456, 1011-1245, 1011-2356, 1011-1346
    // if using finite boundary condition, cellTop_ and cellBottom_ will be the same,
    // and this part should be modified
    if (v7->cells_.size() != 4) {
        if (verbose_) {
            printf("Topology Warning: vertex %ld has %ld neighboring cells", v7->id_, v7->cells_.size());
        }
//        for (auto cell : v7->cells_) {
//            printf(" %ld", cell->id_);
//        }
//        printf("\n");
        return 1;
    }
    if (v8->cells_.size() != 4) {
        if (verbose_) {
            printf("Topology Warning: vertex %ld has %ld neighboring cells", v8->id_, v8->cells_.size());
        }
//        for (auto cell : v8->cells_) {
//            printf(" %ld", cell->id_);
//        }
//        printf("\n");
        return 1;
    }
    if (v9->cells_.size() != 4) {
        if (verbose_) {
            printf("Topology Warning: vertex %ld has %ld neighboring cells", v9->id_, v9->cells_.size());
        }
//        for (auto cell : v9->cells_) {
//            printf(" %ld", cell->id_);
//        }
//        printf("\n");
        return 1;
    }
    Cell * c123 = NULL;
    Cell * c456 = NULL;
    Cell * c1245 = NULL;
    Cell * c2356 = NULL;
    Cell * c1346 = NULL;
    std::vector<Cell *> topBottomCells;
    for (auto cell7 : v7->cells_) {
        if (std::find(cell7->polygons_.begin(), cell7->polygons_.end(), polygon) != cell7->polygons_.end()) {
            topBottomCells.push_back(cell7);
        }
    }
    if (topBottomCells.size() != 2) {
        if (verbose_) {
            printf("Topology Error: vertex %ld has %ld topBottomCells\n", v7->id_, topBottomCells.size());
        }
        exit(1);
    }
    c123 = topBottomCells[0];
    c456 = topBottomCells[1];

    for (auto cell : v7->cells_) {
        if (std::find(v8->cells_.begin(), v8->cells_.end(), cell) != v8->cells_.end()) {
            if (std::find(topBottomCells.begin(), topBottomCells.end(), cell) == topBottomCells.end()) {
                c1245 = cell;
                break;
            }
        }
    }
    for (auto cell : v8->cells_) {
        if (std::find(v9->cells_.begin(), v9->cells_.end(), cell) != v9->cells_.end()) {
            if (std::find(topBottomCells.begin(), topBottomCells.end(), cell) == topBottomCells.end()) {
                c2356 = cell;
                break;
            }
        }
    }
    for (auto cell : v9->cells_) {
        if (std::find(v7->cells_.begin(), v7->cells_.end(), cell) != v7->cells_.end()) {
            if (std::find(topBottomCells.begin(), topBottomCells.end(), cell) == topBottomCells.end()) {
                c1346 = cell;
                break;
            }
        }
    }
    if (c1245 == NULL) {
        printf("Topology Error: c1245 not found in polygon %ld\n", polygon->id_);
        exit(1);
    }
    if (c2356 == NULL) {
        printf("Topology Error: c2356 not found in polygon %ld\n", polygon->id_);
        exit(1);
    }
    if (c1346 == NULL) {
        printf("Topology Error: c1346 not found in polygon %ld\n", polygon->id_);
        exit(1);
    }

    std::vector<Cell *> tmpCells = {c123, c456, c1245, c2356, c1346};
    if (verbose_) {
        dumpCells(true, false, tmpCells);
    }

    if (!checkCells(tmpCells)) {
        return 1;
    }

    // locate three side polygons: 1-1011-4, 2-1011-5, 3-1011-6
    Polygon * p14 = commonPolygon(c1245, c1346);
    Polygon * p25 = commonPolygon(c2356, c1245);
    Polygon * p36 = commonPolygon(c1346, c2356);
    // locate three polygons: 10-12, 10-23, 10-13
    Polygon * p12 = commonPolygon(c1245, c123);
    Polygon * p23 = commonPolygon(c2356, c123);
    Polygon * p13 = commonPolygon(c1346, c123);
    // locate three polygons: 11-45, 11-56, 11-46
    Polygon * p45 = commonPolygon(c1245, c456);
    Polygon * p56 = commonPolygon(c2356, c456);
    Polygon * p46 = commonPolygon(c1346, c456);
    if (p14 == NULL || p25 == NULL || p36 == NULL ||
        p12 == NULL || p23 == NULL || p13 == NULL ||
        p45 == NULL || p56 == NULL || p46 == NULL) {
        if (verbose_) {
            printf("Topology Warning: no common polygon or more than one common polygon\n");
        }
        return 1;
    }
    // check if side pair of polygons already have common edge
    if (commonEdge(p14, p25) != NULL) {
        return 1;
    }
    if (commonEdge(p25, p36) != NULL) {
        return 1;
    }
    if (commonEdge(p36, p14) != NULL) {
        return 1;
    }
    // locate nine edges: 7-8, 8-9, 7-9, 7-1, 8-2, 9-3, 7-4, 8-5, 9-6
    Edge * e78 = commonEdge(p12, p45);
    Edge * e89 = commonEdge(p23, p56);
    Edge * e79 = commonEdge(p13, p46);
    Edge * e71 = commonEdge(p12, p13);
    Edge * e82 = commonEdge(p23, p12);
    Edge * e93 = commonEdge(p13, p23);
    Edge * e74 = commonEdge(p45, p46);
    Edge * e85 = commonEdge(p56, p45);
    Edge * e96 = commonEdge(p46, p56);
    if (e78 == NULL || e89 == NULL || e79 == NULL ||
        e71 == NULL || e82 == NULL || e93 == NULL ||
        e74 == NULL || e85 == NULL || e96 == NULL) {
        if (verbose_) {
            printf("Topology Warning: H->I no common edge or more than one common edge\n");
        }
//        run_->updatePolygonVertices();
//        std::vector<Polygon *> tmp_polygons = {p14,p25,p36,p12,p23,p13,p45,p46,p56};
//        dumpVtk(tmp_polygons, false, true);
        return 1;
    }
    // locate six vertices: 1, 2, 3, 4, 5, 6
    Vertex * v1 = e71->otherVertex(v7);
    Vertex * v2 = e82->otherVertex(v8);
    Vertex * v3 = e93->otherVertex(v9);
    Vertex * v4 = e74->otherVertex(v7);
    Vertex * v5 = e85->otherVertex(v8);
    Vertex * v6 = e96->otherVertex(v9);
    if (v1 == NULL || v2 == NULL || v3 == NULL ||
        v4 == NULL || v5 == NULL || v6 == NULL) {
        printf("Topology Error: no common vertex\n");
        exit(1);
    }

    const auto topDirection = c123->polygonDirections_.find(polygon->id_);
    const auto bottomDirection = c456->polygonDirections_.find(polygon->id_);
    // CHANGED 2026-11-24: these two checks used to run further down, after the mesh had been edited,
    // and stop the whole simulation (exit). With 'validate_topology on' the swap is now skipped instead
    // (nothing has been edited yet). Without it the original behaviour is kept: same message, exit.
    if (topDirection == c123->polygonDirections_.end() ||
        bottomDirection == c456->polygonDirections_.end() ||
        topDirection->second == bottomDirection->second) {
        if (!run_->validateTopology_) {
            printf("Reconnection Error: c123 %ld and c456 %ld have the same direction on polygon 789\n", c123->id_, c456->id_);
            exit(1);
        }
        return 1;
    }
    if (find(c123->polygons_.begin(), c123->polygons_.end(), polygon) == c123->polygons_.end() ||
        find(c456->polygons_.begin(), c456->polygons_.end(), polygon) == c456->polygons_.end()) {
        if (!run_->validateTopology_) {
            printf("polygon 789 %ld not found in cell->polygons\n", polygon->id_);
            exit(1);
        }
        return 1;
    }
    ProposedSwap proposal(*run_, {c123, c456, c1245, c2356, c1346},
                          {p14, p25, p36, p12, p23, p13, p45, p56, p46});
    // create vertices 10, 11
    Vertex * v10 = new Vertex(run_, run_->count_vertices_);
    run_->count_vertices_ = run_->count_vertices_ + 1;
    run_->vertices_.push_back(v10);
    Vertex * v11 = new Vertex(run_, run_->count_vertices_);
    run_->count_vertices_ = run_->count_vertices_ + 1;
    run_->vertices_.push_back(v11);
    ////////// compute positions of vertices 10, 11     ///////////////
    double r78[3];
    double r79[3];
    computeDistance(v7->position_, v8->position_, r78);
    computeDistance(v7->position_, v9->position_, r79);
    // r0: midpoint position of triangle 789
    // uT: unit normal vector of triangle 789
    double r0[3];
    double uT[3];
    for (int m = 0; m < 3; m++) {
        r0[m] = v7->position_[m] + 1.0/3.0*(r78[m] + r79[m]);
    }
    uT[0] = r78[1]*r79[2] - r79[1]*r78[2];
    uT[1] = r79[0]*r78[2] - r78[0]*r79[2];
    uT[2] = r78[0]*r79[1] - r79[0]*r78[1];
    double uTL = sqrt(uT[0]*uT[0] + uT[1]*uT[1] + uT[2]*uT[2]);
    for (int m = 0; m < 3; m++) {
        uT[m] = uT[m]/uTL;
    }

    // compute positions of vertices 10, 11
    if (!c123->polygonDirections_[polygon->id_]) {
        // uT points to the top cell c123
        for (int m = 0; m < 3; m++) {
            v10->position_[m] = r0[m] + 0.5*Lth_*uT[m];
            v11->position_[m] = r0[m] - 0.5*Lth_*uT[m];
        }
    } else {
        // uT points to the bottom cell c456
        for (int m = 0; m < 3; m++) {
            v10->position_[m] = r0[m] - 0.5*Lth_*uT[m];
            v11->position_[m] = r0[m] + 0.5*Lth_*uT[m];
        }
    }
    run_->box_->resetPosition(v10->position_);
    run_->box_->resetPosition(v11->position_);
    ////////// compute positions of vertices 10, 11 done      /////////
    if (verbose_) {
        std::vector<Vertex *> tmpVertices = {v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11};
        dumpVertices(tmpVertices);
    }
    // associate cells to vertices 10, 11
    v10->cells_.push_back(c1245);
    v10->cells_.push_back(c1346);
    v10->cells_.push_back(c2356);
    v10->cells_.push_back(c123);
    v11->cells_.push_back(c1245);
    v11->cells_.push_back(c1346);
    v11->cells_.push_back(c2356);
    v11->cells_.push_back(c456);
    // create edge 10-11
    Edge * e1011 = run_->addEdge(v10, v11);
    // create edges: 10-1,10-2,10-3,11-4,11-5,11-6
    Edge * e1 = run_->addEdge(v10, v1);
    Edge * e2 = run_->addEdge(v10, v2);
    Edge * e3 = run_->addEdge(v10, v3);
    Edge * e4 = run_->addEdge(v11, v4);
    Edge * e5 = run_->addEdge(v11, v5);
    Edge * e6 = run_->addEdge(v11, v6);
    // update side polygon 1-1011-4
    p14->shrink(e71);
    p14->shrink(e74);
    p14->expand(e1011);
    p14->expand(e1);
    p14->expand(e4);
    // update side polygon 2-1011-5
    p25->shrink(e82);
    p25->shrink(e85);
    p25->expand(e1011);
    p25->expand(e2);
    p25->expand(e5);
    // update side polygon 3-1011-6
    p36->shrink(e93);
    p36->shrink(e96);
    p36->expand(e1011);
    p36->expand(e3);
    p36->expand(e6);
    // update polygon 10-12
    p12->shrink(e78);
    p12->shrink(e71);
    p12->shrink(e82);
    p12->expand(e1);
    p12->expand(e2);
    // update polygon 10-23
    p23->shrink(e89);
    p23->shrink(e82);
    p23->shrink(e93);
    p23->expand(e2);
    p23->expand(e3);
    // update polygon 10-13
    p13->shrink(e79);
    p13->shrink(e71);
    p13->shrink(e93);
    p13->expand(e1);
    p13->expand(e3);
    // update polygon 11-45
    p45->shrink(e78);
    p45->shrink(e74);
    p45->shrink(e85);
    p45->expand(e4);
    p45->expand(e5);
    // update polygon 11-56
    p56->shrink(e89);
    p56->shrink(e85);
    p56->shrink(e96);
    p56->expand(e5);
    p56->expand(e6);
    // update polygon 11-46
    p46->shrink(e79);
    p46->shrink(e74);
    p46->shrink(e96);
    p46->expand(e4);
    p46->expand(e6);

    std::vector<Cell *> tmp_cells = {c123, c456};
    for (auto cell : tmp_cells) {
        auto it = find(cell->polygons_.begin(), cell->polygons_.end(), polygon);
        cell->polygons_.erase(it);
    }
    if (!proposal.accept("H->I", polygon->id_)) return 1;

    // delete vertices 7, 8, 9
    run_->deleteVertex(v7);
    run_->deleteVertex(v8);
    run_->deleteVertex(v9);
    // delete edges
    run_->deleteEdge(e78);
    run_->deleteEdge(e89);
    run_->deleteEdge(e79);
    run_->deleteEdge(e71);
    run_->deleteEdge(e82);
    run_->deleteEdge(e93);
    run_->deleteEdge(e74);
    run_->deleteEdge(e85);
    run_->deleteEdge(e96);
    // delete polygon 789
    /////// check links ////////////////
    if (polygon->link_) {
        polygon->link_ = false;
        for (auto cell : tmp_cells) {
            if (cell->type_ > 0) {
                cell->link_ = false;
            }
        }
    }
    ///////-check-links-////////////////
    run_->deletePolygon(polygon);

//    // debug: turn off reconnection for all related edges
//    std::vector<Polygon *> tmp_polygons = {p14, p25, p36, p12, p23, p13, p45, p56, p46};
//    for (auto polygon : tmp_polygons) {
//        for (auto edge : polygon->edges_) {
//            edge->candidate_ = false;
//        }
//    }

    if (verbose_) {
        std::vector<Cell *> tmpCells = {c123, c456, c1245, c2356, c1346};
        dumpCells(false, false, tmpCells);
    }

    count_HI_ += 1;
    ++acceptedProposals_;

    return 0;
}

Polygon * Reconnection::commonPolygon(Cell * c1, Cell * c2) {
    std::vector<Polygon *> candidates;
    for (auto polygon : c1->polygons_) {
        if (std::find(c2->polygons_.begin(), c2->polygons_.end(), polygon) != c2->polygons_.end()) {
            candidates.push_back(polygon);
        }
    }
    if (candidates.size() == 1) {
        return candidates[0];
    } else if (candidates.size() == 0) {
        return NULL;
    } else {
//        c1->logPolygons("c1");
//        c2->logPolygons("c2");
        if (verbose_) {
            printf("Topology Warning: cell %ld and %ld have more than two common polygons\n", c1->id_, c2->id_);
        }
        return NULL;
    }
}

Edge * Reconnection::commonEdge(Polygon * p1, Polygon * p2) {
    std::vector<Edge *> candidates;
    for (auto edge : p1->edges_) {
        if (std::find(p2->edges_.begin(), p2->edges_.end(), edge) != p2->edges_.end()) {
            candidates.push_back(edge);
        }
    }
    if (candidates.size() == 1) {
        return candidates[0];
    } else if (candidates.size() == 0) {
        return NULL;
    } else {
//        p1->logEdges("e1");
//        p2->logEdges("e2");
        if (verbose_) {
            printf("Topology Warning: polygon %ld and %ld have more than two common edges\n", p1->id_, p2->id_);
        }
        return NULL;
    }
}

int Reconnection::computeDirection(double * r0, double * r1, double * w) {
    for (int m = 0; m < 3; m++) {
        w[m] = r1[m] - r0[m];
    }
    run_->box_->resetDistance(w);
    double wL = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    for (int m = 0; m < 3; m++) {
        w[m] = w[m]/wL;
    }

    return 0;
}

int Reconnection::computeDistance(double * r0, double * r1, double * w) {
    for (int m = 0; m < 3; m++) {
        w[m] = r1[m] - r0[m];
    }
    run_->box_->resetDistance(w);

    return 0;
}

int Reconnection::dumpVertices(std::vector<Vertex *> tmpVertices) {
    //////////////////////////////////////////////////////////////////////////////////////
    for (auto vertex : tmpVertices) {
        run_->verboseReconnection_ << vertex->id_ << " ";
    }
    run_->verboseReconnection_ << endl;
    for (auto vertex : tmpVertices) {
        run_->verboseReconnection_ << right << setw(12) << scientific << setprecision(5) << vertex->position_[0];
        run_->verboseReconnection_ << " " << right << setw(12) << scientific << setprecision(5) << vertex->position_[1];
        run_->verboseReconnection_ << " " << right << setw(12) << scientific << setprecision(5) << vertex->position_[2];
        run_->verboseReconnection_ << endl;
    }

    return 0;
}

int Reconnection::dumpCells(bool printTime, bool IH, std::vector<Cell *> tmpCells) {
    //////////////////////////////////////////////////////////////////////////////////////
    if (printTime) {
        if (IH) {
            run_->verboseReconnection_ << "I->H";
        } else {
            run_->verboseReconnection_ << "H->I";
        }
        run_->verboseReconnection_ << " time " << run_->simulation_time_ << " " << run_->count_reconnect_ << endl;
        for (auto cell : tmpCells) {
            run_->verboseReconnection_ << cell->id_ << " ";
        }
        run_->verboseReconnection_ << endl;
    }
    for (auto cell : tmpCells) {
        run_->verboseReconnection_ << cell->polygons_.size();
        for (auto polygon : cell->polygons_) {
            run_->verboseReconnection_ << " " << polygon->edges_.size();
        }
        run_->verboseReconnection_ << endl;
    }
    if (!printTime) {
        run_->verboseReconnection_ << endl;
    }

    return 0;
}

bool Reconnection::checkCells(std::vector<Cell *> cells) {
    int nReal = 0;
    int nEmpty = 0;
    for (auto cell : cells) {
        if (cell->type_ == 1) {
            nReal++;
        }
        if (cell->type_ < 0) {
            nEmpty++;
        }
    }
    if (nReal > 0 && nEmpty > 0) {
        return false;
    }
    if (nEmpty > 1) {
        return false;
    }

    return true;
}
