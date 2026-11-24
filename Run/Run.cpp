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
#include <algorithm>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include "Run.h"
#include "MeshValidation.h"

using namespace std;

Run::Run() {
//    dt_ = 0.001;
//    dtr_ = 10*dt_;
//    dump_period_ = 10000*dt_;
//    log_period_ = 100*dt_;
//    t_start_ = 0.;
//    t_end_ = 10000.;
    mu_ = 1.0;
    kB_ = 1.0;
//    temperature_ = 1.0e-5;
    NCell_ = 512;
    // CHANGED 2026-11: start-up values of the new switches (see Run.h): no linkers loaded
    // from a file yet, and the engine's own linker rules (as in the original code).
    linksLoaded_ = false;
    linkRulesPython_ = false;
    exactFaceCenterForces_ = false;
}

int Run::start() {
    count_dump_ = 0;
    count_log_ = 0;
    simulation_time_ = t_start_;
    long int simulation_time_counter = 0;
    double t_roundError = 0.01*dt_;
    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): exact restarts.
    // - Swap checks run at reconnect_origin + k dtr (dtr from conf, default 10 dt), so a run cut into pieces
    //   checks at the same times as one long run. Old: count_reconnect_ = 0, i.e. a check right after the
    //   first step of every engine start, then every dtr from there.
    // - rng.input (written as rng.output by the previous run) continues its noise stream and next IDs.
    const double timeTolerance = min(t_roundError, 32. * numeric_limits<double>::epsilon() *
                                     max(1., max(abs(t_start_), abs(t_end_))));
    const double sinceOrigin = t_start_ - reconnectOrigin_;
    count_reconnect_ = static_cast<long>(floor((sinceOrigin + timeTolerance) / dtr_)) + 1;
    ifstream rngInput("rng.input");
    if (rngInput) {
        if (!rng_.load(rngInput)) throw runtime_error("Invalid rng.input checkpoint");
        std::string marker;
        if (rngInput >> marker) {
            long vertex, edge, face;
            if (marker != "next_ids" || !(rngInput >> vertex >> edge >> face) ||
                vertex < 0 || edge < 0 || face < 0)
                throw runtime_error("Invalid next IDs in rng.input checkpoint");
            count_vertices_ = max(count_vertices_, vertex);
            count_edges_ = max(count_edges_, edge);
            count_polygons_ = max(count_polygons_, face);
        }
    }
    long int stepIndex = 0;
    auto start = chrono::steady_clock::now();

    printf("\nSimulation Start ...\n");
    printf("Real time elapsed: Rte\n");
    printf("Time        ");
    printf("Rte   ");
    printf("Volume   ");
    printf("I->H     ");
    printf("H->I     ");
    printf("Linker     ");
//    printf("NTS       ");
//    printf("ETE       ");
    printf("E_V      ");
    printf("E_S      ");
    printf("ECM_str  ");
    printf("ECM_bend ");
    printf("Link_str ");
    printf("Energy      \n");

    // CHANGED 2026-11: two start-up warnings for the new settings (they change nothing).
    // (1) Linkers are created only when a time step lands on the link start time. Warn if no
    //     step ever will, so a run silently without linkers is not mistaken for a working one.
    double tLinkStart = fiberLink_->linkStartTime_;
    double stepsToLink = (tLinkStart - t_start_) / dt_;
    if (!linkRulesPython_ && !linksLoaded_ && tLinkStart <= t_end_ + t_roundError &&
        (tLinkStart < t_start_ - t_roundError || fabs(stepsToLink - floor(stepsToLink + 0.5)) > 0.01)) {
        printf("WARNING: link start time %g is not on a time step of this run (start %g, dt %g): no linkers will be created\n",
               tLinkStart, t_start_, dt_);
    }
    // (2) Contact: a fiber node in contact feels a spring of about k * (fiber segment length);
    //     explicit time stepping overshoots once mu * k * l0 * dt goes above about 1.
    if (cellFiberContact_->k_ > 0.) {
        double stability = mu_ * cellFiberContact_->k_ * fiberElasticity_->l0_ * dt_;
        printf("contact stability number mu*k*l0*dt = %.3f%s\n", stability,
               stability > 1. ? "  WARNING: above 1, lower k or dt" : " (fine, below 1)");
    }

    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): the last frame is the one at exactly t_end; the loop
    // stops there. Old: while (simulation_time_ < t_end_ + t_roundError), which took one more step after the
    // last dump (that step was never saved).
    while (true) {
        const bool finalFrame = simulation_time_ >= t_end_;
        // update geometry information
        updateGeoinfo();
        // update volumeForces
        volume_->updateForces();
        if (validateTopology_) MeshValidation::positiveVolumes(*this);  // CHANGED 2026-11-24: optional mesh check
        // update interfaceForces
        interface_->updateForces();
        // update fiberForces
        fiberElasticity_->updateForces();
        // update linkForces
        fiberLink_->updateForces();
        // CHANGED 2026-11: new cell-fiber contact force (Energy/CellFiberContact.cpp). Without a
        // 'contact' line in conf it only sets every contact force to zero.
        cellFiberContact_->updateForces();
//        // update contactForces
//        contact_->updateForces();

        // update velocities
        updateVerticesVelocity();
        updateNodesVelocity();

        // CHANGED 2026-11: linkers used to be created only when the clock read exactly 500
        // (typed in here), so a run that started after 500, which is every restart, never had any.
        // Now the time comes from conf (link line, 7th number, default 500), and nothing is created
        // if the linkers were loaded from links.input or if Python decides them ('linkrules python').
        // Old: if (simulation_time_ > 500. - t_roundError && simulation_time_ < 500. + t_roundError) {
        double tLink = fiberLink_->linkStartTime_;
        if (!linkRulesPython_ && !linksLoaded_ && simulation_time_ > tLink - t_roundError && simulation_time_ < tLink + t_roundError) {
            initializeLinks();
        }


        // log to screen
        if (finalFrame || simulation_time_ - t_start_ + timeTolerance >= count_log_ * log_period_) {
            volume_->updateEnergy();
            interface_->updateEnergy();
            fiberElasticity_->updateEnergy();
            fiberLink_->updateEnergy();
            printf("%-12.2f%-6.1f%-9.1f%-9ld%-9ld%-5d+%-5ld%-9.1f%-9.1f%-9.1f%-9.1f%-9.1f%-9.1f\n", simulation_time_,
                   (chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count())/1.0e6,
                   volume_->totalVolume_,
                   reconnection_->count_IH_,
                   reconnection_->count_HI_,
                   fiberLink_->N_, fiberLink_->nFormedLinks_,
                   volume_->energy_,
                   interface_->energy_,
                   fiberElasticity_->stretchEnergy_,
                   fiberElasticity_->bendEnergy_,
                   fiberLink_->stretchEnergy_,
                   volume_->energy_+interface_->energy_+fiberElasticity_->stretchEnergy_+fiberElasticity_->bendEnergy_+fiberLink_->stretchEnergy_);
            start = chrono::steady_clock::now();
            reconnection_->count_IH_ = 0;
            reconnection_->count_HI_ = 0;
            count_log_++;
        }
        // dump
        if (finalFrame || simulation_time_ - t_start_ + timeTolerance >= count_dump_ * dump_period_) {
            if (simulation_time_ >= -timeTolerance || finalFrame) {
                dumpTopo();
                dumpECM();
                dumpCellCenter();
                dumpCellShapeIndex();
                dumpCellVolume();
//                dumpReconnection();
                if (dumpVtk_) dumpConfigurationVtk();  // CHANGED 2026-11-24: 'dump topo' skips the VTK files
                dumpLinkInfo();
                dumpSpheroidShape();
                // dumpVertexForce();
            }
//            dumpCellCenter();
//            dumpCellShapeIndex();
            count_dump_++;
        }

        if (finalFrame) break;

        // decrease the rest length of link spring from 2.0 to 0.2 in time interval 5000, every 200 time intervals.
        if (simulation_time_ > fiberLink_->shrinkStartTime_ - t_roundError  && fiberLink_->l0_ > fiberLink_->l1_) {
            if ((simulation_time_ - fiberLink_->shrinkStartTime_ + t_roundError ) > simulation_time_counter * 10.){
                // CHANGED 2026-11-19: never below the floor l1 (0.3 > 0.2 used to give 0.1).
                // Old: fiberLink_->l0_ = fiberLink_->l0_ - fiberLink_->shrinkSpeed_;
                fiberLink_->l0_ = max(fiberLink_->l1_, fiberLink_->l0_ - fiberLink_->shrinkSpeed_);
                simulation_time_counter = simulation_time_counter + 1;
            }
        }

        // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): Euler step and swap check. The step ends exactly on
        // t_end or on the next swap-check time. Old: updateVerticesPosition(); updateNodesPosition(); then the
        // swap check, then simulation_time_ += dt_ (see the old loop condition above).
        // The endpoint is a real frame. Reconnect after advancing to its labeled time.
        const double nominalTime = t_start_ + (stepIndex + 1) * dt_;
        double nextTime = min(nominalTime, t_end_);
        if (reconnection_->Lth_ > 0.) {
            const double reconnectTime = reconnectOrigin_ + count_reconnect_ * dtr_;
            if (reconnectTime < nextTime - timeTolerance) nextTime = reconnectTime;
        }
        if (nextTime >= t_end_ - timeTolerance) nextTime = t_end_;
        if (nextTime >= nominalTime - timeTolerance) ++stepIndex;
        double step = nextTime - simulation_time_;
        if (step <= 0.) throw runtime_error("Time step cannot advance at this precision");
        if (abs(step - dt_) <= timeTolerance) step = dt_;
        updateVerticesPosition(step);
        updateNodesPosition(step);
        simulation_time_ = nextTime;
        if (simulation_time_ - reconnectOrigin_ + timeTolerance >= count_reconnect_ * dtr_) {
            if (reconnection_->Lth_ > 0.) {
                reconnection_->start();
                ++reconnectSweeps_;
                if (linksLoaded_ || simulation_time_ > fiberLink_->linkStartTime_ + t_roundError)
                    checkLinks();
            }
            count_reconnect_ = static_cast<long>(
                floor((simulation_time_ - reconnectOrigin_ + timeTolerance) / dtr_)) + 1;
        }
    }

//    for (long int i = 0; i < cells_.size(); i++) {
//        printf("%f\n", cells_[i]->volume_);
//    }

    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): save the noise stream and next IDs for the next run
    // (rng.output -> rng.input), and a short summary (swap checks, accepted and rejected swaps).
    ofstream rngOutput("rng.output");
    if (!rng_.save(rngOutput)) throw runtime_error("Could not write rng.output");
    rngOutput << "next_ids " << count_vertices_ << ' ' << count_edges_ << ' ' << count_polygons_ << '\n';
    rngOutput.close();
    if (!rngOutput) throw runtime_error("Could not close rng.output");
    ofstream summary("run_summary.txt");
    summary << setprecision(numeric_limits<double>::max_digits10)
            << "final_time " << simulation_time_ << '\n'
            << "reconnection_sweeps " << reconnectSweeps_ << '\n'
            << "accepted_reconnection_proposals " << reconnection_->acceptedProposals_ << '\n'
            << "rejected_reconnection_proposals " << reconnection_->rejectedProposals_ << '\n';
    if (!verboseReconnection_.str().empty()) dumpReconnection();
    summary.close();
    if (!summary) throw runtime_error("Could not write run_summary.txt");
    return 0;
}

int     Run::updateVerticesVelocity() {
    for (auto vertex : vertices_) {
        for (int m = 0; m < 3; m++) {
            // CHANGED 2026-11: add the contact force (exactly zero when contact is off).
            // Old: vertex->velocity_[m] = mu_ * (vertex->volumeForce_[m] + vertex->interfaceForce_[m] + vertex->linkForce_[m]);
            vertex->velocity_[m] = mu_ * (vertex->volumeForce_[m] + vertex->interfaceForce_[m] + vertex->linkForce_[m] + vertex->contactForce_[m]);
        }
    }
    // remove drift velocity
//    if (true) {
//        double averageVelocity[3] = {0., 0., 0.};
//        long int nRealVertices = 0;
//        for (auto vertex : vertices_) {
//            if (vertex->type_ > 0) {
//                for (int m = 0; m < 3; m++) {
//                    averageVelocity[m] = averageVelocity[m] + vertex->velocity_[m];
//                }
//                nRealVertices++;
//            }
//        }
//        for (int m = 0; m < 3; m++) {
//            averageVelocity[m] = averageVelocity[m] / nRealVertices;
//        }
//        for (auto vertex : vertices_) {
//            if (vertex->type_ > 0) {
//                for (int m = 0; m < 3; m++) {
//                    vertex->velocity_[m] = vertex->velocity_[m] - averageVelocity[m];
//                }
//            }
//        }
//    }

    return 0;
}

// CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): one seeded noise stream (rng_, Random.h) instead of a
// generator re-seeded from std::random_device at every step, and the step length is passed in (it is dt
// except for a shorter last step onto t_end or a swap-check time). Same update otherwise: real-cell vertices
// move by velocity*step + noise (no noise for pulled vertices); medium vertices by at most 0.1 per step.
int Run::updateVerticesPosition(double step) {
    const double mediumStep = dte_ * (step / dt_);
    const double noise = sqrt(2. * mu_ * kB_ * temperature_ * step);
    const double mediumNoise = sqrt(2. * mu_ * kB_ * temperature_ * mediumStep);
    for (auto vertex : vertices_) {
        for (int m = 0; m < 3; ++m) {
            if (vertex->type_ > 0) {
                vertex->position_[m] = vertex->position_[m] + vertex->velocity_[m] * step +
                    (noise && !vertex->pull_ ? noise * rng_.normal() : 0.);
            } else {
                // CHANGED 2026-11-24: the original arithmetic, so a full step is bit-identical to the original
                // (cap the velocity, then multiply). The first port capped the displacement at exactly 0.1,
                // which can differ from (0.1/dte)*dte in the last digit, enough to change chaotic medium motion.
                double velocity = vertex->velocity_[m];
                if (velocity * mediumStep > 0.1) velocity = 0.1 / mediumStep;
                if (velocity * mediumStep < -0.1) velocity = -0.1 / mediumStep;
                vertex->position_[m] = vertex->position_[m] + velocity * mediumStep +
                    (mediumNoise ? mediumNoise * rng_.normal() : 0.);
            }
        }
        box_->resetPosition(vertex->position_);
    }
    return 0;
}

int     Run::updateNodesVelocity() {
    for (auto node : nodes_) {
        for (int m = 0; m < 3; m++) {
            node->velocity_[m] = mu_ * (node->stretchForce_[m] + node->bendForce_[m] + node->linkForce_[m] + node->contactForce_[m]);
        }
    }

    return 0;
}

int     Run::updateNodesPosition(double step) {
    for (auto node : nodes_) {
        for (int m = 0; m < 3; m++) {
            node->position_[m] = node->position_[m] + node->velocity_[m] * step;
        }
        box_->resetPosition(node->position_);
    }

    return 0;
}

int     Run::initializeLinks() {
    updateGeoinfo();
    updatePolygonCells();
    links_.clear();

    boundaryPolygons_.clear();
    for (auto polygon : polygons_) {
        if (polygon->type_ == 2) {
            boundaryPolygons_.push_back(polygon);
        }
    }
    std::vector<Node *> boundaryNodes_;
    std::vector<Node *> tmpNodes_;
    for (auto node : nodes_) {
        if (node->link_) {
            boundaryNodes_.push_back(node);
        }
    }
    if (boundaryNodes_.size() < fiberLink_->N_) {
        fiberLink_->N_ = boundaryNodes_.size();
    }
    std::shuffle(boundaryNodes_.begin(), boundaryNodes_.end(), rng_.engine());  // CHANGED 2026-11-24: same seeded stream

    // set pairs of polygon and node
    int countNode = 0;
    for (auto node : boundaryNodes_) {
        Polygon * pairPolygon = NULL;
        double minDist = 1000.;
        for (auto polygon : boundaryPolygons_) {
            if (polygon->link_) {
                continue;
            }
            bool skipFlag = false;
            for (auto cell : polygon->cells_) {
                if (cell->type_ > 0 && cell->link_) {
                    skipFlag = true;
                    break;
                }
            }
            if (skipFlag) {
                continue;
            }
            polygon->updateCenter();
            polygon->updateArea();
            if (polygon->area_ < fiberLink_->minArea_) {
                continue;
            }
            double dx[3];
            dx[0] = node->position_[0] - polygon->center_[0];
            dx[1] = node->position_[1] - polygon->center_[1];
            dx[2] = node->position_[2] - polygon->center_[2];
//            box_->resetDistance(dx);
            double dist = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
            if (dist > 3.) {
                continue;
            }
            if (dist < minDist) {
                minDist = dist;
                pairPolygon = polygon;
            }
        }
        if (pairPolygon != NULL) {
            Link * link = new Link(this, pairPolygon, node);
            links_.push_back(link);
            pairPolygon->link_ = true;
            for (auto cell : pairPolygon->cells_) {
                if (cell->type_ > 0) {
                    cell->link_ = true;
                }
            }
            countNode++;
        }
        if(countNode == fiberLink_->N_) {
            break;
        }
    }

//    int count = 0;
//    for (auto link : links_) {
//        cout << count << " " << link->polygon_->id_ << " " << link->node_->id_ << endl;
//        count++;
//    }

    return 0;
}

// CHANGED 2026-11 (new function): rebuild the cell-ECM linkers of a previous run.
// Why: the engine writes its linkers to NNNNNNN.linkInfo.txt at every dump ("PolygonID NodeID",
// one line per linker) but never read them back, and it creates linkers only once, at the link
// start time. So every restart after that time ran with zero linkers. To restart with them, copy
// the last linkInfo file of the previous run to links.input in the new run folder (run_growth.py
// does this). If links.input does not exist, nothing is loaded and the run is the original one.
// Which lines are accepted: the face must exist, be on the organoid surface (type 2) and not be
// linked already, and each ECM node is used once. With the engine's rules ('linkrules engine',
// the default) a cell may also hold only one linker; a rejected line's node is re-attached to the
// best free surface face at the first checkLinks() (same rule as a linker whose face vanished).
// With 'linkrules python' several linkers per cell are allowed and a rejected line is dropped.
int     Run::loadLinks(std::string filename) {
    ifstream in(filename.c_str());
    if (!in.is_open()) {
        return 0;
    }
    updateGeoinfo();
    updatePolygonCells();
    unordered_map<long int, Polygon *> polygonByID;
    for (auto polygon : polygons_) {
        polygonByID[polygon->id_] = polygon;
    }
    unordered_map<long int, Node *> nodeByID;
    for (auto node : nodes_) {
        nodeByID[node->id_] = node;
    }

    links_.clear();
    pendingFreeNodes_.clear();
    long int nRestored = 0;
    long int nFree = 0;
    long int nSkipped = 0;
    std::unordered_map<long int, bool> nodeUsed;
    string buffer;
    while (getline(in, buffer)) {
        istringstream iss(buffer);
        long int polygonID, nodeID;
        if (!(iss >> polygonID >> nodeID)) {
            continue;  // header line ("PolygonID NodeID") or blank line
        }
        // CHANGED 2026-11-24: optional 3rd column = a passive linker's saved rest length
        double savedRest = -1.;
        if (!(iss >> savedRest) || !(savedRest > 0.)) savedRest = -1.;
        auto nodeIt = nodeByID.find(nodeID);
        // CHANGED 2026-11-24 (Matrigel-like mesh): under the Python link rules a node may appear on several
        // lines; linkers.py enforces how many (link_node_capacity, default 1, so collagen runs are unchanged).
        // The forces of several linkers on one node add up in FiberLink::updateForces. Old line:
        //     if (nodeIt == nodeByID.end() || nodeUsed[nodeID]) {
        if (nodeIt == nodeByID.end() || (nodeUsed[nodeID] && !linkRulesPython_)) {
            nSkipped++;  // unknown ECM node, or a node that is already on another line
            continue;
        }
        nodeUsed[nodeID] = true;
        Node * node = nodeIt->second;
        auto polygonIt = polygonByID.find(polygonID);
        bool ok = (polygonIt != polygonByID.end()) && polygonIt->second->type_ == 2 && !polygonIt->second->link_;
        if (ok && !linkRulesPython_) {  // engine rules: one linker per cell
            for (auto cell : polygonIt->second->cells_) {
                if (cell->type_ > 0 && cell->link_) {
                    ok = false;
                    break;
                }
            }
        }
        if (ok) {
            Polygon * polygon = polygonIt->second;
            Link * link = new Link(this, polygon, node);
            // CHANGED 2026-11-24: passive linkers keep their saved rest length; a new one gets its length now
            if (fiberLink_->restAtAttach_)
                link->restLength_ = savedRest > 0. ? savedRest : fiberLink_->linkLength(link);
            links_.push_back(link);
            polygon->link_ = true;
            for (auto cell : polygon->cells_) {
                if (cell->type_ > 0) {
                    cell->link_ = true;
                }
            }
            nRestored++;
        } else if (!linkRulesPython_) {
            // face missing (e.g. its cell divided in Python), not on the surface, already linked,
            // or its cell already has a linker: re-attach this node at the first checkLinks()
            pendingFreeNodes_.push_back(node);
            nFree++;
        } else {
            nSkipped++;
        }
    }
    if (nRestored + nFree > 0) {
        linksLoaded_ = true;
    }
    cout << "Restored " << nRestored << " cell-ECM links from " << filename
         << " (" << nFree << " ECM nodes will be re-attached at the first link check, "
         << nSkipped << " lines skipped)" << endl;
    return 0;
}

int     Run::checkLinks() {
    updatePolygonCells();
    std::vector<Link *> tmpLinks = links_;
    links_.clear();
    std::vector<Node *> freeNodes;
    // CHANGED 2026-11: ECM nodes that loadLinks() could not put back on their saved face are
    // re-attached below, by the same rule the original code uses for a linker whose face vanished.
    for (auto node : pendingFreeNodes_) {
        freeNodes.push_back(node);
    }
    pendingFreeNodes_.clear();

    // CHANGED 2026-11: with 'linkrules python' Python decides the linkers between runs
    // (linkers.py). Here a linker is only removed when a neighbour swap deleted its face
    // (deletePolygon sets id_ = -1); nothing is re-attached. The original rule below would also
    // remove every linker whose face is badly shaped for a moment (an edge shorter than 0.02 just
    // before a swap, area < 0.1, ...), which in tests removed 25-45% of the linkers per time
    // unit, several times Python's own turnover.
    if (linkRulesPython_) {
        for (auto link : tmpLinks) {
            if (link->polygon_->id_ >= 0) {
                links_.push_back(link);
            } else {
                delete link;
            }
        }
        return 0;
    }

    for (auto link : tmpLinks) {
        Polygon * polygon = link->polygon_;
        Node * node = link->node_;
//        cout << run_->simulation_time_ << " " << polygon->id_ << endl;

        if (polygon->id_ > 0) {
            if (polygon->checkShape()) {
                links_.push_back(link);
            } else {
                polygon->link_ = false;
                for (auto cell : polygon->cells_) {
                    if (cell->type_ > 0) {
                        cell->link_ = false;
                    }
                }
                delete link;
                freeNodes.push_back(node);
            }
        } else {
            delete link;
            freeNodes.push_back(node);
        }
    }

    boundaryPolygons_.clear();
    for (auto polygon : polygons_) {
        if (polygon->type_ == 2) {
            boundaryPolygons_.push_back(polygon);
        }
    }

    // add pairs of polygon and free node
    for (auto node : freeNodes) {
        Polygon * pairPolygon = NULL;
        double minDiff = 1000.;
        for (auto polygon : boundaryPolygons_) {
            if (polygon->link_) {
                continue;
            }
            bool skipFlag = false;
            for (auto cell : polygon->cells_) {
                if (cell->type_ > 0 && cell->link_) {
                    skipFlag = true;
                    break;
                }
            }
            if (skipFlag) {
                continue;
            }
            if (!polygon->checkShape()) {
                continue;
            }
            double dx[3];
            dx[0] = node->position_[0] - polygon->center_[0];
            dx[1] = node->position_[1] - polygon->center_[1];
            dx[2] = node->position_[2] - polygon->center_[2];
            box_->resetDistance(dx);
            double diff = fabs(sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]) - fiberLink_->l0_);
//            if (diff > 1.0) {
//                continue;
//            }
            if (diff < minDiff) {
                minDiff = diff;
                pairPolygon = polygon;
            }
        }
        if (pairPolygon != NULL) {
            Link * link = new Link(this, pairPolygon, node);
            links_.push_back(link);
            pairPolygon->link_ = true;
            for (auto cell : pairPolygon->cells_) {
                if (cell->type_ > 0) {
                    cell->link_ = true;
                }
            }
            fiberLink_->nFormedLinks_++;
        }
    }

    return 0;
}

int     Run::updateLinks() {
    std::vector<Link *> tmpLinks = links_;
    links_.clear();
    std::vector<Node *> freeNodes;

    for (auto link : tmpLinks) {
        Polygon * polygon = link->polygon_;
        Node * node = link->node_;
//        cout << run_->simulation_time_ << " " << polygon->id_ << endl;

        double dx[3];
        dx[0] = node->position_[0] - polygon->center_[0];
        dx[1] = node->position_[1] - polygon->center_[1];
        dx[2] = node->position_[2] - polygon->center_[2];
        box_->resetDistance(dx);
//        double l = sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
        if (polygon->checkShape()) {
            links_.push_back(link);
        } else {
            delete link;
            polygon->link_ = false;
            freeNodes.push_back(node);
        }
    }

    boundaryPolygons_.clear();
    for (auto polygon : polygons_) {
        if (polygon->type_ == 2) {
            boundaryPolygons_.push_back(polygon);
        }
    }

    // add pairs of polygon and free node
    for (auto node : freeNodes) {
        Polygon * pairPolygon = NULL;
        double minDiff = 1000.;
        for (auto polygon : boundaryPolygons_) {
            if (polygon->link_) {
                continue;
            }
            if (!polygon->checkShape()) {
                continue;
            }
            double dx[3];
            dx[0] = node->position_[0] - polygon->center_[0];
            dx[1] = node->position_[1] - polygon->center_[1];
            dx[2] = node->position_[2] - polygon->center_[2];
            box_->resetDistance(dx);
            double diff = fabs(sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]) - fiberLink_->l0_);
//            if (diff > 1.0) {
//                continue;
//            }
            if (diff < minDiff) {
                minDiff = diff;
                pairPolygon = polygon;
            }
        }
        if (pairPolygon != NULL) {
            Link * link = new Link(this, pairPolygon, node);
            links_.push_back(link);
            pairPolygon->link_ = true;
        }
    }

    return 0;
}

int     Run::updatePolygonVertices() {
    // update vertices in polygon
    for (long int i = 0; i < polygons_.size(); i++) {
        polygons_[i]->updateVertices();
    }

    return 0;
}

int     Run::updateCellVertices() {
    // update vertices in cell
    updatePolygonVertices();
    for (auto cell : cells_) {
        cell->vertices_.clear();
        for (auto polygon : cell->polygons_) {
            for (auto vertex : polygon->vertices_) {
                if (std::find(cell->vertices_.begin(), cell->vertices_.end(), vertex) == cell->vertices_.end()) {
                    // new vertex to be added
                    cell->vertices_.push_back(vertex);
                }
            }
        }
    }

    return 0;
}

int     Run::updateVertexEdges() {
    for (long int i = 0; i < vertices_.size(); i++) {
        vertices_[i]->edges_.clear();
    }
    for (long int i = 0; i < edges_.size(); i++) {
        edges_[i]->vertices_[0]->edges_.push_back(edges_[i]);
        edges_[i]->vertices_[1]->edges_.push_back(edges_[i]);
    }

    return 0;
}

int     Run::updateVertexCells() {
    for (auto vertex : vertices_) {
        vertex->cells_.clear();
    }
    for (auto cell : cells_) {
        for (auto polygon : cell->polygons_) {
            for (auto edge : polygon->edges_) {
                for (auto vertex : edge->vertices_) {
                    if (std::find(vertex->cells_.begin(), vertex->cells_.end(), cell) == vertex->cells_.end()) {
                        // new cell to be added
                        vertex->cells_.push_back(cell);
                    }
                }
            }
        }
    }

//    for (long int i = 0; i < vertices_.size(); i++) {
//        printf("%d\n", vertices_[i]->cells_.size());
//    }

    return 0;
}

int     Run::updateEdgeCells() {
    for (auto edge : edges_) {
        edge->cells_.clear();
    }
    for (auto cell : cells_) {
        for (auto polygon : cell->polygons_) {
            for (auto edge : polygon->edges_) {
                if (std::find(edge->cells_.begin(), edge->cells_.end(), cell) == edge->cells_.end()) {
                    // new cell to be added
                    edge->cells_.push_back(cell);
                }
            }
        }
    }

    return 0;
}

int     Run::updatePolygonCells() {
    for (long int i = 0; i < polygons_.size(); i++) {
        polygons_[i]->cells_.clear();
    }
    for (long int i = 0; i < cells_.size(); i++) {
        for (int j = 0; j < cells_[i]->polygons_.size(); j++) {
            cells_[i]->polygons_[j]->cells_.push_back(cells_[i]);
        }
    }

    return 0;
}

int     Run::updateCellShapeIndex() {
    for (auto cell : cells_) {
        double area = 0.;
        for (auto polygon : cell->polygons_) {
            area += polygon->area_;
        }
        cell->shapeIndex_ = area * pow(cell->volume_, (-1.0)*2.0/3.0);
    }

    return 0;
}

int     Run::updateGeoinfo() {
    // update edge midpoint and length
    for (long int i = 0; i < edges_.size(); i++) {
        edges_[i]->update();
//        printf("%6f\n", run->edges_[i]->length_);
    }
    // update polygon center position
    for (long int i = 0; i < polygons_.size(); i++) {
        polygons_[i]->updateCenter();
    }

    return 0;
}

int     Run::updateEmptyCells() {
    for (auto polygon : polygons_) {
        polygon->type_ = 0;
    }
    for (auto vertex : vertices_) {
        vertex->type_ = 0;
    }
    for (auto edge : edges_) {
        edge->type_ = 0;
    }
    for (auto cell : cells_) {
        if (cell->type_ == 1) {
            for (auto polygon : cell->polygons_) {
                polygon->type_ = 1;
            }
        }
    }
    for (auto polygon : polygons_) {
        if (polygon->type_ == 1) {
            if (polygon->cells_[0]->type_ == 0 || polygon->cells_[1]->type_ == 0) {
                polygon->type_ = 2;
            }
//            if (polygon->cells_[0]->type_ < 0 || polygon->cells_[1]->type_ < 0) {
//                polygon->type_ = 2;
//            }
        }
    }
    for (auto edge : edges_) {
        int n = 0;
        for (auto cell : edge->cells_) {
            if (cell->type_ > 0) {
                n++;
            }
        }
        edge->type_ = n;
    }
    for (auto vertex : vertices_) {
        int n = 0;
        for (auto cell : vertex->cells_) {
            if (cell->type_ > 0) {
                n++;
            }
        }
        vertex->type_ = n;
    }

    // check isolated cell
    for (auto cell : cells_) {
        if (cell->type_ == 1) {
            bool isolated = true;
            for (auto polygon : cell->polygons_) {
                if (polygon->type_ != 2) {
                    isolated = false;
                    break;
                }
            }
            if (isolated) {
                cell->type_ = 0;
                for (auto polygon : cell->polygons_) {
                    polygon->type_ = 0;
                    for (auto edge: polygon->edges_) {
                        edge->type_ = 0;
                        for (auto vertex : edge->vertices_) {
                            vertex->type_ = 0;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int     Run::deleteVertex(Vertex * vertex) {
    auto it = find(vertices_.begin(), vertices_.end(), vertex);
    if (it != vertices_.end()) {
//        int index = it - vertices_.begin();
        vertices_.erase(it);
    } else {
        printf("vertex %ld not found in vertices_\n", vertex->id_);
        exit(1);
    }
    delete vertex;

    return 0;
}

int     Run::deleteEdge(Edge * edge) {
    auto it = find(edges_.begin(), edges_.end(), edge);
    if (it != edges_.end()) {
        edges_[it-edges_.begin()]->markToDelete_ = true;
//        edges_.erase(it);
    } else {
        printf("edge %ld not found in edges_\n", edge->id_);
        exit(1);
    }
//    delete edge;

    return 0;
}

int     Run::deletePolygon(Polygon * polygon) {
    auto it = find(polygons_.begin(), polygons_.end(), polygon);
    if (it != polygons_.end()) {
//        int index = it - vertices_.begin();
        polygons_.erase(it);
    } else {
        printf("polygon %ld not found in polygons_\n", polygon->id_);
        exit(1);
    }
//    delete polygon;
    polygon->id_ = -1;

    return 0;
}

Edge *  Run::addEdge(Vertex * v0, Vertex * v1) {
    Edge * edge = new Edge(this, count_edges_);
    count_edges_ += 1;
    edges_.push_back(edge);
    if (v0->id_ < v1->id_) {
        edge->vertices_.push_back(v0);
        edge->vertices_.push_back(v1);
    } else {
        edge->vertices_.push_back(v1);
        edge->vertices_.push_back(v0);
    }
    edge->update();
    edge->candidate_ = false;

    return edge;
}

// CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): output file names. Whole times give the usual 7-digit
// name (0000080); other times get their full value, so two dumps within one time unit no longer overwrite
// each other. Old: setw(7) << floor(simulation_time_ + 0.01 dt).
string Run::dumpTag() const {
    ostringstream tag;
    const double integerTime = round(simulation_time_);
    if (integerTime >= 0. && abs(simulation_time_ - integerTime) <=
            min(0.01 * dt_, 32. * numeric_limits<double>::epsilon() * max(1., abs(simulation_time_)))) {
        tag << setw(7) << setfill('0') << fixed << setprecision(0) << integerTime;
    } else {
        tag << setprecision(numeric_limits<double>::max_digits10) << simulation_time_;
    }
    return tag.str();
}

int Run::dumpConfigurationVtk() {
    updatePolygonVertices();
    //////////////////////////////////////////////////////////////////////////////////////
    stringstream filename;
    filename << dumpTag() << ".sample.vtk";
    ofstream out(filename.str().c_str());
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    out << "# vtk DataFile Version 2.0" << endl;
    out << "polydata" << endl;
    out << "ASCII" << endl;
    out << "DATASET POLYDATA" << endl;
    out << "POINTS " << vertices_.size() << " double" << endl;
    for (long int i = 0; i < vertices_.size(); i++) {
        // reset vertex id for dumping polygons
        vertices_[i]->dumpID_ = i;
        out << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[0];
        out << " " << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[1];
        out << " " << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[2];
        out << endl;
    }
    out << endl;

    long int Npolygons = 0;
    long int NpolygonVertices = 0;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ > 0 && !polygons_[i]->crossBoundary()) {
            Npolygons++;
            NpolygonVertices += polygons_[i]->vertices_.size();
        }
    }
    out << "POLYGONS " << Npolygons << " " << Npolygons + NpolygonVertices << endl;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ > 0 && !polygons_[i]->crossBoundary()) {
            out << left << setw(6) << polygons_[i]->vertices_.size();
            for (int j = 0; j < polygons_[i]->vertices_.size(); j++) {
                out << " " << left << setw(6) << polygons_[i]->vertices_[j]->dumpID_;
            }
            out << endl;
        }
    }
    out << endl;

    out << "CELL_DATA " << Npolygons << endl;
    out << "SCALARS type int 1" << endl;
    out << "LOOKUP_TABLE default" << endl;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ > 0 && !polygons_[i]->crossBoundary()) {
            out << left << setw(6) << polygons_[i]->type_ << endl;
        }
    }
    out << endl;

    out.close();

    //////////////////////////////////////////////////////////////////////////////////////
    // extra.vtk: not so useful, commenting out this block
    /*
    stringstream filename2;
    filename2 << dumpTag() << ".extra.vtk";
    ofstream out2(filename2.str().c_str());
    if (!out2.is_open()) {
        cout << "Error opening output file " << filename2.str().c_str() << endl;
        exit(1);
    }
    out2 << "# vtk DataFile Version 2.0" << endl;
    out2 << "polydata" << endl;
    out2 << "ASCII" << endl;
    out2 << "DATASET POLYDATA" << endl;

    long int Nvertices = 0;
    for (auto vertex : vertices_) {
        if (vertex->type_ > 0) {
            // reset vertex id for dumping polygons
            vertex->dumpID_ = Nvertices;
            Nvertices++;
        }
    }
    out2 << "POINTS " << Nvertices << " double" << endl;
    for (auto vertex : vertices_) {
        if (vertex->type_ > 0) {
            out2 << right << setw(12) << scientific << setprecision(5) << vertex->position_[0];
            out2 << " " << right << setw(12) << scientific << setprecision(5) << vertex->position_[1];
            out2 << " " << right << setw(12) << scientific << setprecision(5) << vertex->position_[2];
            out2 << endl;
        }
    }
    out2 << endl;

    long int Nedges = 0;
    long int NedgeVertices = 0;
    for (auto edge : edges_) {
        if (edge->type_ >= 2 && !edge->crossBoundary()) {
            Nedges++;
            NedgeVertices += 2;
        }
    }
    out2 << "LINES " << Nedges << " " << Nedges + NedgeVertices << endl;
    for (auto edge : edges_) {
        if (edge->type_ >= 2 && !edge->crossBoundary()) {
            out2 << left << setw(6) << 2;
            out2 << " " << left << setw(6) << edge->vertices_[0]->dumpID_;
            out2 << " " << left << setw(6) << edge->vertices_[1]->dumpID_;
            out2 << endl;
        }
    }
    out2 << endl;

    out2 << "POINT_DATA " << Nvertices << endl;
    out2 << "SCALARS type int 1" << endl;
    out2 << "LOOKUP_TABLE default" << endl;
    for (auto vertex : vertices_) {
        if (vertex->type_ > 0) {
            out2 << left << setw(6) << vertex->type_ + 10 << endl;
        }
    }
    out2 << endl;

    out2.close();
    */
    
    //////////////////////////////////////////////////////////////////////////////////////
    // dump empty cells vtk: this is not so useful, so we comment out this 
    /*
    stringstream filename1;
    filename1 << dumpTag() << ".empty.vtk";
    ofstream out1(filename1.str().c_str());
    if (!out1.is_open()) {
        cout << "Error opening output file " << filename1.str().c_str() << endl;
        exit(1);
    }
    out1 << "# vtk DataFile Version 2.0" << endl;
    out1 << "polydata" << endl;
    out1 << "ASCII" << endl;
    out1 << "DATASET POLYDATA" << endl;
    out1 << "POINTS " << vertices_.size() << " double" << endl;
    for (long int i = 0; i < vertices_.size(); i++) {
        // reset vertex id for dumping polygons
        vertices_[i]->dumpID_ = i;
        out1 << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[0];
        out1 << " " << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[1];
        out1 << " " << right << setw(12) << scientific << setprecision(5) << vertices_[i]->position_[2];
        out1 << endl;
    }
    out1 << endl;

    Npolygons = 0;
    NpolygonVertices = 0;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ != 1 && !polygons_[i]->crossBoundary()) {
            Npolygons++;
            NpolygonVertices += polygons_[i]->vertices_.size();
        }
    }
    out1 << "POLYGONS " << Npolygons << " " << Npolygons + NpolygonVertices << endl;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ != 1 && !polygons_[i]->crossBoundary()) {
            out1 << left << setw(6) << polygons_[i]->vertices_.size();
            for (int j = 0; j < polygons_[i]->vertices_.size(); j++) {
                out1 << " " << left << setw(6) << polygons_[i]->vertices_[j]->dumpID_;
            }
            out1 << endl;
        }
    }
    out1 << endl;

    out1 << "CELL_DATA " << Npolygons << endl;
    out1 << "SCALARS type int 1" << endl;
    out1 << "LOOKUP_TABLE default" << endl;
    for (long int i = 0; i < polygons_.size(); i++) {
        if (polygons_[i]->type_ != 1 && !polygons_[i]->crossBoundary()) {
            out1 << left << setw(6) << polygons_[i]->type_ << endl;
        }
    }
    out1 << endl;

    out1.close();
    */
    //////////////////////////////////////////////////////////////////////////////////////
    stringstream ECMfilename;
    ECMfilename << dumpTag() << ".ECM.vtk";
    ofstream ECMout(ECMfilename.str().c_str());
    if (!ECMout.is_open()) {
        cout << "Error opening output file " << ECMfilename.str().c_str() << endl;
        exit(1);
    }
    ECMout << "# vtk DataFile Version 2.0" << endl;
    ECMout << "polydata" << endl;
    ECMout << "ASCII" << endl;
    ECMout << "DATASET POLYDATA" << endl;
    ECMout << "POINTS " << nodes_.size() << " double" << endl;
    for (long int i = 0; i < nodes_.size(); i++) {
        // reset vertex id for dumping polygons
        nodes_[i]->dumpID_ = i;
        ECMout << right << setw(12) << scientific << setprecision(5) << nodes_[i]->position_[0];
        ECMout << " " << right << setw(12) << scientific << setprecision(5) << nodes_[i]->position_[1];
        ECMout << " " << right << setw(12) << scientific << setprecision(5) << nodes_[i]->position_[2];
        ECMout << endl;
    }
    ECMout << endl;

    long int Nfibers = 0;
    for (auto fiber : fibers_) {
        for (int i = 0; i < fiber->nodes_.size() - 1; i++) {
            Node * ni = fiber->nodes_[i];
            Node * nj = fiber->nodes_[i + 1];
            if (!box_->crossBoundary(ni->position_, nj->position_)) {
                Nfibers++;
            }
        }
    }
    ECMout << "LINES " << Nfibers << " " << 3*Nfibers << endl;
    for (auto fiber : fibers_) {
        for (int i = 0; i < fiber->nodes_.size() - 1; i++) {
            Node * ni = fiber->nodes_[i];
            Node * nj = fiber->nodes_[i + 1];
            if (!box_->crossBoundary(ni->position_, nj->position_)) {
                ECMout << left << setw(6) << 2;
                ECMout << " " << left << setw(6) << ni->dumpID_;
                ECMout << " " << left << setw(6) << nj->dumpID_;
                ECMout << endl;
            }
        }
    }
    ECMout << endl;

    ECMout << "CELL_DATA " << Nfibers << endl;
    ECMout << "SCALARS strain double" << endl;
    ECMout << "LOOKUP_TABLE default" << endl;
    for (auto fiber : fibers_) {
        for (int i = 0; i < fiber->nodes_.size() - 1; i++) {
            Node * ni = fiber->nodes_[i];
            Node * nj = fiber->nodes_[i + 1];
            if (!box_->crossBoundary(ni->position_, nj->position_)) {
                double lt = 0.;
                for (int m = 0; m < 3; m++) {
                    lt = lt+pow ((ni->position_[m]-nj->position_[m]), 2.0) ;
                }
                double strain = (sqrt(lt) - fiberElasticity_->l0_) / fiberElasticity_->l0_;
                ECMout << left << scientific << setprecision(7) << strain << endl;
            }
        }
    }

    ECMout << endl;

    ECMout.close();

    //////////////////////////////////////////////////////////////////////////////////////
    stringstream Linkfilename;
    Linkfilename << dumpTag() << ".link.vtk";
    ofstream Linkout(Linkfilename.str().c_str());
    if (!Linkout.is_open()) {
        cout << "Error opening output file " << Linkfilename.str().c_str() << endl;
        exit(1);
    }
    Linkout << "# vtk DataFile Version 2.0" << endl;
    Linkout << "polydata" << endl;
    Linkout << "ASCII" << endl;
    Linkout << "DATASET POLYDATA" << endl;
    Linkout << "POINTS " << links_.size()*2 << " double" << endl;
    for (long int i = 0; i < links_.size(); i++) {
        Link * link = links_[i];
        // reset vertex id for dumping polygons
        Linkout << right << setw(12) << scientific << setprecision(5) << link->polygon_->center_[0];
        Linkout << " " << right << setw(12) << scientific << setprecision(5) << link->polygon_->center_[1];
        Linkout << " " << right << setw(12) << scientific << setprecision(5) << link->polygon_->center_[2];
        Linkout << endl;
        Linkout << right << setw(12) << scientific << setprecision(5) << link->node_->position_[0];
        Linkout << " " << right << setw(12) << scientific << setprecision(5) << link->node_->position_[1];
        Linkout << " " << right << setw(12) << scientific << setprecision(5) << link->node_->position_[2];
        Linkout << endl;
    }
    Linkout << endl;

    Linkout << "LINES " << links_.size() << " " << 3*links_.size() << endl;
    for (long int i = 0; i < links_.size(); i++) {
        Link * link = links_[i];
        Linkout << left << setw(6) << 2;
        Linkout << " " << left << setw(6) << 2*i;
        Linkout << " " << left << setw(6) << 2*i + 1;
        Linkout << endl;
    }
    Linkout << endl;

    Linkout << "CELL_DATA " << links_.size() << endl;
    Linkout << "SCALARS strain double" << endl;
    Linkout << "LOOKUP_TABLE default" << endl;
    for (long int i = 0; i < links_.size(); i++) {
        Link * link = links_[i];
        double linklength = 0.;
        for (int m = 0; m < 3; m++) {
            linklength = linklength + pow ((link->polygon_->center_[m]-link->node_->position_[m]), 2.0) ;
        }
        double currentL0 = max(fiberLink_->l0_, fiberLink_->l1_);
        double strain = (sqrt(linklength) - currentL0) / currentL0;
        Linkout << left << scientific << setprecision(7) << strain << endl;}
    Linkout << endl;

    Linkout.close();

    return 0;
}

int     Run::dumpLinkInfo(){
    stringstream Linkfilename;
    Linkfilename << dumpTag() << ".linkInfo.txt";
    ofstream Linkout(Linkfilename.str().c_str());
    if (!Linkout.is_open()) {
        cout << "Error opening output file " << Linkfilename.str().c_str() << endl;
        exit(1);
    }
    // CHANGED 2026-11-24: with passive linkers ('link_rest attach') a 3rd column saves each linker's rest
    // length, so the next run keeps it (loadLinks). Otherwise the file is as before.
    const bool saveRest = fiberLink_->restAtAttach_;
    Linkout << left << setw(12) <<"PolygonID";
    Linkout << left << setw(12) <<"NodeID";
    if (saveRest) Linkout << " RestLength";
    Linkout << endl;

    for (long int i = 0; i < links_.size(); i++) {
        Link * link = links_[i];
        Linkout << left << setw(12) << link->polygon_->id_;
        Linkout << left << setw(12) << link->node_->id_;
        if (saveRest) Linkout << " " << setprecision(numeric_limits<double>::max_digits10) << fiberLink_->restLength(link);
        Linkout << endl;
    }
    return 0;
}
int     Run::dumpCellCenter() {
    updateCellVertices();
    stringstream filename;
    filename << "cellCenter.txt";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    out << "time ";
    out << left << setw(12) << simulation_time_;
    out << endl;

    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        double center[3] = {0., 0., 0.};
        double reference[3];
        for (int m = 0; m < 3; m++) {
            reference[m] = cell->vertices_[0]->position_[m];
        }
        for (auto vertex : cell->vertices_) {
            double dx[3];
            for (int m = 0; m < 3; m++) {
                dx[m] = (vertex->position_[m] - reference[m]);
            }
            box_->resetDistance(dx);
            for (int m = 0; m < 3; m++) {
                center[m] = center[m] + dx[m];
            }
        }
        for (int m = 0; m < 3; m++) {
            center[m] = center[m]/cell->vertices_.size() + reference[m];
        }
        box_->resetPosition(center);
        out << left << setw(6) << cell->id_;
        out << " " << right << setw(12) << scientific << setprecision(5) << center[0];
        out << " " << right << setw(12) << scientific << setprecision(5) << center[1];
        out << " " << right << setw(12) << scientific << setprecision(5) << center[2];
        out << endl;
    }
    out << endl;

    out.close();

    return 0;
}

int     Run::dumpCellShapeIndex() {
    updateCellShapeIndex();
    stringstream filename;
    filename << "cellShapeIndex.txt";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    out << "time ";
    out << left << setw(12) << simulation_time_;
    double averageShapeIndex = 0.;
    int countCells = 0;
    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        averageShapeIndex += cell->shapeIndex_;
        countCells++;
    }
    averageShapeIndex /= countCells;
    out << " " << left << setw(12) << averageShapeIndex;
    out << endl;

    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        out << left << setw(6) << cell->id_;
        out << " " << cell->shapeIndex_;
        out << endl;
    }
    out << endl;

    out.close();

    return 0;
}

int     Run::dumpCellVolume() {
    stringstream filename;
    filename << "cellVolume.txt";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    out << setprecision(numeric_limits<double>::max_digits10) << "time ";
    out << left << setw(12) << simulation_time_;
    double totalVolume = 0.;
    int countCells = 0;
    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        totalVolume += cell->volume_;
        countCells++;
    }
    out << " " << left << setw(12) << totalVolume;
    out << endl;

    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        out << left << setw(6) << cell->id_;
        out << " " << cell->volume_;
        out << endl;
    }
    out << endl;

    out.close();

    return 0;
}

int     Run::dumpVertexForce() {
    stringstream filename;
    filename << dumpTag() << ".vertexForce.txt";
    //filename << "vertexForce.txt";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    //out << "time ";
    //out << left << setw(12) << simulation_time_;
    //out << endl;


    for (auto vertex : vertices_) {
        out << left << setw(6) << vertex->id_;
        for (int m = 0; m < 3; m++) {
            out << " " << right << setw(12) << scientific << setprecision(5) << (vertex->volumeForce_[m] + vertex->interfaceForce_[m] + vertex->linkForce_[m]);
        }
        out << endl;
    }

    out << endl;

    out.close();

    return 0;
}

int     Run::dumpTopo() {
    stringstream filename;
    // filename << dumpTag() << ".topo.txt";
    filename << dumpTag() << ".sample.topo";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    out << "time ";
    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, commit 1676eb89): full precision (17 digits) in the restart files
    // sample.topo and ECM.topo, so a run continued from them starts from exactly the same positions.
    // Old: setprecision(5), i.e. 6 digits, which nudged every position at every 1-unit restart.
    out << setprecision(numeric_limits<double>::max_digits10) << left << setw(12) << simulation_time_;
    out << " box";
    out << " " << left << setw(12) << box_->size_[0];
    out << " " << left << setw(12) << box_->size_[1];
    out << " " << left << setw(12) << box_->size_[2];
    out << endl;

    out << "vertices ";
    out << left << setw(12) << vertices_.size();
    out << endl;
    for (auto vertex : vertices_) {
        out << left << setw(6) << vertex->id_;
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << vertex->position_[0];
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << vertex->position_[1];
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << vertex->position_[2];
        out << endl;
    }

    out << "edges ";
    out << left << setw(12) << edges_.size();
    out << endl;
    for (auto edge : edges_) {
        out << left << setw(6) << edge->id_;
        for (auto vertex : edge->vertices_) {
            out << " " << right << setw(12) << scientific << setprecision(5) << vertex->id_;
        }
        out << endl;
    }

    out << "polygons ";
    out << left << setw(12) << polygons_.size();
    out << endl;
    for (auto polygon : polygons_) {
        out << left << setw(6) << polygon->id_;
        for (auto edge : polygon->edges_) {
            out << " " << right << setw(12) << scientific << setprecision(5) << edge->id_;
        }
        out << endl;
    }

    out << "cells ";
    out << left << setw(12) << cells_.size() - 1;
    out << endl;
    for (auto cell : cells_) {
        if (cell->type_ < 0) {
            continue;
        }
        out << left << setw(6) << cell->id_;
        for (auto polygon : cell->polygons_) {
            out << " " << right << setw(12) << scientific << setprecision(5) << polygon->id_;
        }
        out << " " << right << setw(6) << cell->type_;
        out << endl;
    }

    out << endl;
    out.close();

    return 0;
}

int     Run::dumpECM()  {
    stringstream filename;
    filename << dumpTag() << ".ECM.topo";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }
    // out << "time ";
    // out << setprecision(numeric_limits<double>::max_digits10) << left << setw(12) << simulation_time_;
    // out << " box";
    // out << " " << left << setw(12) << box_->size_[0];
    // out << " " << left << setw(12) << box_->size_[1];
    // out << " " << left << setw(12) << box_->size_[2];
    // out << endl;

    out << "nodes ";
    out << left << setw(12) << nodes_.size();
    out << endl;
    for (auto node : nodes_) {
        out << left << setw(6) << node->id_;
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << node->position_[0];
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << node->position_[1];
        out << " " << right << setw(12) << scientific << setprecision(numeric_limits<double>::max_digits10) << node->position_[2];
        out << " " << right << setw(3) << int(node->link_);
        out << endl;
    }

    out << "fibers ";
    out << left << setw(12) << fibers_.size();
    out << endl;
    for (auto fiber : fibers_) {
        out << left << setw(6) << fiber->id_;
        for (auto node : fiber->nodes_) {
            out << " " << right << setw(12) << scientific << setprecision(5) << node->id_;
        }
        out << endl;
    }
    out << endl;
    out.close();

    return 0;
}
int     Run::dumpReconnection() {
    stringstream filename;
    filename << "reconnections.txt";
    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }

    out << verboseReconnection_.str();
    verboseReconnection_.str("");

    out.close();

    return 0;
}

int     Run::dumpSpheroidShape() {
    stringstream filename;
    filename << "spheroidShape.txt";

    ofstream out(filename.str().c_str(), std::ios_base::app);
    if (!out.is_open()) {
        cout << "Error opening output file " << filename.str().c_str() << endl;
        exit(1);
    }

    out << left << setw(12) << simulation_time_;
    double area = 0.;
    for (auto polygon : polygons_) {
        if (polygon->type_ == 2) {
            area += polygon->area_;
        }
    }
    out << " " << left << setw(12) << area;        
    
    double volume = 0.;
    for (auto cell : cells_) {
        if (cell->type_ <= 0) {
            continue;
        }
        volume += cell->volume_;
    }
    out << " " << left << setw(12) << volume;
    out << " " << left << setw(12) << area * pow(volume, (-1.0)*2.0/3.0);
    out << endl;
    
    out.close();

    return 0;
}
