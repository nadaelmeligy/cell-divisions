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
#include <random>
#include <stdexcept>
#include "Run/Run.h"
#include "Run/MeshValidation.h"

using namespace std;

int     InitializeAll(string spheroid_filename,string ECM_filename, string cellParameters_filename, Run *);
int     LoadCellParameters(string filename, Run *);
int     LoadConf(string filename, Run *);

#ifndef TVM_NO_MAIN
int main(int argc, char *argv[]) {
    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, src at commit 1676eb89), extended: 'tvm --runtime-features' prints
    // which conf options this binary has, so run_growth.py can refuse to run an older binary silently.
    if (argc > 1 && string(argv[1]) == "--runtime-features") {
        // CHANGED 2026-11-24 (Matrigel-like mesh): v3 = also keeps several linkers on one ECM node under the
        // Python link rules (Run::loadLinks), needed for run_growth.py --link-node-capacity > 1. Old line:
        //     cout << "tvm-runtime-v2\n";  // CHANGED 2026-11-24: v2 = also accepts link_rest
        cout << "tvm-runtime-v3\n";
        return 0;
    }
    string spheroid_filename = "sample.topo";
    string ECM_filename = "ECM.topo";
    string cellParameters_filename = "cellParameters.input";
    // CHANGED 2026-11: file with the linkers to restart with (see Run::loadLinks). The engine
    // writes its linkers to NNNNNNN.linkInfo.txt; copy the last one to links.input to restart
    // with them. An optional 4th command-line argument gives a different file name.
    string links_filename = "links.input";
    if (argc > 1 && argc < 4) {
        cerr << "Usage: tvm [sample.topo ECM.topo cellParameters.input [links.input]]\n";
        return 1;
    }
    if (argc > 1) {
        spheroid_filename = argv[1];
        ECM_filename = argv[2];
        cellParameters_filename = argv[3];
    }
    if (argc > 4) {
        links_filename = argv[4];
    }
    Run * run = new Run();
    // CHANGED 2026-11-24: optional 4th argument = the links file (default links.input); errors are
    // reported and give exit code 1 instead of a crash.
    try {
        InitializeAll(spheroid_filename,ECM_filename,cellParameters_filename, run);
        run->updatePolygonVertices();
        // Restore any linkers saved by the previous segment.
        run->loadLinks(links_filename);
        return run->start();
    } catch (const std::exception& error) {
        cerr << "Simulation error: " << error.what() << endl;
        return 1;
    }
}
#endif

int InitializeAll(string spheroid_filename,string ECM_filename, string cellParameters_filename, Run * run) {
    printf("Initialization start ...\n");

    // load initial spheroid configuration
    ifstream topofile(spheroid_filename.c_str());
    if (!topofile.is_open()) {
        cout << "Error opening sample topo file" << endl;
        exit(1);
    }
    cout << "\n\nLoading spheroid from " << spheroid_filename <<"\n\n";
    string buffer;
    string delimiter = " ";
    size_t pos = 0;
    long int tmp_id;
    vector<string> tokens;
    vector<vector<string>> lines;

    while (getline(topofile, buffer))
    {
        pos = buffer.find((char)13);
        if (pos != string::npos) {
            buffer = buffer.substr(0, pos);
        }
        if (buffer.length() == 0) continue;

        tokens.clear();
        while ((pos = buffer.find(delimiter)) != string::npos) {
            string token = buffer.substr(0, pos);
            if (token.length() > 0) {
                tokens.push_back(token);
            }
            buffer.erase(0, pos + delimiter.length());
        }
        if (buffer.length() > 0) {
            tokens.push_back(buffer);
        }
        lines.push_back(tokens);
    }
    // if the first line starts with "time", ignore it
    if (!lines.empty() && !lines[0].empty() && lines[0][0] == "time") {
        cout << "ignoring first line of topo file starting with 'time'" << endl;
        lines.erase(lines.begin());
    }

    bool verticesFlag = false;
    bool edgesFlag = false;
    bool polygonsFlag = false;
    bool cellsFlag = false;
    for (int i = 0; i < lines.size(); i++) {
        tokens = lines[i];
        if (tokens[0] == "vertices") {
            verticesFlag = true;
        } else if (tokens[0] == "edges") {
            verticesFlag = false;
            edgesFlag = true;
        } else if (tokens[0] == "polygons") {
            edgesFlag = false;
            polygonsFlag = true;
        } else if (tokens[0] == "cells") {
            polygonsFlag = false;
            cellsFlag = true;
        } else {
            if (verticesFlag) {
                tmp_id = atol(tokens[0].c_str());
                Vertex * vertex = new Vertex(run, tmp_id);
                for (int j = 1; j < tokens.size(); j++) {
                    vertex->position_[j - 1] = atof(tokens[j].c_str());
                }
                run->vertices_.push_back(vertex);
            }
            if (edgesFlag) {
                tmp_id = atol(tokens[0].c_str());
                Edge * edge = new Edge(run, tmp_id);
                for (int j = 1; j < tokens.size(); j++) {
                    tmp_id = atol(tokens[j].c_str());
                    for (auto vertex : run->vertices_) {
                        if (vertex->id_ == tmp_id) {
                            edge->vertices_.push_back(vertex);
                            break;
                        }
                    }
                }
                run->edges_.push_back(edge);
            }
            if (polygonsFlag) {
                tmp_id = atol(tokens[0].c_str());
                Polygon * polygon = new Polygon(run, tmp_id);
                for (int j = 1; j < tokens.size(); j++) {
                    tmp_id = atol(tokens[j].c_str());
                    for (auto edge : run->edges_) {
                        if (edge->id_ == tmp_id) {
                            polygon->edges_.push_back(edge);
                            break;
                        }
                    }
                }
                run->polygons_.push_back(polygon);
            }
            if (cellsFlag) {
                tmp_id = atol(tokens[0].c_str());
                Cell * cell = new Cell(run, tmp_id);
                for (int j = 1; j < tokens.size() - 1; j++) {
                    tmp_id = atol(tokens[j].c_str());
                    for (auto polygon : run->polygons_) {
                        if (polygon->id_ == tmp_id) {
                            cell->polygons_.push_back(polygon);
                            break;
                        }
                    }
                }
                cell->type_ = atoi(tokens[tokens.size() - 1].c_str());
                run->cells_.push_back(cell);
            }
        }
    }

    for (auto vertex : run->vertices_) {
        if (run->count_vertices_ < vertex->id_ + 1) {
            run->count_vertices_ = vertex->id_ + 1;
        }
    }
    for (auto edge : run->edges_) {
        if (run->count_edges_ < edge->id_ + 1) {
            run->count_edges_ = edge->id_ + 1;
        }
    }
    for (auto polygon : run->polygons_) {
        if (run->count_polygons_ < polygon->id_ + 1) {
            run->count_polygons_ = polygon->id_ + 1;
        }
    }
    int nEmptyCells = 0;
    for (auto cell : run->cells_) {
        if (run->count_cells_ < cell->id_ + 1) {
            run->count_cells_ = cell->id_ + 1;
        }
        if (cell->type_ == 0) {
            nEmptyCells++;
        }
    }
    cout << "Number of vertices: " << run->vertices_.size() << endl;
    cout << "Maximum vertex ID: " << run->count_vertices_ - 1 << endl;
    cout << "Number of edges: " << run->edges_.size() << endl;
    cout << "Maximum edge ID: " << run->count_edges_ - 1 << endl;
    cout << "Number of polygons: " << run->polygons_.size() << endl;
    cout << "Maximum polygon ID: " << run->count_polygons_ - 1 << endl;
    cout << "Number of all cells: " << run->cells_.size() << endl;
    cout << "Number of empty cells: " << nEmptyCells << endl;
    cout << "Maximum cell ID: " << run->count_cells_ - 1 << endl;

    // create emptySpace and assign component polgons
    Cell * emptySpace = new Cell(run, -1);
    emptySpace->type_ = (-1);
    run->updatePolygonCells();
    long int nPolygon2N = 0;
    for (auto polygon : run->polygons_) {
        if (polygon->cells_.size() < 2) {
            emptySpace->polygons_.push_back(polygon);
        } else {
            nPolygon2N++;
        }
    }
    cout << "Number of polygons with two neighboring cells: " << nPolygon2N << "/" << run->polygons_.size() << endl;
    run->cells_.push_back(emptySpace);
    run->emptySpace_ = emptySpace;
    // assign emptySpace to vertices
    run->updateCellVertices();
    for (auto vertex : emptySpace->vertices_) {
        vertex->cells_.push_back(emptySpace);
    }

    // load initial ECM configuration
    ifstream ECMtopofile(ECM_filename);
    // CHANGED 2026-11-24: bug fix, this checked the cell file (topofile) instead of the ECM file.
    if (!ECMtopofile.is_open()) {
        cout << "Error opening ECM topo file" << endl;
        exit(1);
    }
    cout << "\n\nLoading ECM from " << ECM_filename <<"\n\n";

    tokens.clear();
    lines.clear();
    while (getline(ECMtopofile, buffer)) {
        pos = buffer.find((char)13);
        if (pos != string::npos) {
            buffer = buffer.substr(0, pos);
        }
        if (buffer.length() == 0) continue;

        tokens.clear();
        while ((pos = buffer.find(delimiter)) != string::npos) {
            string token = buffer.substr(0, pos);
            if (token.length() > 0) {
                tokens.push_back(token);
            }
            buffer.erase(0, pos + delimiter.length());
        }
        if (buffer.length() > 0) {
            tokens.push_back(buffer);
        }
        lines.push_back(tokens);
    }
    bool nodesFlag = false;
    bool fibersFlag = false;
    for (int i = 0; i < lines.size(); i++) {
        tokens = lines[i];
        if (tokens[0] == "nodes") {
            nodesFlag = true;
        } else if (tokens[0] == "fibers") {
            nodesFlag = false;
            fibersFlag = true;
        } else {
            if (nodesFlag) {
                tmp_id = atol(tokens[0].c_str());
                Node * node = new Node(run, tmp_id);
                for (int j = 1; j < tokens.size() - 1; j++) {
                    node->position_[j - 1] = atof(tokens[j].c_str());
                }
                if (atol(tokens[tokens.size() - 1].c_str()) > 0) {
                    node->link_ = true;
                }
                run->nodes_.push_back(node);
            }
            if (fibersFlag) {
                tmp_id = atol(tokens[0].c_str());
                Fiber * fiber = new Fiber(run, tmp_id);
                for (int j = 1; j < tokens.size(); j++) {
                    tmp_id = atol(tokens[j].c_str());
                    for (auto node : run->nodes_) {
                        if (node->id_ == tmp_id) {
                            fiber->nodes_.push_back(node);
                            node->fibers_.push_back(fiber);
                            break;
                        }
                    }
                }
                run->fibers_.push_back(fiber);
            }
        }
    }
    cout << "Number of nodes: " << run->nodes_.size() << endl;
    cout << "Number of fibers: " << run->fibers_.size() << endl;

    // initialize volume object
    run->volume_ = new Volume(run);
    // initialize interface object
    run->interface_ = new Interface(run);
    // initialize fiberElasticity object
    run->fiberElasticity_ = new FiberElasticity(run);
    // initialize fiberLink object
    run->fiberLink_ = new FiberLink(run);
    // CHANGED 2026-11: create the new cell-fiber contact force (off unless conf has a 'contact' line)
    run->cellFiberContact_ = new CellFiberContact(run);
//    // initialize contact object
//    run->contact_ = new Contact(run);
    // initialize reconnection object
    run->reconnection_ = new Reconnection(run);

    LoadConf("conf", run);
    LoadCellParameters(cellParameters_filename, run);

    // update geometry and topology information
    if (run->validateTopology_) MeshValidation::topology(*run);
    run->updateGeoinfo();
    run->updateVertexCells();
    run->updateEdgeCells();
    run->volume_->updatePolygonDirections();
    if (run->validateTopology_) {
        run->volume_->updateVolume();
        MeshValidation::positiveVolumes(*run);
    }
    run->updateEmptyCells();

    return 0;
}

int LoadCellParameters(string filename, Run * run){
    ifstream topofile(filename.c_str());
    if (!topofile.is_open()) {
        cout << filename << " not present or could not be processed.\n";
        cout << "Proceeding with homogeneous cell properties." << endl;
        return 0;
    }

    // Build id -> Cell* map so lookups are correct even with non-contiguous ids
    unordered_map<long int, Cell*> cell_by_id;
    cell_by_id.reserve(run->cells_.size());
    for (auto cell : run->cells_) {
        cell_by_id[cell->id_] = cell;
    }

    string buffer;
    vector<vector<string>> lines;
    vector<string> header;
    bool header_read = false;

    auto strip_and_tokenize = [](string buffer) -> vector<string> {
        size_t pos = buffer.find((char)13);
        if (pos != string::npos) {
            buffer = buffer.substr(0, pos);
        }
        vector<string> tokens;
        istringstream iss(buffer);
        string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        return tokens;
    };

    while (getline(topofile, buffer)) {
        vector<string> tokens = strip_and_tokenize(buffer);
        if (tokens.empty()) continue;
        if (!header_read) {
            header = tokens;
            header_read = true;
            continue;
        }
        lines.push_back(tokens);
    }

    if (!header_read) {
        cout << filename << " is empty or has no header." << endl;
        cout << "Proceeding with homogeneous cell properties." << endl;
        return 0;
    }

    int idx_id = -1, idx_v0 = -1, idx_s0 = -1, idx_kv = -1;
    for (size_t i = 0; i < header.size(); i++) {
        if (header[i] == "id")      idx_id = i;
        else if (header[i] == "v0") idx_v0 = i;
        else if (header[i] == "s0") idx_s0 = i;
        else if (header[i] == "kv") idx_kv = i;
    }

    if (idx_id < 0) {
        cout << filename << " file header missing required 'id' column." << endl;
        return 0;
    }
    cout << "\n\nLoading cell parameters from " << filename <<"\n\n";

    for (size_t i = 0; i < lines.size(); i++) {
        vector<string>& tokens = lines[i];
        if (tokens.size() != header.size()) {
            cout << "Skipping malformed line " << (i + 2)
                 << " in " << filename << " file (column count mismatch)." << endl;
            continue;
        }

        long int tmp_id = atol(tokens[idx_id].c_str());

        auto it = cell_by_id.find(tmp_id);
        if (it == cell_by_id.end()) {
            cout << "Skipping line " << (i + 2) << " in " << filename
                 << ": no cell with id " << tmp_id << " exists." << endl;
            continue;
        }
        Cell * cell = it->second;

        if (idx_v0 >= 0) cell->v0_ = atof(tokens[idx_v0].c_str());
        if (idx_s0 >= 0) cell->s0_ = atof(tokens[idx_s0].c_str());
        if (idx_kv >= 0) cell->kv_ = atof(tokens[idx_kv].c_str());
    }
    return 0;
}

int LoadConf(string filename, Run * run) {
    ifstream conf(filename.c_str());

    if (!conf.is_open()) {
        cout << "Error opening conf file" << endl;
        exit(1);
    }

    string buffer;
    string delimiter = " ";
    size_t pos = 0;
    vector<string> tokens;
    vector<vector<string>> lines;

    int time_written = 0;
    int dump_written = 0;
    int log_screen_written = 0;
    int s0_written = 0;
    int Lth_written = 0;
    int temperature_written = 0;
    int kv_written = 0;
    int box_written = 0;
    int fiber_written = 0;
    int link_written = 0;
    double reconnectInterval = 0.;
    double reconnectOrigin = 0.;
    bool hasReconnectInterval = false;
    bool hasReconnectOrigin = false;
    bool hasSeed = false;

    while (getline(conf, buffer))
    {
        pos = buffer.find((char)13);
        if (pos != string::npos) {
            buffer = buffer.substr(0, pos);
        }
        if ((buffer.length() == 0) || (buffer[0] == '#')) continue;

        tokens.clear();
        while ((pos = buffer.find(delimiter)) != string::npos) {
            string token = buffer.substr(0, pos);
            if (token.length() > 0) {
                tokens.push_back(token);
            }
            buffer.erase(0, pos + delimiter.length());
        }
        if (buffer.length() > 0) {
            tokens.push_back(buffer);
        }
        lines.push_back(tokens);
    }

    for (int i = 0; i < lines.size(); i++) {
        tokens = lines[i];
        if (tokens[0] == "time") {
            if (tokens.size() != 4) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->t_start_ = atof(tokens[1].c_str());
            run->t_end_ = atof(tokens[2].c_str());
            run->dt_ = atof(tokens[3].c_str());
            run->dte_ = 5*run->dt_;
            time_written = 1;
            cout << "time: " << run->t_start_ << " ~ " << run->t_end_ << " ~ " << run->dt_ << endl;
        }
        // CHANGED 2026-11-24 (Tao Zhang's Rust TVM): 'dtr T' = time between swap checks, independent of dt
        // (default 10 dt, as before); 'reconnect_origin T' = the time the checks count from (default t_start).
        else if ((tokens[0] == "dtr" || tokens[0] == "reconnect_origin") && tokens.size() == 2) {
            char* tail;
            double value = strtod(tokens[1].c_str(), &tail);
            if (*tail || !isfinite(value)) {
                cerr << "conf file error: invalid " << tokens[0] << endl;
                exit(1);
            }
            if (tokens[0] == "dtr") {
                reconnectInterval = value;
                hasReconnectInterval = true;
            } else {
                reconnectOrigin = value;
                hasReconnectOrigin = true;
            }
        }
        // CHANGED 2026-11-24 (Tao Zhang's Rust TVM): 'seed N' = seed of the run's noise stream.
        else if (tokens[0] == "seed" && tokens.size() == 2) {
            if (tokens[1].find_first_not_of("0123456789") != string::npos) {
                cerr << "conf file error: seed must be an unsigned 64-bit integer" << endl;
                exit(1);
            }
            try {
                run->rng_.seed(stoull(tokens[1]));
                hasSeed = true;
            } catch (const exception&) {
                cerr << "conf file error: invalid seed" << endl;
                exit(1);
            }
        }
        else if (tokens[0] == "dump") {
            if (tokens.size() != 3) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            // CHANGED 2026-11-24: 'dump topo P' writes the topology files only (no VTK); 'dump vtk P' as before.
            if (tokens[1] == "vtk" || tokens[1] == "topo") {
                run->dumpVtk_ = tokens[1] == "vtk";
                run->dump_period_ = atof(tokens[2].c_str());
                dump_written = 1;
                cout << "dump: " << tokens[1] << " " << run->dump_period_ << endl;
            }
        }
        else if (tokens[0] == "log") {
            if (tokens.size() != 2) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->log_period_ = atof(tokens[1].c_str());
            log_screen_written = 1;
            cout << "log: " << run->log_period_ << endl;
        }
        else if (tokens[0] == "s0") {
            if (tokens.size() != 3) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            double avg_s0 = atof(tokens[1].c_str());
            for (auto cell : run->cells_) {
                if (cell->type_){
                    cell->s0_ = avg_s0;
                }
            }
            run->interface_->kL_ = atof(tokens[2].c_str());
            s0_written = 1;
            cout << "Average real cell s0: " << avg_s0 << " global kL: " << run->interface_->kL_ << endl;
        }
        else if (tokens[0] == "Lth") {
            if (tokens.size() != 2 && tokens.size() != 3) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->reconnection_->Lth_ = atof(tokens[1].c_str());
            if (tokens.size() == 3) {
                if (tokens[2] == "verbose") {
                    run->reconnection_->verbose_ = true;
                } else {
                    cerr << "conf file error: ";
                    for (int j = 0; j < tokens.size(); j++) {
                        cerr << tokens[j] << " ";
                    }
                    cerr << endl;
                    exit(1);
                }
            }
            Lth_written = 1;
            cout << "Lth: " << run->reconnection_->Lth_ << " verbose: " << run->reconnection_->verbose_ << endl;
        }
        else if (tokens[0] == "T") {
            if (tokens.size() != 2) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->temperature_ = atof(tokens[1].c_str());
            temperature_written = 1;
            cout << "temperature: " << run->temperature_ << endl;
        }
        else if (tokens[0] == "kv") {
            if (tokens.size() != 2) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            double avg_kv = atof(tokens[1].c_str());
             for (auto cell : run->cells_) {
                if (cell->type_){
                    cell->kv_ = avg_kv;
                }
            }
            kv_written = 1;
            cout << "Average real cell kv: " << avg_kv << endl;
        }
        else if (tokens[0] == "fiber") {
            // CHANGED 2026-11: an optional 4th number picks the bending energy (FiberElasticity.h):
            //     fiber ks l0 kb [bendForm]     0 = original (default), 1 = worm-like chain
            // Old: if (tokens.size() != 4) {
            if (tokens.size() == 5) {
                run->fiberElasticity_->bendForm_ = atoi(tokens[4].c_str());
            }
            if ((tokens.size() != 4 && tokens.size() != 5) ||
                (tokens.size() == 5 && tokens[4] != "0" && tokens[4] != "1")) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->fiberElasticity_->ks_ = atof(tokens[1].c_str());
            run->fiberElasticity_->l0_ = atof(tokens[2].c_str());
            run->fiberElasticity_->kb_ = atof(tokens[3].c_str());
            fiber_written = 1;
            // CHANGED 2026-11: also prints the bending energy form.
            // Old: cout << "fiber ks: " << ... << " kb: " << run->fiberElasticity_->kb_ << endl;
            cout << "fiber ks: " << run->fiberElasticity_->ks_ << " l0: " << run->fiberElasticity_->l0_ << " kb: " << run->fiberElasticity_->kb_
                 << " bend: " << (run->fiberElasticity_->bendForm_ == 1 ? "worm-like chain kb (1 + cos)" : "original 0.5 kb (1 + cos)^2") << endl;
        }
        else if (tokens[0] == "link") {
            // CHANGED 2026-11: the link line may now have a 7th number, the time at which the
            // linkers are created (it used to be 500, typed into Run.cpp):
            //     link N ks l0 l1 shrinkSpeed shrinkStartTime [linkStartTime]
            // Without it the time stays 500, so old conf files work unchanged.
            // Old: if (tokens.size() != 7) {
            if (tokens.size() == 8) {
                run->fiberLink_->linkStartTime_ = atof(tokens[7].c_str());
            }
            if (tokens.size() != 7 && tokens.size() != 8) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->fiberLink_->N_ = atoi(tokens[1].c_str());
            run->fiberLink_->ks_ = atof(tokens[2].c_str());
            run->fiberLink_->l0_ = atof(tokens[3].c_str());
            run->fiberLink_->l1_ = atof(tokens[4].c_str());
            run->fiberLink_->shrinkSpeed_ = atof(tokens[5].c_str());
            run->fiberLink_->shrinkStartTime_ = atof(tokens[6].c_str());
            link_written = 1;
            // CHANGED 2026-11: this print now also shows the link start time (last line).
            cout << "N: " << run->fiberLink_->N_
            << " link ks: " << run->fiberLink_->ks_
            << " l0: " << run->fiberLink_->l0_
            << " l1: " << run->fiberLink_->l1_
            << " shrink start time: " << run->fiberLink_->shrinkStartTime_
            << " shrink speed: " << run->fiberLink_->shrinkSpeed_
            << " link start time: " << run->fiberLink_->linkStartTime_ << endl;
        }
        else if (tokens[0] == "box") {
            if (tokens.size() != 7) {
                cerr << "conf file error: ";
                for (int j = 0; j < tokens.size(); j++) {
                    cerr << tokens[j] << " ";
                }
                cerr << endl;
                exit(1);
            }
            run->box_ = new Box(run);
            for (int k = 0; k < 3; k++) {
                run->box_->size_[k] = atof(tokens[k + 1].c_str());
                if (run->box_->size_[k] < (-1.0e-6)) {
                    cerr << "conf file error: ";
                    for (int j = 0; j < tokens.size(); j++) {
                        cerr << tokens[j] << " ";
                    }
                    cerr << endl;
                    exit(1);
                }
                if (tokens[k + 4] == "p") {
                    run->box_->boundaryCondition_[k] = true;
                } else if (tokens[k + 4] == "f"){
                    run->box_->boundaryCondition_[k] = false;
                } else {
                    cerr << "conf file error: ";
                    for (int j = 0; j < tokens.size(); j++) {
                        cerr << tokens[j] << " ";
                    }
                    cerr << endl;
                    exit(1);
                }
            }
            box_written = 1;
            cout << "box: " << run->box_->size_[0] << " " << run->box_->size_[1] << " " << run->box_->size_[2] << " ";
            cout << "periodic boundary condition: " << run->box_->boundaryCondition_[0] << " " << run->box_->boundaryCondition_[1] << " " << run->box_->boundaryCondition_[2] << endl;
        }
        // CHANGED 2026-11: two new, optional conf lines (without them nothing changes).
        //   contact k range [spacing]  turns on the cell-fiber contact force (Energy/CellFiberContact)
        //   linkrules python           Python decides the linkers (linkers.py); 'engine' = original rules
        else if (tokens[0] == "contact" && (tokens.size() == 3 || tokens.size() == 4)) {
            run->cellFiberContact_->k_ = atof(tokens[1].c_str());
            run->cellFiberContact_->range_ = atof(tokens[2].c_str());
            if (tokens.size() == 4) run->cellFiberContact_->spacing_ = atof(tokens[3].c_str());
            if (run->cellFiberContact_->k_ < 0. || run->cellFiberContact_->range_ <= 0. ||
                run->cellFiberContact_->spacing_ < 0.01) {
                cerr << "conf file error: contact k range [spacing] needs k >= 0, range > 0, spacing >= 0.01" << endl;
                exit(1);
            }
            cout << "cell-fiber contact: k " << run->cellFiberContact_->k_ << " range " << run->cellFiberContact_->range_
                 << " spacing " << run->cellFiberContact_->spacing_ << endl;
        }
        // CHANGED 2026-11-25 (speed): optional 'contact_search grid|brute' (default grid). 'brute' is the
        // original search over every surface triangle, kept to check that the grids give the same forces.
        else if (tokens[0] == "contact_search" && tokens.size() == 2 && (tokens[1] == "grid" || tokens[1] == "brute")) {
            run->cellFiberContact_->bruteSearch_ = tokens[1] == "brute";
            cout << "cell-fiber contact search: " << tokens[1] << endl;
        }
        // CHANGED 2026-11-24 (Tao Zhang's Rust TVM): 'face_center_forces exact|legacy' (default legacy).
        else if (tokens[0] == "face_center_forces" && tokens.size() == 2 &&
                 (tokens[1] == "exact" || tokens[1] == "legacy")) {
            run->exactFaceCenterForces_ = tokens[1] == "exact";
            cout << "face-center forces: " << tokens[1] << endl;
        }
        // CHANGED 2026-11-24 (Tao Zhang's Rust TVM): 'validate_topology on|off' (default off).
        else if (tokens[0] == "validate_topology" && tokens.size() == 2 &&
                 (tokens[1] == "on" || tokens[1] == "off")) {
            run->validateTopology_ = tokens[1] == "on";
        }
        else if (tokens[0] == "linkrules" && tokens.size() == 2 && (tokens[1] == "python" || tokens[1] == "engine")) {
            run->linkRulesPython_ = (tokens[1] == "python");
            cout << "link rules: " << tokens[1] << endl;
        }
        // CHANGED 2026-11-24: 'link_rest attach' = passive linkers (each linker's rest length is its own
        // length when it attaches); 'link_rest fixed' (default) = every linker uses the link line's l0
        else if (tokens[0] == "link_rest" && tokens.size() == 2 && (tokens[1] == "attach" || tokens[1] == "fixed")) {
            run->fiberLink_->restAtAttach_ = (tokens[1] == "attach");
            cout << "linker rest length: " << (run->fiberLink_->restAtAttach_ ? "own length at attachment (passive)" : "fixed l0") << endl;
        }
        else {
            cerr << "conf file error: ";
            for (int j = 0; j < tokens.size(); j++) {
                cerr << tokens[j] << " ";
            }
            cerr << endl;
            exit(1);
        }
    }

    if (conf.bad() || !conf.eof()) {
        cout << "Error reading file [" << filename << "]" << endl;
        exit(1);
    }

    if (time_written == 0) {
        cout << "conf file error: missing time" << endl;
        exit(1);
    }
    run->dtr_ = hasReconnectInterval ? reconnectInterval : 10 * run->dt_;
    // CHANGED 2026-11-24: default origin t_start + dt - dtr = the original schedule (a swap check right after the
    // first step, then every dtr), so a conf without 'reconnect_origin' runs exactly as the original engine.
    // run_growth.py always sets it (to the run's start time), so swap checks do not depend on restarts.
    run->reconnectOrigin_ = hasReconnectOrigin ? reconnectOrigin : run->t_start_ + run->dt_ - run->dtr_;
    // CHANGED 2026-11-24: one noise stream per run (Tao's TVM). 'seed N' in conf makes it repeatable. Without
    // it a seed is drawn at random, as the original drew fresh noise every step, so repeated runs still
    // differ; the seed is printed so a run can be repeated. A rng.input file (restart) overrides both.
    if (!hasSeed) {
        std::random_device device;
        unsigned long long seed = (static_cast<unsigned long long>(device()) << 32) ^ device();
        run->rng_.seed(seed);
        cout << "noise seed: " << seed << " (no 'seed' line in conf: drawn at random)" << endl;
    }
    if (!isfinite(run->dt_) || run->dt_ <= 0. || !isfinite(run->t_start_) ||
        !isfinite(run->t_end_) || run->t_end_ < run->t_start_ ||
        !isfinite(run->dtr_) || run->dtr_ < run->dt_ || run->reconnectOrigin_ > run->t_start_ ||
        !isfinite((run->t_end_ - run->reconnectOrigin_) / run->dtr_) ||
        (run->t_end_ - run->reconnectOrigin_) / run->dtr_ >= static_cast<double>(numeric_limits<long>::max()) ||
        (run->t_end_ - run->t_start_) / run->dt_ >= static_cast<double>(numeric_limits<long>::max()) ||
        (dump_written && (!isfinite(run->dump_period_) || run->dump_period_ < run->dt_)) ||
        (log_screen_written && (!isfinite(run->log_period_) || run->log_period_ < run->dt_))) {
        cerr << "conf file error: invalid simulation times or intervals" << endl;
        exit(1);
    }
    cout << "reconnection interval: " << run->dtr_ << endl;

    if (dump_written == 0) {
        cout << "conf file error: missing dump" << endl;
        exit(1);
    }

    if (log_screen_written == 0) {
        cout << "conf file error: missing log screen" << endl;
        exit(1);
    }

    if (s0_written == 0) {
        cout << "conf file error: s0" << endl;
        exit(1);
    }

    if (Lth_written == 0) {
        cout << "conf file error: Lth" << endl;
        exit(1);
    }

    if (temperature_written == 0) {
        cout << "conf file error: temperature" << endl;
        exit(1);
    }

    if (kv_written == 0) {
        cout << "conf file error: kv" << endl;
        exit(1);
    }

    if (box_written == 0) {
        cout << "conf file error: box" << endl;
        exit(1);
    }

    if (fiber_written == 0) {
        cout << "conf file error: fiber" << endl;
        exit(1);
    }

    if (link_written == 0) {
        cout << "conf file error: link" << endl;
        exit(1);
    }

    conf.close();

    return 0;
}
