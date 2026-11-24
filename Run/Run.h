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

#ifndef RUN_H_INCLUDED
#define RUN_H_INCLUDED

class Run;
#include <vector>
#include <sstream>
#include "Random.h"
#include "../Vertex/Vertex.h"
#include "../Edge/Edge.h"
#include "../Polygon/Polygon.h"
#include "../Cell/Cell.h"
#include "../Energy/Volume.h"
#include "../Energy/Interface.h"
#include "../Reconnection/Reconnection.h"
#include "../ECM/Node.h"
#include "../ECM/Fiber.h"
#include "../ECM/Link.h"
#include "../Energy/FiberElasticity.h"
#include "../Energy/FiberLink.h"
#include "../Energy/CellFiberContact.h"  // CHANGED 2026-11: new cell-fiber contact force
//#include "../Contact/Contact.h"
#include "Box.h"

class Run {
  public:
    double  dt_;    // integration time step
    double  dtr_;   // time interval of network reconnection
    double  dte_;   // time interval of updating empty cells dynamics
    double  mu_;   // inverse damping coefficient of vertex
    double  kB_;
    double  temperature_;
    int     NCell_;
    double pullForce_;
    double pullxMax_;
    double pullStartTime_;
    double pullxSelect_;
    double  simulation_time_;
    double   t_start_;
    double   t_end_;
    long int count_dump_;
    double   dump_period_;
    long int count_log_;
    double   log_period_;
    long int count_reconnect_;
    long int count_vertices_ = 0;
    long int count_edges_ = 0;
    long int count_polygons_ = 0;
    long int count_cells_ = 0;
    Cell * emptySpace_;
    Volume * volume_;
    Interface * interface_;
    FiberElasticity * fiberElasticity_;
    FiberLink * fiberLink_;
    CellFiberContact * cellFiberContact_;  // CHANGED 2026-11: new cell-fiber contact force
//    Contact * contact_;
    Reconnection * reconnection_;
    Box * box_;
    std::stringstream verboseReconnection_;

    std::vector<Vertex *> vertices_;
    std::vector<Edge *> edges_;
    std::vector<Polygon *> polygons_;
    std::vector<Cell *> cells_;
    std::vector<Node *> nodes_;
    std::vector<Fiber *> fibers_;
    std::vector<Link *> links_;
    std::vector<Polygon *> boundaryPolygons_;
    std::vector<Edge *> boundaryEdges_;
    // CHANGED 2026-11: restarting with the previous run's linkers (see Run::loadLinks)
    bool linksLoaded_;                     // true if loadLinks() read at least one linker from links.input
    std::vector<Node *> pendingFreeNodes_; // engine rules: ECM nodes loadLinks() could not put back on
                                           // their saved face; checkLinks() re-attaches them
    // CHANGED 2026-11: conf line 'linkrules python'. The engine then never creates or re-attaches
    // linkers itself: it starts from the linkers in links.input (several per cell allowed, one per
    // face) and removes one only when its face is deleted. Python (linkers.py) forms and breaks
    // them between runs. Default false = the original engine rules.
    bool linkRulesPython_;
    // CHANGED 2026-11-24 (Tao Zhang's Rust TVM, src at commit 1676eb89):
    bool exactFaceCenterForces_;      // conf 'face_center_forces exact': exact face-centre derivative (off = original)
    bool validateTopology_ = false;   // conf 'validate_topology on': check the mesh and every proposed swap
    bool dumpVtk_ = true;             // conf 'dump topo ...': skip the VTK files (topology files only)
    double reconnectOrigin_ = 0.;     // conf 'reconnect_origin': swap checks at origin + k dtr (restart-proof)
    Random rng_;                      // one seeded noise stream per run (Random.h), saved in rng.output
    long int reconnectSweeps_ = 0;    // number of swap checks, written to run_summary.txt

    Run();
    int     start();
    int     updatePolygonVertices();
    int     updatePolygonCells();
    int     updateCellVertices();
    int     updateCellShapeIndex();
    int     updateVertexEdges();
    int     updateVertexCells();
    int     updateEdgeCells();
    int     updateGeoinfo();
    int     updateEmptyCells();
    int     updateVerticesVelocity();
    int     updateVerticesPosition(double step);
    int     updateNodesVelocity();
    int     updateNodesPosition(double step);
    int     initializeLinks();
    int     loadLinks(std::string filename);  // CHANGED 2026-11: new, see Run.cpp
    int     checkLinks();
    int     updateLinks();
    int     deleteVertex(Vertex *);
    int     deleteEdge(Edge *);
    int     deletePolygon(Polygon *);
    Edge *  addEdge(Vertex *, Vertex *);
    int     dumpConfigurationVtk();
    int     dumpCellCenter();
    int     dumpCellShapeIndex();
    int     dumpCellVolume();
    int     dumpLinkInfo();
    int     dumpVertexForce();
    int     dumpTopo();
    int     dumpECM();
    int     dumpReconnection();
    int     dumpSpheroidShape();
    std::string dumpTag() const;
};

#endif
