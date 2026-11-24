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

#ifndef FIBERLINK_H_INCLUDED
#define FIBERLINK_H_INCLUDED

class FiberLink;
#include "../Run/Run.h"

class FiberLink {
public:
    double ks_;
    double l0_;
    double l1_;
    int N_;
    double stretchEnergy_;
    double minArea_;
    double minEdgeLength_;
    double maxEdgeLength_;
    long int nFormedLinks_;
    double shrinkSpeed_;
    double shrinkStartTime_;
    // CHANGED 2026-11: time at which the engine creates the linkers. It was the number 500 typed
    // into Run.cpp; now it is the optional 7th number of the 'link' line in conf (default 500).
    double linkStartTime_;
    // CHANGED 2026-11-24: conf 'link_rest attach' -> true = passive linkers. Each linker's rest length is its
    // own length when it attaches (Link::restLength_), so a new linker carries no tension and only resists
    // being stretched further; the shrink rule (which changes l0_) does not touch these rest lengths.
    // Default false ('link_rest fixed'): every linker uses l0_ from the 'link' line, as before.
    bool restAtAttach_ = false;
    double linkLength(Link *);   // current length |node - face centre| (periodic box)
    double restLength(Link *);   // rest length used for this linker (passive: set on first use)

    explicit FiberLink(Run *);

    int updateForces();
    int updateEnergy();
private:
    Run * run_;
};

#endif
