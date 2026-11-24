/* CHANGED 2026-11 (new file): contact force between ECM fibers and the organoid surface.
 *
 * Why: in the original engine the fibers felt the cells only through the linker springs, so a
 * growing organoid could pass straight through every fiber it was not linked to. (Run.h refers to
 * a Contact module, but it is commented out and is not in the repository.)
 *
 * Turned on by the conf line   contact k range [spacing]
 *   k       stiffness per unit fiber length (k = 0 or no conf line: contact is off)
 *   range   fiber "thickness": a fiber point closer than this to a cell face is pushed out
 *   spacing distance between test points along each fiber segment (default 0.5)
 * With contact off every contact force is exactly zero, so results are identical to the
 * original engine. See CellFiberContact.cpp for how the force is computed.
 */
#ifndef CELLFIBERCONTACT_H_INCLUDED
#define CELLFIBERCONTACT_H_INCLUDED

class CellFiberContact;
#include "../Run/Run.h"

class CellFiberContact {
public:
    double k_;        // stiffness per unit fiber length (0 = off)
    double range_;    // no fiber point may come closer than this to a cell face
    double spacing_;  // distance between test points along a fiber segment
    double maxDepth_; // points more than this far inside the organoid are ignored (fixed at 1.0)
    double energy_;   // contact energy of the last step: sum of 1/2 (k ds) overlap^2
    long int nContacts_;
    // CHANGED 2026-11-25 (speed): the nearest-triangle search and the inside/outside ray test use
    // grids, so each fiber point looks only at nearby triangles instead of all of them (the same
    // forces, bit for bit: see CellFiberContact.cpp). true = the original search over every
    // triangle (conf line 'contact_search brute'), kept for checks.
    bool bruteSearch_;

    explicit CellFiberContact(Run *);
    int updateForces();

private:
    Run * run_;
};

#endif
