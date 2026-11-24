/* CHANGED 2026-11 (new file): contact force between ECM fibers and the organoid surface.
 * (Why it exists and how it is switched on: see CellFiberContact.h.)
 *
 * Every time step:
 * 1. The organoid surface is every face between a real cell and the surrounding medium
 *    (polygon type 2). Each face is split into fan triangles (face centre + two neighbouring
 *    corners), the same triangles the engine uses for cell volume and area. Each triangle gets
 *    an outward unit normal, oriented with the owning cell's polygonDirections_ (as in
 *    Volume::updatePolygonForces).
 * 2. Every fiber segment near the organoid is sampled every `spacing` along its length.
 * 3. For each test point x: find the nearest point q on the surface (distance d), then decide
 *    whether x is outside or inside the organoid:
 *    - q inside a triangle: x is outside if (x - q) . n >= 0, with n that triangle's normal (exact).
 *    - q on an edge or a corner: count how often three fixed rays from x cross the surface; an odd
 *      count means inside (majority of the three rays). Edge and corner normals are not used,
 *      because in a vertex model two organoid faces can meet at a single corner with medium cells
 *      in between, and there the usual corner normal (Baerentzen & Aanaes 2005) gives the wrong side;
 *      in tests that turned far-outside fiber points into "inside" ones and blew the run up.
 * 4. Outside and d < range:  push x away from q with F = k ds (range - d).
 *    Inside and d < maxDepth: push x out towards q with F = k ds (range + d).
 *    ds = segment length / number of test points, so k is a stiffness per unit fiber length and
 *    the total force does not depend on `spacing`.
 * 5. The force on x is shared by the segment's two nodes (weights 1-w and w). The opposite force
 *    goes to the face through the weights of q: to the two triangle corners, and the face centre's
 *    share spread equally over all the face's corners. Total force (momentum) is conserved.
 *
 * CHANGED 2026-11-25 (speed): step 3 used to look at every surface triangle for every test point
 * (and three rays against every triangle). It now uses grids: Grid3 (triangles by face centre) for
 * the nearest point, Grid2 (triangle shadows along each ray) for the ray test. They return the same
 * triangles, so the forces are the same to the last bit; 'contact_search brute' in conf runs the
 * original loops (checked: experiments/speed/check_contact_grid.py).
 */
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <algorithm>
#include "CellFiberContact.h"

using namespace std;

namespace {
struct Tri {
    double c[3], a[3], b[3];   // face centre and two neighbouring corners
    double n[3];               // outward unit normal of this triangle
    double reach2;             // squared search radius around c
    Polygon * polygon;
    Vertex * va;
    Vertex * vb;
};

inline double dot3(const double * x, const double * y) { return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]; }

// does the ray o + s * dir (s > 0) cross triangle (p0, p1, p2)? (Moller-Trumbore)
bool rayCrosses(const double * o, const double * dir, const double * p0, const double * p1, const double * p2) {
    double e1[3], e2[3], h[3], sv[3], q[3];
    for (int m = 0; m < 3; m++) { e1[m] = p1[m] - p0[m]; e2[m] = p2[m] - p0[m]; }
    h[0] = dir[1]*e2[2] - dir[2]*e2[1]; h[1] = dir[2]*e2[0] - dir[0]*e2[2]; h[2] = dir[0]*e2[1] - dir[1]*e2[0];
    double a = dot3(e1, h);
    if (fabs(a) < 1e-15) return false;
    double f = 1.0 / a;
    for (int m = 0; m < 3; m++) sv[m] = o[m] - p0[m];
    double u = f * dot3(sv, h);
    if (u < 0. || u > 1.) return false;
    q[0] = sv[1]*e1[2] - sv[2]*e1[1]; q[1] = sv[2]*e1[0] - sv[0]*e1[2]; q[2] = sv[0]*e1[1] - sv[1]*e1[0];
    double v = f * dot3(dir, q);
    if (v < 0. || u + v > 1.) return false;
    return f * dot3(e2, q) > 1e-12;
}

// closest point q on triangle (p0, p1, p2) to x, with barycentric weights w (Ericson,
// Real-Time Collision Detection, section 5.1.5)
void closestOnTriangle(const double * x, const double * p0, const double * p1, const double * p2,
                       double * q, double * w) {
    double ab[3], ac[3], ap[3];
    for (int m = 0; m < 3; m++) { ab[m] = p1[m]-p0[m]; ac[m] = p2[m]-p0[m]; ap[m] = x[m]-p0[m]; }
    double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) { w[0]=1; w[1]=0; w[2]=0; }
    else {
        double bp[3]; for (int m = 0; m < 3; m++) bp[m] = x[m]-p1[m];
        double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
        if (d3 >= 0 && d4 <= d3) { w[0]=0; w[1]=1; w[2]=0; }
        else {
            double vc = d1*d4 - d3*d2;
            if (vc <= 0 && d1 >= 0 && d3 <= 0) { double v = d1/(d1-d3); w[0]=1-v; w[1]=v; w[2]=0; }
            else {
                double cp[3]; for (int m = 0; m < 3; m++) cp[m] = x[m]-p2[m];
                double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
                if (d6 >= 0 && d5 <= d6) { w[0]=0; w[1]=0; w[2]=1; }
                else {
                    double vb = d5*d2 - d1*d6;
                    if (vb <= 0 && d2 >= 0 && d6 <= 0) { double t = d2/(d2-d6); w[0]=1-t; w[1]=0; w[2]=t; }
                    else {
                        double va = d3*d6 - d5*d4;
                        if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
                            double t = (d4-d3)/((d4-d3)+(d5-d6)); w[0]=0; w[1]=1-t; w[2]=t;
                        } else {
                            double den = 1.0/(va+vb+vc); double v = vb*den, t = vc*den;
                            w[0]=1-v-t; w[1]=v; w[2]=t;
                        }
                    }
                }
            }
        }
    }
    for (int m = 0; m < 3; m++) q[m] = w[0]*p0[m] + w[1]*p1[m] + w[2]*p2[m];
}

// CHANGED 2026-11-25 (speed): the three fixed ray directions of the inside/outside test, moved here
// from updateForces (same numbers) so that the ray grids below can use them as well.
const double kRays[3][3] = {{0.2673, 0.5345, 0.8018}, {-0.8018, 0.2673, 0.5345}, {0.5345, -0.8018, 0.2673}};

// CHANGED 2026-11-25 (speed): uniform grids of triangle indices (compressed rows: cell k holds
// idx[off[k] .. off[k+1]), filled in increasing triangle order).
// Grid3: each triangle in the cell of its face centre c, for the nearest-triangle search.
struct Grid3 {
    double lo[3], h;
    int n[3];
    vector<int> off, idx;
    int cellIndex(const double * p, int m) const {
        int i = (int)floor((p[m] - lo[m]) / h);
        return i < 0 ? 0 : (i >= n[m] ? n[m] - 1 : i);
    }
};
// Grid2: for one ray direction d, each triangle in every cell that its shadow (its projection on
// the plane perpendicular to d, as a padded box) covers. A ray from x can only cross triangles
// whose shadow contains the shadow of x, and those are all in x's cell.
struct Grid2 {
    double e1[3], e2[3], lo[2], h;
    int n[2];
    vector<int> off, idx;
};

void unit3(double * v) { double l = sqrt(dot3(v, v)); for (int m = 0; m < 3; m++) v[m] /= l; }
void cross3(const double * a, const double * b, double * c) {
    c[0] = a[1]*b[2] - a[2]*b[1]; c[1] = a[2]*b[0] - a[0]*b[2]; c[2] = a[0]*b[1] - a[1]*b[0];
}

Grid2 buildRayGrid(const vector<Tri> & tris, const double * d) {
    Grid2 g;
    double axis[3] = {0., 0., 0.};
    int least = 0;
    for (int m = 1; m < 3; m++) if (fabs(d[m]) < fabs(d[least])) least = m;
    axis[least] = 1.;
    cross3(d, axis, g.e1); unit3(g.e1);
    cross3(d, g.e1, g.e2); unit3(g.e2);
    const double pad = 1e-9;   // far above the rounding of the projections (~1e-14 here)
    size_t nt = tris.size();
    vector<double> box(4 * nt);
    double lo[2] = {1e30, 1e30}, hi[2] = {-1e30, -1e30};
    for (size_t i = 0; i < nt; i++) {
        const double * p[3] = {tris[i].c, tris[i].a, tris[i].b};
        double b[4] = {1e30, -1e30, 1e30, -1e30};
        for (int k = 0; k < 3; k++) {
            double u = dot3(p[k], g.e1), v = dot3(p[k], g.e2);
            b[0] = min(b[0], u); b[1] = max(b[1], u); b[2] = min(b[2], v); b[3] = max(b[3], v);
        }
        b[0] -= pad; b[1] += pad; b[2] -= pad; b[3] += pad;
        for (int k = 0; k < 4; k++) box[4 * i + k] = b[k];
        lo[0] = min(lo[0], b[0]); hi[0] = max(hi[0], b[1]); lo[1] = min(lo[1], b[2]); hi[1] = max(hi[1], b[3]);
    }
    g.h = 0.5;
    for (int a = 0; a < 2; a++) { g.lo[a] = lo[a]; g.n[a] = (int)floor((hi[a] - lo[a]) / g.h) + 1; }
    g.off.assign((size_t)g.n[0] * g.n[1] + 1, 0);
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            for (size_t k = 1; k < g.off.size(); k++) g.off[k] += g.off[k - 1];
            g.idx.resize(g.off.back());
        }
        vector<int> fill(pass == 1 ? g.off.begin() : g.off.end(), g.off.end());
        for (size_t i = 0; i < nt; i++) {
            int u0 = (int)floor((box[4*i] - g.lo[0]) / g.h), u1 = (int)floor((box[4*i+1] - g.lo[0]) / g.h);
            int v0 = (int)floor((box[4*i+2] - g.lo[1]) / g.h), v1 = (int)floor((box[4*i+3] - g.lo[1]) / g.h);
            for (int iu = u0; iu <= u1; iu++) for (int iv = v0; iv <= v1; iv++) {
                size_t c = (size_t)iu * g.n[1] + iv;
                if (pass == 0) g.off[c + 1]++;
                else g.idx[fill[c]++] = (int)i;
            }
        }
    }
    return g;
}

// number of triangles that the ray x + s d (s > 0) crosses, using the ray grid of d
int countCrossings(const vector<Tri> & tris, const Grid2 & g, const double * x, const double * d) {
    int iu = (int)floor((dot3(x, g.e1) - g.lo[0]) / g.h), iv = (int)floor((dot3(x, g.e2) - g.lo[1]) / g.h);
    if (iu < 0 || iu >= g.n[0] || iv < 0 || iv >= g.n[1]) return 0;   // no triangle's shadow is there
    size_t c = (size_t)iu * g.n[1] + iv;
    int crossings = 0;
    for (int k = g.off[c]; k < g.off[c + 1]; k++) {
        const Tri & t = tris[g.idx[k]];
        crossings += rayCrosses(x, d, t.c, t.a, t.b);
    }
    return crossings;
}
}  // namespace

CellFiberContact::CellFiberContact(Run * run) {
    run_ = run;
    k_ = 0.;
    range_ = 0.2;
    spacing_ = 0.5;
    maxDepth_ = 1.0;
    energy_ = 0.;
    nContacts_ = 0;
    bruteSearch_ = false;   // CHANGED 2026-11-25 (speed): grids by default; true = original search
}

int CellFiberContact::updateForces() {
    for (auto v : run_->vertices_) for (int m = 0; m < 3; m++) v->contactForce_[m] = 0.;
    for (auto n : run_->nodes_) for (int m = 0; m < 3; m++) n->contactForce_[m] = 0.;
    energy_ = 0.;
    nContacts_ = 0;
    if (k_ <= 0.) return 0;

    double margin = max(range_, maxDepth_);

    // 1. surface triangles and their outward normals
    vector<Tri> tris;
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    for (auto polygon : run_->polygons_) {
        if (polygon->type_ != 2) continue;
        Cell * owner = NULL;
        for (auto cell : polygon->cells_) if (cell->type_ > 0) owner = cell;
        size_t nv = polygon->vertices_.size();
        if (owner == NULL || nv < 3) continue;
        bool outward = owner->polygonDirections_[polygon->id_];
        double reach = 0.;
        for (auto v : polygon->vertices_) {
            double d[3]; for (int m = 0; m < 3; m++) d[m] = v->position_[m] - polygon->center_[m];
            reach = max(reach, sqrt(dot3(d, d)));
            for (int m = 0; m < 3; m++) { lo[m] = min(lo[m], v->position_[m]); hi[m] = max(hi[m], v->position_[m]); }
        }
        for (size_t i = 0; i < nv; i++) {
            Tri t;
            t.polygon = polygon;
            t.va = polygon->vertices_[i];
            t.vb = polygon->vertices_[(i + 1) % nv];
            double u[3], w[3];
            for (int m = 0; m < 3; m++) {
                t.c[m] = polygon->center_[m]; t.a[m] = t.va->position_[m]; t.b[m] = t.vb->position_[m];
                u[m] = t.a[m] - t.c[m]; w[m] = t.b[m] - t.c[m];
            }
            t.n[0] = u[1]*w[2] - u[2]*w[1]; t.n[1] = u[2]*w[0] - u[0]*w[2]; t.n[2] = u[0]*w[1] - u[1]*w[0];
            double ln = sqrt(dot3(t.n, t.n));
            if (ln < 1e-14) continue;  // degenerate sliver: covered by its neighbours
            for (int m = 0; m < 3; m++) t.n[m] = (outward ? 1. : -1.) * t.n[m] / ln;
            t.reach2 = (reach + margin) * (reach + margin);
            tris.push_back(t);
        }
    }
    if (tris.empty()) return 0;

    // CHANGED 2026-11-25 (speed): grids for the two searches in step 3 (not built with
    // 'contact_search brute'). The searches used to visit every surface triangle for every fiber
    // test point, about two thirds of the engine's time once the organoid has ~1,000 cells.
    // A triangle passes the reach test below only if its face centre is within searchR of x, so
    // the nearest-triangle search needs only the grid cells around x; the ray test needs only the
    // triangles whose shadow covers x (Grid2). Same triangles, same arithmetic: same forces.
    Grid3 g3;
    Grid2 g2[3];
    double searchR = 0.;
    if (!bruteSearch_) {
        double reachMax2 = 0.;
        for (const auto & t : tris) reachMax2 = max(reachMax2, t.reach2);
        searchR = sqrt(reachMax2) * (1. + 1e-12) + 1e-12;
        g3.h = max(0.25, 0.5 * searchR);
        for (int m = 0; m < 3; m++) { g3.lo[m] = lo[m]; g3.n[m] = (int)floor((hi[m] - lo[m]) / g3.h) + 1; }
        size_t ncell = (size_t)g3.n[0] * g3.n[1] * g3.n[2];
        g3.off.assign(ncell + 1, 0);
        vector<int> cellOf(tris.size());
        for (size_t i = 0; i < tris.size(); i++) {
            const double * c = tris[i].c;
            cellOf[i] = (g3.cellIndex(c, 2) * g3.n[1] + g3.cellIndex(c, 1)) * g3.n[0] + g3.cellIndex(c, 0);
            g3.off[cellOf[i] + 1]++;
        }
        for (size_t k = 1; k <= ncell; k++) g3.off[k] += g3.off[k - 1];
        g3.idx.resize(tris.size());
        vector<int> fill(g3.off.begin(), g3.off.end() - 1);
        for (size_t i = 0; i < tris.size(); i++) g3.idx[fill[cellOf[i]]++] = (int)i;
        for (int r = 0; r < 3; r++) g2[r] = buildRayGrid(tris, kRays[r]);
    }

    // 2. test points along every fiber segment near the organoid
    for (auto fiber : run_->fibers_) {
        for (size_t s = 0; s + 1 < fiber->nodes_.size(); s++) {
            Node * n0 = fiber->nodes_[s];
            Node * n1 = fiber->nodes_[s + 1];
            double d[3]; for (int m = 0; m < 3; m++) d[m] = n1->position_[m] - n0->position_[m];
            double L = sqrt(dot3(d, d));
            if (L > 10.) continue;  // segment wraps the periodic box, far from the organoid
            bool near = true;
            for (int m = 0; m < 3; m++) {
                double a = min(n0->position_[m], n1->position_[m]), b = max(n0->position_[m], n1->position_[m]);
                if (b < lo[m] - margin || a > hi[m] + margin) near = false;
            }
            if (!near) continue;
            int ns = max(1, (int)ceil(L / spacing_));
            double ds = L / ns;
            for (int k = 0; k < ns; k++) {
                double w = (k + 0.5) / ns;
                double x[3]; for (int m = 0; m < 3; m++) x[m] = n0->position_[m] + w * d[m];

                // 3. nearest surface point q
                double best = 1e30, bq[3], bw[3];
                const Tri * bt = NULL;
                if (bruteSearch_) {   // the original search: every triangle
                    for (const auto & t : tris) {
                        double r[3]; for (int m = 0; m < 3; m++) r[m] = x[m] - t.c[m];
                        if (dot3(r, r) > t.reach2) continue;
                        double q[3], wt[3];
                        closestOnTriangle(x, t.c, t.a, t.b, q, wt);
                        double dq[3]; for (int m = 0; m < 3; m++) dq[m] = x[m] - q[m];
                        double dist = sqrt(dot3(dq, dq));
                        if (dist < best) {
                            best = dist; bt = &t;
                            for (int m = 0; m < 3; m++) { bq[m] = q[m]; bw[m] = wt[m]; }
                        }
                    }
                } else {
                    // CHANGED 2026-11-25 (speed): only the grid cells within searchR of x. The cells are
                    // visited in another order than the triangle list, so on an exact tie the lowest
                    // triangle index wins, which is the triangle the original loop keeps ('<' above).
                    long bi = -1;
                    int c0[3], c1[3];
                    bool none = false;
                    for (int m = 0; m < 3; m++) {
                        int a = (int)floor((x[m] - searchR - g3.lo[m]) / g3.h);
                        int b = (int)floor((x[m] + searchR - g3.lo[m]) / g3.h);
                        if (b < 0 || a >= g3.n[m]) none = true;
                        c0[m] = max(a, 0); c1[m] = min(b, g3.n[m] - 1);
                    }
                    if (!none)
                    for (int iz = c0[2]; iz <= c1[2]; iz++)
                    for (int iy = c0[1]; iy <= c1[1]; iy++)
                    for (int ix = c0[0]; ix <= c1[0]; ix++) {
                        int cell = (iz * g3.n[1] + iy) * g3.n[0] + ix;
                        for (int kk = g3.off[cell]; kk < g3.off[cell + 1]; kk++) {
                            int i = g3.idx[kk];
                            const Tri & t = tris[i];
                            double r[3]; for (int m = 0; m < 3; m++) r[m] = x[m] - t.c[m];
                            if (dot3(r, r) > t.reach2) continue;
                            double q[3], wt[3];
                            closestOnTriangle(x, t.c, t.a, t.b, q, wt);
                            double dq[3]; for (int m = 0; m < 3; m++) dq[m] = x[m] - q[m];
                            double dist = sqrt(dot3(dq, dq));
                            if (dist < best || (dist == best && i < bi)) {
                                best = dist; bi = i;
                                for (int m = 0; m < 3; m++) { bq[m] = q[m]; bw[m] = wt[m]; }
                            }
                        }
                    }
                    if (bi >= 0) bt = &tris[bi];
                }
                if (bt == NULL) continue;

                if (best >= margin) continue;  // too far to matter on either side

                // inside or outside? q inside a triangle: that triangle's normal decides (exact).
                // q on an edge or a corner: ray crossings decide (majority of three fixed rays).
                double dx[3]; for (int m = 0; m < 3; m++) dx[m] = x[m] - bq[m];
                const double eps = 1e-9;
                bool interior = bw[0] > eps && bw[1] > eps && bw[2] > eps;
                bool outside;
                if (interior) {
                    outside = dot3(dx, bt->n) >= 0.;
                } else {
                    // CHANGED 2026-11-25 (speed): the ray directions are kRays (same numbers), and the
                    // crossings are counted over the triangles whose shadow covers x (countCrossings); a
                    // count does not depend on the order. Old:
                    //   static const double rays[3][3] = {{0.2673, 0.5345, 0.8018}, {-0.8018, 0.2673, 0.5345},
                    //                                     {0.5345, -0.8018, 0.2673}};
                    //   ... for (const auto & t : tris) crossings += rayCrosses(x, rays[r], t.c, t.a, t.b);
                    int insideVotes = 0;
                    for (int r = 0; r < 3; r++) {
                        int crossings = 0;
                        if (bruteSearch_) for (const auto & t : tris) crossings += rayCrosses(x, kRays[r], t.c, t.a, t.b);
                        else crossings = countCrossings(tris, g2[r], x, kRays[r]);
                        insideVotes += crossings % 2;
                    }
                    outside = insideVotes < 2;
                }

                // 4. push the test point out
                double overlap, dir[3];
                if (outside) {
                    if (best >= range_) continue;
                    overlap = range_ - best;
                    for (int m = 0; m < 3; m++) dir[m] = best > 1e-12 ? dx[m] / best : bt->n[m];
                } else {
                    if (best > maxDepth_) continue;
                    overlap = range_ + best;
                    for (int m = 0; m < 3; m++) dir[m] = best > 1e-12 ? -dx[m] / best : bt->n[m];
                }
                double stiffness = k_ * ds;
                double Fabs = stiffness * overlap;
                energy_ += 0.5 * stiffness * overlap * overlap;
                nContacts_++;

                // 5. force on the fiber nodes, equal and opposite force on the face
                double nCorners = (double)bt->polygon->vertices_.size();
                for (int m = 0; m < 3; m++) {
                    double F = Fabs * dir[m];
                    n0->contactForce_[m] += (1. - w) * F;
                    n1->contactForce_[m] += w * F;
                    double centreShare = -bw[0] * F / nCorners;
                    for (auto v : bt->polygon->vertices_) v->contactForce_[m] += centreShare;
                    bt->va->contactForce_[m] -= bw[1] * F;
                    bt->vb->contactForce_[m] -= bw[2] * F;
                }
            }
        }
    }
    return 0;
}
