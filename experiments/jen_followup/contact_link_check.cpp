// Checks the actual C++ contact and linker implementations on a single flat face.
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../../Run/Run.h"

namespace {
void expect(double actual, double expected, const char* label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12) {
        std::cerr << label << ": expected " << expected << ", got " << actual << '\n';
        throw std::runtime_error(label);
    }
}
}

int main() {
    Run run;
    Box box(&run);
    run.box_ = &box;
    for (int m = 0; m < 3; ++m) {
        box.size_[m] = 8;
        box.boundaryCondition_[m] = true;
    }
    Cell cell(&run, 1);
    cell.type_ = 1;
    Vertex a(&run, 1), b(&run, 2), c(&run, 3), d(&run, 4);
    Edge ab(&run, 1), bc(&run, 2), cd(&run, 3), da(&run, 4);
    Polygon face(&run, 1);
    run.vertices_ = {&a, &b, &c, &d};
    double xy[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    face.type_ = 2;
    face.cells_ = {&cell};
    face.vertices_ = run.vertices_;
    face.edges_ = {&ab, &bc, &cd, &da};
    for (int i = 0; i < 4; ++i) {
        Vertex* v = face.vertices_[i];
        v->position_[0] = xy[i][0];
        v->position_[1] = xy[i][1];
        v->position_[2] = 0;
        face.edges_[i]->vertices_ = {v, face.vertices_[(i + 1) % 4]};
        face.edges_[i]->length_ = 2;
    }
    cell.polygonDirections_[face.id_] = true;
    run.polygons_ = {&face};
    Fiber fiber(&run, 1);
    Node first(&run, 1), second(&run, 2);
    run.fibers_ = {&fiber};
    run.nodes_ = {&first, &second};
    fiber.nodes_ = run.nodes_;
    first.position_[0] = -0.2;
    second.position_[0] = 0.2;
    first.position_[1] = second.position_[1] = -0.5;

    CellFiberContact contact(&run);
    contact.k_ = 50;
    auto check_contact = [&](double z, double force) {
        first.position_[2] = second.position_[2] = z;
        contact.updateForces();
        double surface = 0;
        for (auto* vertex : face.vertices_) {
            surface += vertex->contactForce_[2];
            expect(vertex->contactForce_[0], 0, "contact tangential force");
            expect(vertex->contactForce_[1], 0, "contact tangential force");
        }
        for (auto* node : run.nodes_) {
            expect(node->contactForce_[2], force / 2, "node normal force");
            expect(node->contactForce_[0], 0, "fiber tangential force");
            expect(node->contactForce_[1], 0, "fiber tangential force");
        }
        expect(surface, -force, "equal and opposite face force");
        expect(contact.nContacts_, force > 0, "sample count");
    };
    check_contact(0.25, 0);
    check_contact(0.10, 2);
    check_contact(0, 4);
    check_contact(-0.10, 6);
    check_contact(-0.99, 23.8);
    check_contact(-1.01, 0);

    // With two samples on a tilted segment, the lower endpoint moves outward faster.
    first.position_[0] = -0.4;
    second.position_[0] = 0.4;
    first.position_[2] = 0.05;
    second.position_[2] = 0.15;
    contact.updateForces();
    double ds = std::hypot(0.8, 0.1) / 2;
    double lowForce = contact.k_ * ds * (0.2 - 0.075);
    double highForce = contact.k_ * ds * (0.2 - 0.125);
    expect(contact.nContacts_, 2, "tilted-segment samples");
    expect(first.contactForce_[2], 0.75 * lowForce + 0.25 * highForce, "lower endpoint force");
    expect(second.contactForce_[2], 0.25 * lowForce + 0.75 * highForce, "higher endpoint force");
    std::cout << "Tilted fiber endpoint forces (near, far): (" << first.contactForce_[2]
              << ", " << second.contactForce_[2] << ").\n";

    FiberLink spring(&run);
    spring.l0_ = 1.5;
    spring.ks_ = 10;
    Link link(&run, &face, &first);
    run.links_ = {&link};
    first.position_[0] = first.position_[1] = 0;
    for (double z : {1., 1.5, 2.}) {
        first.position_[2] = z;
        spring.updateForces();
        spring.updateEnergy();
        const double force = -10 * (z - 1.5);
        expect(first.linkForce_[2], force, "Hookean linker force");
        double surface = 0;
        for (auto* vertex : face.vertices_) surface += vertex->linkForce_[2];
        expect(surface, -force, "equal and opposite linker force");
        expect(spring.stretchEnergy_, 5 * std::pow(z - 1.5, 2), "link energy");
    }
    spring.ks_ = 0;
    first.position_[2] = 0;
    spring.updateForces();
    spring.updateEnergy();
    expect(run.links_.size(), 1, "attachment retained");
    expect(spring.stretchEnergy_, 0, "zero-stiffness energy");
    for (int i = 0; i < 3; ++i) {
        expect(first.linkForce_[i], 0, "zero-stiffness node force");
        for (auto* vertex : face.vertices_) expect(vertex->linkForce_[i], 0, "zero-stiffness face force");
    }
    std::cout << "Flat-face contact and fixed-rest-length / zero-stiffness linker checks passed.\n";

    // An irregular face exposes the existing equal sharing of the center reaction.
    c.position_[0] = 2;
    for (auto* edge : face.edges_) edge->update();
    face.updateCenter();
    double barycentric_center_weight = 0.5;
    for (int m = 0; m < 2; ++m) {
        double q = barycentric_center_weight * face.center_[m] + 0.2 * a.position_[m] + 0.3 * b.position_[m];
        first.position_[m] = q;
        second.position_[m] = q;
    }
    first.position_[0] -= 0.2;
    second.position_[0] += 0.2;
    check_contact(0.10, 2);
    double torque[2] = {0, 0};
    auto accumulate_torque = [&](const double* position, const double* force) {
        torque[0] += position[1] * force[2] - position[2] * force[1];
        torque[1] += position[2] * force[0] - position[0] * force[2];
    };
    double corner_mean[2] = {0, 0};
    for (auto* vertex : face.vertices_) {
        accumulate_torque(vertex->position_, vertex->contactForce_);
        for (int m = 0; m < 2; ++m) corner_mean[m] += vertex->position_[m] / 4;
    }
    for (auto* node : run.nodes_) accumulate_torque(node->position_, node->contactForce_);
    expect(torque[0], face.center_[1] - corner_mean[1], "contact torque from center reaction in x");
    expect(torque[1], corner_mean[0] - face.center_[0], "contact torque from center reaction in y");
    if (std::hypot(torque[0], torque[1]) < 1e-3) throw std::runtime_error("irregular-face diagnostic did not expose torque");
    std::cout << "Existing irregular-face contact torque about (x, y): ("
              << torque[0] << ", " << torque[1] << ").\n";
}
