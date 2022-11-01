#pragma once

#include <array>
#include <limits>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "glm/ext.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#define M_PI  3.14159265358979323846264338327950288

const double MAX_ENERGY = 1e50;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
using Vector_3f = K::Vector_3;
using Plane_3f = K::Plane_3;
using Point_3f = K::Point_3;

#define TO_CGAL(v) Point_3f({v[0], v[1], v[2]})

struct Vertex
{
    Vertex() = default;
    glm::vec3 pos;
    int point_marker;
};


struct Tetrahedron
{
    Tetrahedron() = default;
    std::array<unsigned int, 4> v;
    std::array<int, 4> n;
    std::array<int, 4> face_idx;
    int region_id;
    bool temporary = false;
    float winding_number;
};

class TetQuality {
public:
    double min_d_angle = 0;
    double max_d_angle = 0;
    //    double asp_ratio_2;

    double slim_energy = 0;
    double volume = 0;

    TetQuality() = default;
    TetQuality(double d_min, double d_max, double r)
        : min_d_angle(d_min), max_d_angle(d_max)
    { }

    /*bool isBetterThan(const TetQuality& tq, int energy_type, const State& state) {
        if (energy_type == state.ENERGY_AMIPS || energy_type == state.ENERGY_DIRICHLET) {
            return slim_energy < tq.slim_energy;
        }
        else if (energy_type == state.ENERGY_AD) {
            return min_d_angle > tq.min_d_angle && max_d_angle < tq.max_d_angle;
        }
        else
            return false;
    }*/

    /*bool isBetterOrEqualThan(const TetQuality& tq, int energy_type, const State& state) {
        if (energy_type == state.ENERGY_AMIPS || energy_type == state.ENERGY_DIRICHLET) {
            return slim_energy <= tq.slim_energy;
        }
        else if (energy_type == state.ENERGY_AD) {
            return min_d_angle >= tq.min_d_angle && max_d_angle <= tq.max_d_angle;
        }
        else
            return false;
    }*/
};

double comformalAMIPSEnergy_new(const double* T);
void calTetQuality_AMIPS(const std::vector<Vertex>& tet_vertices, const Tetrahedron tet, TetQuality& t_quality) {
    std::array<double, 12> T;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            T[i * 3 + j] = tet_vertices[tet.v[i]].pos[j];
        }
    }
    t_quality.slim_energy = comformalAMIPSEnergy_new(T.data());
    if (std::isinf(t_quality.slim_energy) || std::isnan(t_quality.slim_energy))
        t_quality.slim_energy = MAX_ENERGY;
}

void calTetQuality_AD(const std::vector<Vertex>& tet_vertices, const Tetrahedron tet, TetQuality& t_quality) {
    std::array<Vector_3f, 4> nv;
    std::array<double, 4> nv_length;
    std::array<double, 4> heights;

    for (int i = 0; i < 4; i++) {
        Plane_3f pln(TO_CGAL(tet_vertices[tet.v[(i + 1) % 4]].pos),
            TO_CGAL(tet_vertices[tet.v[(i + 2) % 4]].pos),
            TO_CGAL(tet_vertices[tet.v[(i + 3) % 4]].pos));
        if (pln.is_degenerate()) {
            t_quality.min_d_angle = 0;
            t_quality.max_d_angle = M_PI;
            return;
        }
        Point_3f tmp_p = pln.projection(TO_CGAL(tet_vertices[tet.v[i]].pos));
        if (tmp_p == TO_CGAL(tet_vertices[tet.v[i]].pos)) {
            t_quality.min_d_angle = 0;
            t_quality.max_d_angle = M_PI;
            return;
        }
        nv[i] = TO_CGAL(tet_vertices[tet.v[i]].pos) - tmp_p;
        heights[i] = CGAL::squared_distance(TO_CGAL(tet_vertices[tet.v[i]].pos), tmp_p);


                //re-scale
        std::array<double, 3> tmp_nv = { {CGAL::abs(nv[i][0]), CGAL::abs(nv[i][1]), CGAL::abs(nv[i][2])} };
        auto tmp = std::max_element(tmp_nv.begin(), tmp_nv.end());
        if (*tmp == 0 || heights[i] == 0) {
            t_quality.min_d_angle = 0;
            t_quality.max_d_angle = M_PI;
            //            t_quality.asp_ratio_2 = state.MAX_ENERGY;
            return;
        }
        else if (*tmp < 1e-5) {
            nv[i] = Vector_3f(nv[i][0] / *tmp, nv[i][1] / *tmp, nv[i][2] / *tmp);
            nv_length[i] = sqrt(heights[i] / ((*tmp) * (*tmp)));
        }
        else {
            nv_length[i] = sqrt(heights[i]);
        }
    }

    std::vector<std::array<int, 2>> opp_edges;
    for (int i = 0; i < 3; i++) {
        opp_edges.push_back(std::array<int, 2>({ {0, i + 1} }));
        opp_edges.push_back(std::array<int, 2>({ {i + 1, (i + 1) % 3 + 1} }));
    }

    ////compute dihedral angles
    std::array<double, 6> dihedral_angles;
    for (int i = 0; i < (int)opp_edges.size(); i++) {
        double dihedral_angle = -nv[opp_edges[i][0]] * nv[opp_edges[i][1]] /
            (nv_length[opp_edges[i][0]] * nv_length[opp_edges[i][1]]);
        if (dihedral_angle > 1)
            dihedral_angles[i] = 0;
        else if (dihedral_angle < -1)
            dihedral_angles[i] = M_PI;
        else
            dihedral_angles[i] = std::acos(dihedral_angle);
    }
    //    std::sort(dihedral_angles.begin(), dihedral_angles.end());
    auto it = std::minmax_element(dihedral_angles.begin(), dihedral_angles.end());
    t_quality.min_d_angle = *(it.first);
    t_quality.max_d_angle = *(it.second);

    //    std::sort(heights.begin(), heights.end());
    //    auto h = std::min_element(heights.begin(), heights.end());
    //    t_quality.asp_ratio_2 = max_e_l / *h;
}

double comformalAMIPSEnergy_new(const double* T) {
    double helper_0[12];
    helper_0[0] = T[0];
    helper_0[1] = T[1];
    helper_0[2] = T[2];
    helper_0[3] = T[3];
    helper_0[4] = T[4];
    helper_0[5] = T[5];
    helper_0[6] = T[6];
    helper_0[7] = T[7];
    helper_0[8] = T[8];
    helper_0[9] = T[9];
    helper_0[10] = T[10];
    helper_0[11] = T[11];
    double helper_1 = helper_0[2];
    double helper_2 = helper_0[11];
    double helper_3 = helper_0[0];
    double helper_4 = helper_0[3];
    double helper_5 = helper_0[9];
    double helper_6 = 0.577350269189626 * helper_3 - 1.15470053837925 * helper_4 + 0.577350269189626 * helper_5;
    double helper_7 = helper_0[1];
    double helper_8 = helper_0[4];
    double helper_9 = helper_0[7];
    double helper_10 = helper_0[10];
    double helper_11 = 0.408248290463863 * helper_10 + 0.408248290463863 * helper_7 + 0.408248290463863 * helper_8 -
        1.22474487139159 * helper_9;
    double helper_12 = 0.577350269189626 * helper_10 + 0.577350269189626 * helper_7 - 1.15470053837925 * helper_8;
    double helper_13 = helper_0[6];
    double helper_14 = -1.22474487139159 * helper_13 + 0.408248290463863 * helper_3 + 0.408248290463863 * helper_4 +
        0.408248290463863 * helper_5;
    double helper_15 = helper_0[5];
    double helper_16 = helper_0[8];
    double helper_17 = 0.408248290463863 * helper_1 + 0.408248290463863 * helper_15 - 1.22474487139159 * helper_16 +
        0.408248290463863 * helper_2;
    double helper_18 = 0.577350269189626 * helper_1 - 1.15470053837925 * helper_15 + 0.577350269189626 * helper_2;
    double helper_19 = 0.5 * helper_13 + 0.5 * helper_4;
    double helper_20 = 0.5 * helper_8 + 0.5 * helper_9;
    double helper_21 = 0.5 * helper_15 + 0.5 * helper_16;
    return -(helper_1 * (-1.5 * helper_1 + 0.5 * helper_2 + helper_21) +
        helper_10 * (-1.5 * helper_10 + helper_20 + 0.5 * helper_7) +
        helper_13 * (-1.5 * helper_13 + 0.5 * helper_3 + 0.5 * helper_4 + 0.5 * helper_5) +
        helper_15 * (0.5 * helper_1 - 1.5 * helper_15 + 0.5 * helper_16 + 0.5 * helper_2) +
        helper_16 * (0.5 * helper_1 + 0.5 * helper_15 - 1.5 * helper_16 + 0.5 * helper_2) +
        helper_2 * (0.5 * helper_1 - 1.5 * helper_2 + helper_21) +
        helper_3 * (helper_19 - 1.5 * helper_3 + 0.5 * helper_5) +
        helper_4 * (0.5 * helper_13 + 0.5 * helper_3 - 1.5 * helper_4 + 0.5 * helper_5) +
        helper_5 * (helper_19 + 0.5 * helper_3 - 1.5 * helper_5) +
        helper_7 * (0.5 * helper_10 + helper_20 - 1.5 * helper_7) +
        helper_8 * (0.5 * helper_10 + 0.5 * helper_7 - 1.5 * helper_8 + 0.5 * helper_9) +
        helper_9 * (0.5 * helper_10 + 0.5 * helper_7 + 0.5 * helper_8 - 1.5 * helper_9)) *
        pow(pow((helper_1 - helper_2) * (helper_11 * helper_6 - helper_12 * helper_14) -
            (-helper_10 + helper_7) * (-helper_14 * helper_18 + helper_17 * helper_6) +
            (helper_3 - helper_5) * (-helper_11 * helper_18 + helper_12 * helper_17), 2), -0.333333333333333);
}