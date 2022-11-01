// Hybrid.cpp : Bu dosya 'main' işlevi içeriyor. Program yürütme orada başlayıp biter.
//

#include <opencv2/opencv.hpp>

#include "tetgen.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <limits>
#define M_PI  3.14159265358979323846264338327950288

#include <time.h>
#include "tiny_obj_loader.h"
#include <string>
#include <windows.h>
#include <fstream>
#include "glm/ext.hpp"
#include <algorithm>
#include <chrono> 
#include <unordered_set>
#include <fstream>
#include <set>
#include <CGAL/intersections.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include <CGAL/Polygon_mesh_slicer.h>
#include <omp.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>

#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/transform.h>
#include <CGAL/boost/graph/IO/OBJ.h>
#include <CGAL/boost/graph/IO/STL.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>


#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <Eigen/Dense>
//#include <igl/fast_winding_number.h>
//#include <igl/winding_number.h>
//#include <igl/read_triangle_mesh.h>
//#include <igl/write_triangle_mesh.h>
//#include <igl/copyleft/tetgen/tetrahedralize.h>
//#include <igl/copyleft/cgal/remesh_self_intersections.h>
//#include <igl/copyleft/tetgen/cdt.h>
#include <CGAL/Bbox_3.h>
#include "tetwild_helper.h"


typedef long long int llint;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Exact_predicates_exact_constructions_kernel EK;
typedef CGAL::Surface_mesh<K::Point_3>             SMesh;
typedef CGAL::Surface_mesh<EK::Point_3>             ESMesh;
typedef K::Point_3 Point_3;
namespace PMP = CGAL::Polygon_mesh_processing;
namespace params = PMP::parameters;
typedef boost::graph_traits<SMesh>::halfedge_descriptor   halfedge_descriptor;
typedef boost::graph_traits<SMesh>::edge_descriptor   edge_descriptor;
typedef boost::graph_traits<SMesh>::face_descriptor   face_descriptor;
typedef boost::graph_traits<SMesh>::vertex_descriptor   vertex_descriptor;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using cv::Mat;
using cv::PCA;
using cv::Point3d;
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::unordered_map;
using glm::vec3;
using glm::dvec3;
using std::pair;
using map_int = unordered_map<int, int>;
using map_int_pair = unordered_map<int, pair<int, int>>;
namespace std {
	template <> struct hash<std::pair<int, int>> {
		inline size_t operator()(const std::pair<int, int>& v) const {
			std::hash<int> int_hasher;
			return int_hasher(v.first) ^ int_hasher(v.second);
		}
	};
}

struct FaceInfo2
{
	FaceInfo2() = default;
	int nesting_level;
	bool in_domain() const {
		return nesting_level % 2 == 1;
	}
};

// 1.79705 -88.757 152.689
#define IS_POS_PLANE(p, c) p.oriented_side(c) == CGAL::ON_POSITIVE_SIDE
#define IS_NEG_PLANE(p, c) p.oriented_side(c) == CGAL::ON_NEGATIVE_SIDE
#define INDEX_TO_VD(mesh, i) *(mesh.vertices_begin() + i) 
#define CPU_MODE true
#define EXISTS(x, i) x.find(i) != x.end()
#define MIN(v) *std::min_element(v.begin(), v.end())
#define MAX(v) *std::max_element(v.begin(), v.end())
#define ZERO_PAIR std::make_pair(0, 0)
#define PAIR_TO_VECTOR(p) {p.first, p.second}
#define SORT(v) std::sort(v.begin(), v.end())
#define PRINT(v) cout << v[0] << " " << v[1] << " " << v[2] << endl;
constexpr int TEST_NUMBER = 10000000;
struct TetMeshIn
{
	// Input Data
	std::vector<glm::vec3> points;
	std::vector<int> facets;
	std::vector<int> facet_indices;
	std::vector<int> facet_markerlist;
	std::vector<int> point_markerlist;

	// Visibility of the face
	std::vector<bool> is_face_visible;

	// Meshing arguments
	bool preserve_triangles;
	float quality;
	bool write_mesh;

	// num_points: number of vertices
	// num_facets: number of facets
	// facets_size: size of facets array (TODO: Rename?)
	TetMeshIn(int num_points, int num_facets, int facets_size)
		: points(num_points),
		facets(facets_size),
		facet_indices(num_facets),
		facet_markerlist(num_facets),
		point_markerlist(num_points, 0),
		is_face_visible(facets_size, true),
		preserve_triangles(true),
		quality(5),
		write_mesh(true)
	{
	}

	// Utility methods
	int num_points() const
	{
		return points.size();
	}

	int num_facets() const
	{
		return facet_indices.size();
	}

	int facets_size() const
	{
		return facets.size();
	}
};

struct Face
{
	Face() = default;
	int v[3];
	int face_marker;
};


// Generic Tetmesh output
struct TetMeshOut
{
	// Output Data
	vector<Vertex> points;
	vector<Face> faces;
	vector<Tetrahedron> tets;
	//tetgenio* tetgen_out;
};

class MetGen {
private:
	double density_control;
	double threshold;
	bool remove_extra;

public:
	float total_io;

	MetGen() = default;
	MetGen(double density_control, double threshold, bool remove_extra) {
		this->density_control = density_control;
		this->threshold = threshold;
		this->remove_extra = remove_extra;
	}

	void convert_input(TetMeshIn& src, tetgenio& dst)
	{
		// Initialize dst
		dst.numberofpoints = src.num_points();
		dst.pointlist = new REAL[dst.numberofpoints * 3];
		dst.numberoffacets = src.num_facets();
		dst.facetlist = new tetgenio::facet[dst.numberoffacets];
		dst.facetmarkerlist = new int[dst.numberoffacets];
		dst.pointmarkerlist = new int[dst.numberofpoints];
		// Fill dst
		for (size_t i = 0; i < dst.numberofpoints; i++)
		{
			glm::vec3& p = src.points[i];
			dst.pointlist[(3 * i) + 0] = p.x;
			dst.pointlist[(3 * i) + 1] = p.y;
			dst.pointlist[(3 * i) + 2] = p.z;
		}

		std::memcpy(dst.facetmarkerlist, &src.facet_markerlist[0], sizeof(int) * src.num_facets());
		std::memcpy(dst.pointmarkerlist, &src.point_markerlist[0], sizeof(int) * dst.numberofpoints);

		for (size_t i = 0; i < src.num_facets(); i++)
		{
			const int current_facet_index = src.facet_indices[i];
			const int next_facet_index = (i == src.num_facets() - 1) ? src.facets_size() : src.facet_indices[i + 1];
			const int num_vertices = next_facet_index - current_facet_index;
			assert(next_facet_index > current_facet_index);

			tetgenio::facet& f = dst.facetlist[i];
			f.numberofpolygons = 1;
			f.polygonlist = new tetgenio::polygon[f.numberofpolygons];

			f.numberofholes = 0;
			f.holelist = nullptr;

			tetgenio::polygon* p = &f.polygonlist[0];
			p->numberofvertices = num_vertices;
			p->vertexlist = new int[p->numberofvertices];

			for (int i = next_facet_index - 1, j = 0; i >= current_facet_index; i--, j++)
			{
				p->vertexlist[j] = src.facets[i];
			}
		}
	}

	void convert_output(tetgenio& src, TetMeshIn& in, TetMeshOut& dst)
	{
		dst.points.resize(src.numberofpoints);
		dst.tets.resize(src.numberoftetrahedra);
		dst.faces.resize(src.numberoftrifaces);
		//dst.tetgen_out = &src;
		for (int i = 0; i < src.numberofpoints; i++)
		{
			dst.points[i].pos = glm::make_vec3(&src.pointlist[3 * i]);
			dst.points[i].point_marker = src.pointmarkerlist[i];
		}
		for (int i = 0; i < src.numberoftetrahedra; i++)
		{
			//dst.tets[i].region_id = (int)src.tetrahedronattributelist[i];

			for (int j = 0; j < 4; j++)
			{

				dst.tets[i].v[j] = src.tetrahedronlist[4 * i + j];
				dst.tets[i].n[j] = src.neighborlist[4 * i + j];
				dst.tets[i].face_idx[j] = src.tet2facelist[4 * i + j];

				// Set face invisible if is_face_visible flag is false
				//if (dst.tets[i].face_idx[j] > 0)
				//{
				//	++dst.constrained_face_count;
				//	if (!in.is_face_visible[dst.tets[i].face_idx[j] - 1])
				//		dst.tets[i].face_idx[j] = -1;
				//}

				//if (dst.tets[i].n[j] == -1)
				//	dst.air_region_id = dst.tets[i].region_id;
			}
		}

		for (int i = 0; i < src.numberoftrifaces; i++) {
			dst.faces[i].face_marker = src.trifacemarkerlist[i];
			for (int j = 0; j < 3; j++) {
				dst.faces[i].v[j] = src.trifacelist[3 * i + j];
			}
		}

		delete &src;

	}

	TetMeshIn createTetMeshInput(const vector<vec3>& vertices, const vector<int>& triangles,
		const map_int_pair& new_vertices, map_int vd_to_index, int index,
		map_int* boundary_map = nullptr, map_int* inverse_used_vertices = nullptr) {

		TetMeshIn in_data((int)vertices.size(), (int)triangles.size() / 3, triangles.size());

		// Fill TetMeshIn struct
		in_data.preserve_triangles = true;
		in_data.quality = 10.0;

		in_data.points = vertices;
		in_data.facets = triangles;
		printf("%d, %d\n", in_data.facets.size(), triangles.size());
		int cnt = 0;
		if (index == 0) {
			map_int inverse_boundary_map = *boundary_map;

			inverse_boundary_map = invert_map(*boundary_map);
			printf("Count: %d %d\n", boundary_map->size(), inverse_boundary_map.size());

		}
		for (int i = 0; i < in_data.num_facets(); i++) {
			in_data.facet_indices[i] = 3 * i;
			//if (boundary_map == nullptr) {
				//in_data.facet_markerlist[i] = i + 1;
			//}
			//else if (inverse_boundary_map.find(i) != inverse_boundary_map.end()) {
			in_data.facet_markerlist[i] = i + 1;
			cnt++;
			/*for (int j = 0; j < 3; j++) {
				in_data.point_markerlist[in_data.facets[3 * i + j]] = 100;
			}*/
			//}
		}
		map_int index_to_vd = invert_map(vd_to_index);
		for (int i = 0; i < in_data.points.size(); i++) {
			if (!(EXISTS(index_to_vd, i)) && vd_to_index.size() > 0) {
				printf("Assertion error\n");
			}
			if (inverse_used_vertices != nullptr) {
				in_data.point_markerlist[i] = inverse_used_vertices->at(i) + 1;
			}
			else {
				in_data.point_markerlist[i] = index_to_vd[i] + TEST_NUMBER;
			}
		}
		printf("Count: %d\n", cnt);

		for (auto const& it : new_vertices) {
			if (EXISTS(vd_to_index, it.first)) {
				in_data.point_markerlist[vd_to_index[it.first]] = it.first + 1;
			}
		}
		return in_data;
	}

	tetgenio* tetgen_tetrahedralize(TetMeshIn& in, TetMeshOut& out, bool isConstrained, std::string output_name, bool& success)
	{
		tetgenio in_data;
		tetgenio* out_data = new tetgenio();
		convert_input(in, in_data);

		success = true;

		//Run tetgen
		try
		{
			constexpr int BUFFER_SIZE = 128;
			char str[BUFFER_SIZE];
			if (in.preserve_triangles)
			{
#ifdef HAVE_SNPRINTF
				snprintf(str, BUFFER_SIZE, "q%.2fYnAfQ", in.quality);
#else
				sprintf_s(str, BUFFER_SIZE, "q%.2fYnAfQ", in.quality);
#endif
			}
			else
			{
#ifdef HAVE_SNPRINTF
				snprintf(str, BUFFER_SIZE, "q%.2fnnAfQ", in.quality);
#else
				sprintf_s(str, BUFFER_SIZE, "q%.2fnnAfQ", in.quality);
#endif
			}
			tetgenbehavior b;
			if (isConstrained) {
				b.parse_commandline(str);
			}
			b.neighout = 2;//2
			b.meditview = 0;//(in.write_mesh) ? 1 : 0;
			b.regionattrib = 0;
			b.mindihedral = 20;
			

			//b.diagnose = 0;
			//b.facesout = 1;

			b.verbose = 1;
			b.quiet = 0;
			strcpy(b.outfilename, output_name.c_str());
			tetrahedralize(&b, &in_data, out_data);
		}
		catch (int _)
		{
			printf("Error: %d\n", _);
			success = false;
		}

		if (!success) {
			std::cout << "Error in creating " << output_name << std::endl;
			throw "Error!";
		}

		if (in.write_mesh) {
			//printf("Writing time: $%g$ ms\n", out_data->output_time * 1000);
		}
		if (isConstrained && success) {
			convert_output(*out_data, in, out);
		}

		return out_data;
	}


	TetMeshOut* execute_tetgen(TetMeshIn& in, std::string output_name, bool isConstrained) {
		printf("Executing TetGen with #V: %d, #F: %d\n", in.points.size(), in.facets.size() / 3);
		TetMeshOut* _out = new TetMeshOut();
		bool success;
		tetgenio* out = tetgen_tetrahedralize(in, *_out, isConstrained, output_name, success);
		if (!success) {
			return nullptr;
		}
		return _out;
	}

	unordered_map<int, int> invert_map(unordered_map<int, int>& boundary_map)
	{
		map_int inverted_boundary_map;
		for (auto const& pair : boundary_map) {
			inverted_boundary_map.insert({ pair.second, pair.first });
		}

		return inverted_boundary_map;
	}


	void merge(vector<SMesh>& meshes, const vector<TetMeshOut*>& outputs, vector<map_int>& boundary_maps,
				std::chrono::milliseconds& writing_time_spent, std::chrono::milliseconds& winding_time_spent,
				map_int_pair& new_vertices, vector<map_int>& vd_to_index_array) {
		printf("Merge begins\n");
		auto winding_begin = high_resolution_clock::now();

		TetMeshOut* first = outputs[0];

		int total_number_of_vertices = 0, total_number_of_tets = 0, total_number_of_faces = 0;
		for (int i = 0; i < outputs.size(); i++) {
			auto* output = outputs[i];
			//if (output->tetgen_out == nullptr) {
			//	printf("output is nullptr %d\n", i);
			//}
			total_number_of_vertices += output->points.size();
			total_number_of_tets += output->tets.size();
			total_number_of_faces += output->faces.size();
		}
		printf("Total -- #V: %d #F: %d #T: %d\n", total_number_of_vertices, total_number_of_faces, total_number_of_tets);

		vector<int> prefix_sum_vertex_count(outputs.size()), prefix_sum_tet_count(outputs.size()), prefix_sum_face_count(outputs.size());
		for (int i = 0; i < prefix_sum_vertex_count.size(); i++) {
			if (i == 0) {
				prefix_sum_vertex_count[0] = first->points.size();
				prefix_sum_tet_count[0] = first->tets.size();
				prefix_sum_face_count[0] = first->faces.size();
			}
			else {
				prefix_sum_vertex_count[i] = prefix_sum_vertex_count[i - 1] + outputs[i]->points.size();
				prefix_sum_tet_count[i] = prefix_sum_tet_count[i - 1] + outputs[i]->tets.size();
				prefix_sum_face_count[i] = prefix_sum_face_count[i - 1] + outputs[i]->faces.size();
			}
		}

		if (outputs.size() > 1) {
			first->points.resize(total_number_of_vertices);
			first->faces.resize(total_number_of_faces);
			first->tets.resize(total_number_of_tets);
		}
		if (prefix_sum_vertex_count[prefix_sum_vertex_count.size() - 1] != total_number_of_vertices) {
			throw "Assertion error";
		}

		if (prefix_sum_tet_count[prefix_sum_tet_count.size() - 1] != total_number_of_tets) {
			throw "Assertion error";
		}

		if (prefix_sum_face_count[prefix_sum_face_count.size() - 1] != total_number_of_faces) {
			throw "Assertion error";
		}
		vector<map_int> face_marker_to_index_array, vertex_marker_to_index_array;
		map_int vertex_marker_to_index;
		int g = 0;
		vector<int> markers;



		for (int k = 0; k < outputs.size() - 1; k++) {

			// Process left pairs
			map_int face_marker_to_index;
			for (int i = 0; i < outputs[k]->faces.size(); i++) {
				int marker = outputs[k]->faces[i].face_marker - 1;
				if (marker >= 0) {
					face_marker_to_index[marker] = i;
				}
			}
			face_marker_to_index_array.push_back(face_marker_to_index);


			// Process right pairs
			for (int i = 0; i < outputs[k + 1]->points.size(); i++) {
				first->points[i + prefix_sum_vertex_count[k]] = outputs[k + 1]->points[i];
			}
			outputs[k + 1]->points.clear();
			// Process right pairs
			for (int i = 0; i < outputs[k + 1]->faces.size(); i++) {
				first->faces[i + prefix_sum_face_count[k]] = outputs[k + 1]->faces[i];
			}

		}

		map_int vertex_relation;
		for (int o_index = 0; o_index < outputs.size() - 1; o_index++) {

			TetMeshOut *left = outputs[o_index], *right = outputs[o_index + 1];
			map_int& boundary_map = boundary_maps[o_index];
			map_int& face_marker_to_index = face_marker_to_index_array[o_index];
			int match = 0;


			for (int i = 0; i < right->tets.size(); i++) {
				bool matched = false;
				vector<glm::vec3> temp;
				for (int j = 0; j < 4; j++) {
					first->tets[i + prefix_sum_tet_count[o_index]].v[j] = right->tets[i].v[j] + prefix_sum_vertex_count[o_index];
					first->tets[i + prefix_sum_tet_count[o_index]].face_idx[j] = right->tets[i].face_idx[j] + prefix_sum_face_count[o_index];
					first->tets[i + prefix_sum_tet_count[o_index]].n[j] = ((right->tets[i].n[j] == -1) ? -1 : right->tets[i].n[j] + prefix_sum_tet_count[o_index]);
					first->tets[i + prefix_sum_tet_count[o_index]].temporary = right->tets[i].temporary;

					int face_index = right->tets[i].face_idx[j];
					int face_marker_index = right->faces[face_index].face_marker - 1;

					if (boundary_map.find(face_marker_index) != boundary_map.end()) {
						first->faces[face_index + prefix_sum_face_count[o_index]].face_marker *= -1;
						int other_face_marker_index = boundary_map[face_marker_index];

						if (matched) {
							printf("A tet is matched with more than one faces\n");
						}
						match++;

						

						if (!(EXISTS(face_marker_to_index, other_face_marker_index))) {
							throw "Assumption violated";
						}
						int other_face_index = face_marker_to_index[other_face_marker_index];
						first->faces[other_face_index + ((o_index == 0) ? 0 : prefix_sum_face_count[o_index - 1])].face_marker *= -1;
						for (int a = 0; a < 3; a++) {
							int kk = right->faces[face_index].v[a] + prefix_sum_vertex_count[o_index];
							temp.push_back(first->points[kk].pos);
						}

						for (int b = 0; b < 3; b++) {
							int vv = left->faces[other_face_index].v[b] + (o_index > 0 ? prefix_sum_vertex_count[o_index - 1] : 0);
							temp.push_back(first->points[vv].pos);
						}

						for (int a = 0; a < 3; a++) {
							bool found = false;
							for (int b = 0; b < 3; b++) {
								int kk = right->faces[face_index].v[a] + prefix_sum_vertex_count[o_index];
								int vv = left->faces[other_face_index].v[b] + (o_index > 0 ? prefix_sum_vertex_count[o_index - 1] : 0);
								if (first->points[kk].pos == first->points[vv].pos) {
									vertex_relation.insert({ kk, vv });
									found = true;
								}
								/*if (matched) {
									PRINT(first->points[kk].pos);
									PRINT(first->points[vv].pos);
								}*/

							}
							if (!found) {
								throw "Error";
							}
						}
						matched = true;
					}
				}
			}

			if (boundary_maps[o_index].size() != match) {
				printf("%d: %d != %d\n", o_index, boundary_maps[o_index].size(), match);
				throw "Assertion error";
			}
		}
		TetMeshOut* out = first;


		for (int i = 0; i < out->points.size() && outputs.size() > 1; i++) {
			if (!(EXISTS(vertex_relation, i))) {
				vertex_relation.insert({ i, i });
			}
			int index = vertex_relation[i];
			int marker = out->points[index].point_marker;

			if (marker >= 1 && marker < TEST_NUMBER) {
				if (EXISTS(vertex_marker_to_index, marker - 1)) {
					g++;

				}
				else {
					vertex_marker_to_index[marker - 1] = index;
				}
			}
			else if (marker >= TEST_NUMBER) {
				if (EXISTS(vertex_marker_to_index, marker - TEST_NUMBER)) {
					g++;

				}
				else {
					vertex_marker_to_index[marker - TEST_NUMBER] = index;
				}
			}
		}


		vertex_marker_to_index_array.push_back(vertex_marker_to_index);


		output_tetmesh(out, vertex_relation, new_vertices, vertex_marker_to_index_array, vd_to_index_array, writing_time_spent);
		cal_quality_metrics(out);
	}
	void cal_quality_metrics(TetMeshOut* out) {

		vector<TetQuality> tet_qualities(out->tets.size());
		for (int i = 0; i < out->tets.size(); i++) {
			if (out->tets[i].temporary && remove_extra) {
				continue;
			}
			calTetQuality_AMIPS(out->points, out->tets[i], tet_qualities[i]);
			calTetQuality_AD(out->points, out->tets[i], tet_qualities[i]);
		}
		double min = 10, max = 0;
		double min_avg = 0, max_avg = 0;
		// double max_asp_ratio = 0, avg_asp_ratio = 0;
		double max_slim_energy = 0, avg_slim_energy = 0;
		std::array<double, 6> cmp_cnt = { {0, 0, 0, 0, 0, 0} };
		std::array<double, 6> cmp_d_angles = { {6 / 180.0 * M_PI, 12 / 180.0 * M_PI, 18 / 180.0 * M_PI,
											   162 / 180.0 * M_PI, 168 / 180.0 * M_PI, 174 / 180.0 * M_PI} };
		int cnt = 0;
		for (int i = 0; i < tet_qualities.size(); i++) {
			if (out->tets[i].temporary && remove_extra) {
				continue;
			}
			cnt++;
			if (tet_qualities[i].min_d_angle < min)
				min = tet_qualities[i].min_d_angle;
			if (tet_qualities[i].max_d_angle > max)
				max = tet_qualities[i].max_d_angle;
			// if (tet_qualities[i].asp_ratio_2 > max_asp_ratio)
				// max_asp_ratio = tet_qualities[i].asp_ratio_2;
			if (tet_qualities[i].slim_energy > max_slim_energy)
				max_slim_energy = tet_qualities[i].slim_energy;
			min_avg += tet_qualities[i].min_d_angle;
			max_avg += tet_qualities[i].max_d_angle;
			// avg_asp_ratio += tet_qualities[i].asp_ratio_2;
			avg_slim_energy += tet_qualities[i].slim_energy;

			for (int j = 0; j < 3; j++) {
				if (tet_qualities[i].min_d_angle < cmp_d_angles[j])
					cmp_cnt[j]++;
			}
			for (int j = 0; j < 3; j++) {
				if (tet_qualities[i].max_d_angle > cmp_d_angles[j + 3])
					cmp_cnt[j + 3]++;
			}
		}
		printf("tet_count = $%d$\n", cnt);
		printf("min_d_angle = $%f$\nmax_d_angle = $%f$\nmax_slim_energy = $%f$\n", min, max, max_slim_energy);
		printf("avg_min_d_angle = $%f$\navg_max_d_angle = $%f$\navg_slim_energy = $%f$\n", min_avg / cnt, max_avg / cnt, avg_slim_energy / cnt);
		printf("min_d_angle: <6 %f;   <12 %f;  <18 %f\n", cmp_cnt[0] / cnt, cmp_cnt[1] / cnt, cmp_cnt[2] / cnt);
		printf("max_d_angle: >174 %f; >168 %f; >162 %f\n", cmp_cnt[5] / cnt, cmp_cnt[4] / cnt, cmp_cnt[3] / cnt);
	}

	void output_tetmesh(TetMeshOut* out, map_int& vertex_relation, map_int_pair& new_vertices, vector<map_int>& vertex_marker_to_index_array, 
		vector<map_int>& vd_to_index_array, std::chrono::milliseconds& writing_time_spent, string filename = "final")
	{
		map_int& vertex_marker_to_index = vertex_marker_to_index_array[0];
		vector<vec3> cavity_vertices;
		vector<int> cavity_triangles;

		for (int i = 0; i < out->points.size() && new_vertices.size() > 0; i++) {
			if (!(EXISTS(vertex_relation, i))) {
				vertex_relation.insert({ i, i });
			}
			cavity_vertices.push_back({ out->points[i].pos[0], out->points[i].pos[1], out->points[i].pos[2] });
		}

		std::unordered_set<pair<int, int>> all_newly_inserted_edges;

		int aa = 0;
		int temp_tet_count = 0;
		for (int i = 0; i < out->tets.size() && new_vertices.size() > 0; i++) {
			auto vs = vector<int>(4);
			vector<int> newly_inserted_global;
			int index = -1;
			for (int j = 0; j < 4; j++) {
				int vertex_index = out->tets[i].v[j];
				int face_index = out->tets[i].face_idx[j];
				if (out->faces[face_index].face_marker > 0) {
					// It is a boundary face
					vector<int> newly_inserted;

					for (int k = 0; k < 4; k++) {
						if (k == j) continue;
						int x = out->tets[i].v[k];
						if (out->points[x].point_marker > 0 && out->points[x].point_marker < TEST_NUMBER) {
							newly_inserted.push_back(x);
						}
					}
					if (newly_inserted.size() == 2) {
						// These two points form a newly inserted edge
						pair<int, int> edge = { vertex_relation[MIN(newly_inserted)], vertex_relation[MAX(newly_inserted)] };

						vector<int> vec1 = PAIR_TO_VECTOR(new_vertices[out->points[edge.first].point_marker - 1]);
						vector<int> vec2 = PAIR_TO_VECTOR(new_vertices[out->points[edge.second].point_marker - 1]);

						if (vec1[0] == vec1[1] && vec2[0] == vec2[1]) {
							// Handle already existing edge
							PRINT(out->points[edge.first].pos);
							PRINT(out->points[edge.second].pos);
							cout << "." << endl;
							for (int k = 0; k < 4; k++) {
								if (k == j) continue;
								cavity_triangles.push_back(vertex_relation[out->tets[i].v[k]]);
							}
						}
						else {
							all_newly_inserted_edges.insert(edge);
						}

					}
					else if (newly_inserted.size() == 1) {
						auto pair = new_vertices[out->points[newly_inserted[0]].point_marker - 1];
						if (pair.first == pair.second) {
							for (int k = 0; k < 4; k++) {
								if (k == j) continue;
								cavity_triangles.push_back(vertex_relation[out->tets[i].v[k]]);
							}
						}
					}
					else if (newly_inserted.size() == 3) {
						for (int k = 0; k < 4; k++) {
							if (k == j) continue;
							cavity_triangles.push_back(vertex_relation[out->tets[i].v[k]]);
						}
					}

				}
				if (out->points[vertex_index].point_marker > 0 && out->points[vertex_index].point_marker < TEST_NUMBER) {
					index = j;
					newly_inserted_global.push_back(vertex_index);
				}
				vs[j] = vertex_relation[vertex_index] + 1;
			}


			if (newly_inserted_global.size() == 1 && out->tets[i].n[index] == -1) {
				aa++;

				for (int j = 0; j < 4; j++) {//
					if (j == index) continue;
					cavity_triangles.push_back(vertex_relation[out->tets[i].v[j]]);
				}
			}

			if (!newly_inserted_global.empty()) {
				out->tets[i].temporary = true;
				//continue;
				temp_tet_count++;
			}

			/*fprintf(outfile, "%5d  %5d  %5d  %5d 1\n", vs[0], vs[1], vs[2], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[1], vs[2]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[1], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[2], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[1], vs[2], vs[3]);*/
		}

		printf("AA: %d\n", aa);

		if (new_vertices.size() > 0) {
			for (auto& tet : out->tets) {
				if (tet.temporary) continue;

				for (int i = 0; i < 4; i++) {
					int neigh = tet.n[i];
					if (neigh == -1) continue;
					if (out->tets[neigh].temporary) {
						// Add face
						for (int j = 0; j < 4; j++) {
							if (i == j) continue;
							cavity_triangles.push_back(vertex_relation[tet.v[j]]);
						}

					}
				}
			}
		}
		for (const auto& edge : all_newly_inserted_edges) {
			vector<int> vec1 = PAIR_TO_VECTOR(new_vertices[out->points[edge.first].point_marker - 1]);
			vector<int> vec2 = PAIR_TO_VECTOR(new_vertices[out->points[edge.second].point_marker - 1]);
			vector<int> vec3;

			if (edge.first == 1543) {
				printf("Here\n");
			}
			SORT(vec1);
			SORT(vec2);
			vec1.erase(std::unique(vec1.begin(), vec1.end()), vec1.end());
			vec2.erase(std::unique(vec2.begin(), vec2.end()), vec2.end());

			std::set_union(vec1.begin(), vec1.end(), vec2.begin(), vec2.end(), std::back_inserter(vec3));


			if (vec3.size() == 2) {
				throw "here";
				continue;
			}

			if (vec3.size() != 3) {
				for (int k = 0; k < vec3.size(); k++) {
					cout << vec3[k] << " ";
				}
				cout << endl;
				printf("Here\n");
				throw "Assertion Error";
				continue;
			}
			bool non_existent = false;
			for (int k = 0; k < 3; k++) {
				if (!(EXISTS(vertex_marker_to_index, vec3[k]))) {
					printf("Error %d\n", k);
					non_existent = true;
				}
			}

			if (non_existent) {
				for (int k = 0; k < vec3.size(); k++) {
					cout << vec3[k] << " ";
				}
				cout << endl;
				throw "Here";
				continue;
			}
			vector<vertex_descriptor> v = {
				vertex_descriptor(vertex_relation[vertex_marker_to_index[vec3[0]]]),
				vertex_descriptor(vertex_relation[vertex_marker_to_index[vec3[1]]]),
				vertex_descriptor(vertex_relation[vertex_marker_to_index[vec3[2]]]) };

			cavity_triangles.push_back(v[0]);
			cavity_triangles.push_back(v[1]);
			cavity_triangles.push_back(v[2]);
		}


		if (remove_extra && new_vertices.size() > 0) {
			map_int_pair tmp;
			map_int inverse_used_vertices = remove_isolated_vertices(cavity_vertices, cavity_triangles);
			TetMeshIn in = createTetMeshInput(cavity_vertices, cavity_triangles, tmp, map_int(), -1, nullptr, &inverse_used_vertices);
			//in.write_mesh = false;
			//writeOBJ(cavity_vertices, cavity_triangles, "cavity");
			printf("all_newly_inserted_edges size: %d\n", all_newly_inserted_edges.size());

			TetMeshOut* cavity_out = execute_tetgen(in, "cavity", true);
			int prev_vertex_size = out->points.size();
			for (int i = 0; i < cavity_out->points.size(); i++) {
				out->points.push_back(cavity_out->points[i]);
			}

			for (int i = 0; i < cavity_out->tets.size(); i++) {
				Tetrahedron tet = cavity_out->tets[i];
				for (int j = 0; j < 4; j++) {
					int marker = cavity_out->points[tet.v[j]].point_marker - 1;
					if (marker > 0) {
						tet.v[j] = marker;
					}
					else {
						tet.v[j] = tet.v[j] + prev_vertex_size;
					}
				}
				out->tets.push_back(tet);
			}
		}
		auto writing_begin = high_resolution_clock::now();
		FILE* outfile = fopen(string(filename + ".mesh").c_str(), "w");
		FILE* obj_outfile = fopen(string(filename + ".obj").c_str(), "w");
		fprintf(outfile, "MeshVersionFormatted 1\n\n");
		fprintf(outfile, "Dimension\n");
		fprintf(outfile, "3\n\n");

		fprintf(outfile, "\n# Set of mesh vertices\n");
		fprintf(outfile, "Vertices\n");
		fprintf(outfile, "%ld\n", out->points.size());
		for (int i = 0; i < out->points.size(); i++) {
			fprintf(outfile, "%.17g  %.17g  %.17g 0\n", out->points[i].pos[0], out->points[i].pos[1], out->points[i].pos[2]);
			fprintf(obj_outfile, "v %.17g  %.17g  %.17g 0\n", out->points[i].pos[0], out->points[i].pos[1], out->points[i].pos[2]);
		}


		fprintf(outfile, "\n# Set of Tetrahedra\n");
		fprintf(outfile, "Tetrahedra\n");
		fprintf(outfile, "%ld\n", out->tets.size() - ((remove_extra) ? temp_tet_count : 0));
		for (int i = 0; i < out->tets.size(); i++) {
			if (out->tets[i].temporary && remove_extra) {
				continue;
			}
			auto vs = vector<int>(4);
			for (int j = 0; j < 4; j++) {
				int vertex_index = out->tets[i].v[j];
				vs[j] = vertex_index + 1;
			}

			fprintf(outfile, "%5d  %5d  %5d  %5d 1\n", vs[0], vs[1], vs[2], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[1], vs[2]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[1], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[0], vs[2], vs[3]);
			fprintf(obj_outfile, "f %5d %5d %5d\n", vs[1], vs[2], vs[3]);
		}
		fprintf(outfile, "End\n");
		fclose(outfile);
		fclose(obj_outfile);
		writing_time_spent = duration_cast<milliseconds>(high_resolution_clock::now() - writing_begin);
		

	}

	map_int remove_isolated_vertices(vector<vec3>& vertices, vector<int>& triangles) {
		map_int used_vertices;
		vector<vec3> new_vertices;
		for (int v : triangles) {
			used_vertices[v] = -1;
		}
		int index = 0;
		for (int i = 0; i < vertices.size(); i++) {
			if (EXISTS(used_vertices, i)) {
				used_vertices[i] = index;
				index++;
				new_vertices.push_back(vertices[i]);

			}
		}

		for (int& v : triangles) {
			v = used_vertices.at(v);
		}

		vertices.clear();
		vertices = new_vertices;

		return invert_map(used_vertices);
	}



	void repair_tetgen_input(vector<vec3>& vertices, vector<int>& triangles) {
		vector<Point_3> points;
		vector<vector<int>> faces;
		SMesh result;

		for (auto v : vertices) {
			points.push_back({ v[0], v[1], v[2] });
		}


		for (int i = 0; i < triangles.size(); i += 3) {
			faces.push_back({ triangles[i], triangles[i + 1], triangles[i + 2] });
		}

		if (!PMP::orient_polygon_soup(points, faces))
		{
			std::cerr << "Some duplication happened during polygon soup orientation" << std::endl;
		}

		if (!PMP::is_polygon_soup_a_polygon_mesh(faces))
		{
			std::cerr << "Warning: polygon soup does not describe a polygon mesh" << std::endl;
			throw "Error";
		}

		vertices.clear();
		triangles.clear();

		for (auto v : points) {
			vertices.push_back({ v[0], v[1], v[2] });
		}

		for (auto f : faces) {
			triangles.push_back(f[0]);
			triangles.push_back(f[1]);
			triangles.push_back(f[2]);
		}

	}

	void writeOBJ(vector<vec3> vertices, vector<int> triangles, std::string filename) {
		FILE* obj_outfile = fopen(string(filename + ".obj").c_str(), "w");
		printf("WriteObj: %d %d\n", vertices.size(), triangles.size() / 3);
		for (auto& v : vertices) {
			fprintf(obj_outfile, "v %.17g  %.17g  %.17g\n", v.x, v.y, v.z);
		}

		for (int i = 0; i < triangles.size(); i += 3) {
			fprintf(obj_outfile, "f %5d %5d %5d\n", triangles[i] + 1, triangles[i + 1] + 1, triangles[i + 2] + 1);
		}

		fclose(obj_outfile);

	}

	vector<K::Plane_3> apply_PCA(SMesh mesh1, int number_of_cuts) {
		vector<K::Plane_3> planes;
		if (number_of_cuts < 0) return planes;
		//int number_of_cuts = (1 << division_depth) - 1;
		printf("number_of_cuts: %d\n", number_of_cuts);
		//Construct a buffer used by the pca analysis
		int sz = static_cast<int>(mesh1.num_vertices());
		Mat data_pts = Mat(sz, 3, CV_64F);
		auto mesh1_vertex_it = mesh1.vertices().begin();
		for (int i = 0; i < data_pts.rows; i++)
		{
			K::Point_3 v = mesh1.point(*mesh1_vertex_it);
			data_pts.at<double>(i, 0) = CGAL::to_double(v.x());
			data_pts.at<double>(i, 1) = CGAL::to_double(v.y());
			data_pts.at<double>(i, 2) = CGAL::to_double(v.z());
			mesh1_vertex_it++;
		}

		//Perform PCA analysis
		PCA pca_analysis = PCA(data_pts, Mat(), PCA::DATA_AS_ROW, 3);
		//Store the center of the object
		cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
			static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
		//Store the eigenvalues and eigenvectors
		vector<Point3d> eigen_vecs(3);
		for (int i = 0; i < 3; i++)
		{
			eigen_vecs[i] = Point3d(pca_analysis.eigenvectors.at<double>(i, 0),
				pca_analysis.eigenvectors.at<double>(i, 1), pca_analysis.eigenvectors.at<double>(i, 2));
		}

		K::Vector_3 normal = K::Vector_3(eigen_vecs[0].x, eigen_vecs[0].y, eigen_vecs[0].z);

		Mat result = pca_analysis.project(data_pts);
		std::vector<double> projections = result.col(0);
		std::sort(projections.begin(), projections.end());

		for (int i = 1; i <= number_of_cuts; i++) {
			int index = int(double(i) / (number_of_cuts + 1) * projections.size());
			Mat median_proj = Mat(1, 3, CV_64F);
			median_proj.at<double>(0, 0) = projections[index];
			median_proj.at<double>(0, 1) = 0;
			median_proj.at<double>(0, 2) = 0;

			result = pca_analysis.backProject(median_proj);

			Point_3 pnt(result.at<double>(0, 0), result.at<double>(0, 1), result.at<double>(0, 2));
			K::Plane_3 cut_plane(pnt, normal);
			planes.push_back(cut_plane);
		}


		return planes;
	}

	void find_intersections(SMesh& right_mesh)
	{
		vector<pair<face_descriptor, face_descriptor>> inters;
		PMP::self_intersections(faces(right_mesh), right_mesh, std::back_inserter(inters));

		for (auto inter : inters) {
			auto face1 = inter.first;
			auto face2 = inter.second;
			for (halfedge_descriptor hi : halfedges_around_face(right_mesh.halfedge(face1), right_mesh))
			{
				Point_3 p = right_mesh.point(target(hi, right_mesh));
				printf("(%f, %f, %f),", p[0], p[1], p[2]);
			}
			//printf("-----");
			for (halfedge_descriptor hi : halfedges_around_face(right_mesh.halfedge(face2), right_mesh))
			{
				Point_3 p = right_mesh.point(target(hi, right_mesh));
				//printf("(%f, %f, %f),", p[0], p[1], p[2]);
			}
			printf("\n");
		}
		printf("Intersection counts: %zd\n", inters.size());
	}
	PCA pca_analysis;

	K::Vector_3 normal;
	vector<SMesh> clip(SMesh& mesh, vector<map_int>& boundary_maps, vector<K::Plane_3>& cut_planes, map_int_pair& new_vertices) {

		bool bbox = false;
		size_t number_of_pieces = cut_planes.size() + 1;
		int my_cnt = 0;
		for (int i = 0; i < cut_planes.size(); i++) {
			K::Plane_3& cut_plane = cut_planes[i];
			//printf("Plane %d\n", i);
			intersect_and_insert_points(mesh, cut_plane, new_vertices);
		}
#if !CPU_MODE
		vector<std::ofstream> out_streams(number_of_pieces);
#endif
		vector<SMesh> meshes(number_of_pieces);

		for (int i = 0; i < number_of_pieces; i++) {	
#if CPU_MODE
			meshes[i].reserve(mesh.number_of_vertices(), mesh.number_of_edges() / number_of_pieces, mesh.number_of_faces() / number_of_pieces);
#else 
			string file_name = "part_" + std::to_string(i) + ".obj";
			auto& o = out_streams[i];
			o.open(file_name);
			o.precision(20);
#endif
		}
		//CGAL::IO::write_OBJ("deneme.obj", mesh, CGAL::parameters::stream_precision(64));
		
		vector<Point_3> vertices_a;
		std::chrono::steady_clock::time_point start_io;
		for (auto vd : mesh.vertices()) {
			auto& v = mesh.point(vd);
		
#if !CPU_MODE	
			start_io = high_resolution_clock::now();
			for (auto& o : out_streams) {
				o << "v " << v[0] << " " << v[1] << " " << v[2] << endl;
			}
			total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();
#else
			for (auto& m : meshes) {
				auto new_vd = m.add_vertex(v);
			}
#endif
			vertices_a.push_back(v);

		}

		for (auto fd : mesh.faces()) {
			vector<int> indices;
			indices.reserve(3);
			CGAL::Vertex_around_face_circulator<SMesh> vcirc(mesh.halfedge(fd), mesh), done(vcirc);
			do {
				indices.push_back(*vcirc++);
			} while (vcirc != done);

			int v1 = indices[0];
			int v2 = indices[1];
			int v3 = indices[2];
			Point_3 centroid = CGAL::centroid(vertices_a[v1], vertices_a[v2], vertices_a[v3]);
			size_t part_index = -1;
			for (int i = 0; i < cut_planes.size() - 1; i++) {
				if (cut_planes[i].oriented_side(centroid) != cut_planes[i + 1].oriented_side(centroid)) {
					part_index = cut_planes.size() - (i + 1);
					break;
				}
			}
			// That means it belongs to either first or last piece
			if (part_index == -1) {
				part_index = (IS_POS_PLANE(cut_planes[0], centroid)) ? 0 : cut_planes.size();
			}
#if !CPU_MODE
			start_io = high_resolution_clock::now();
			auto& o = out_streams[part_index];
			o << "f " << v1 + 1 << " " << v2 + 1 << " " << v3 + 1 << endl;
			total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();
#else
			SMesh& m = meshes[part_index];
			m.add_face(vertex_descriptor(v1), vertex_descriptor(v2), vertex_descriptor(v3));
#endif
		}


#if !CPU_MODE
		start_io = high_resolution_clock::now();
		for (auto& o : out_streams) {
			o.close();
		}
		total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();

		//for (int i = 0; i < out_streams.size(); i++) {
		//	string file_name = "part_" + std::to_string(i) + ".obj";
		//	generateSmesh(file_name, meshes[i], bbox, bbox);
		//}
#endif

		printf("START: Closing border\n");
		map_int res;
		for (int i = 0; i < meshes.size() - 1; i++) {
			map_int vertex_neighbour;
#if CPU_MODE
			res = close_borders_between_two_meshes(meshes[i], meshes[i + 1], cut_planes[i], new_vertices, vertex_neighbour);
#else
			string left_file_name = "part_" + std::to_string(i) + ".obj";
			string right_file_name = "part_" + std::to_string(i + 1) + ".obj";
			SMesh left, right;
			start_io = high_resolution_clock::now();
			bool readSuccess = CGAL::IO::read_OBJ(left_file_name, left) && CGAL::IO::read_OBJ(right_file_name, right);
			total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();
			if (!readSuccess) {
				throw "Read unsuccessful";
			}

			res = close_borders_between_two_meshes(left, right, cut_planes[i], new_vertices, vertex_neighbour);
			start_io = high_resolution_clock::now();
			CGAL::IO::write_OBJ(left_file_name, left, CGAL::parameters::stream_precision(64));
			CGAL::IO::write_OBJ(right_file_name, right, CGAL::parameters::stream_precision(64));
			total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();
#endif

			boundary_maps.push_back(res);
		}
		printf("FINISH: Closing border\n");

		for (auto& vd : meshes[0].vertices()) {
			if (EXISTS(new_vertices, (int)vd)) {
				my_cnt++;
			}
		}

		cout << "my_cnt: " << my_cnt << " size: " << new_vertices.size() << endl;

#if !CPU_MODE
		//for (int i = 0; i < meshes.size(); i++) {
		//	string file_name = "part_" + std::to_string(i) + ".obj";
		//	SMesh mesh;

		//	bool readSuccess = CGAL::IO::read_OBJ(file_name, mesh);

		//	if (!readSuccess) {
		//		throw "Read unsuccessful";
		//	}


		//	PMP::remove_isolated_vertices(mesh);
		//	CGAL::IO::write_OBJ(file_name, mesh, CGAL::parameters::stream_precision(20));
		//}
		//meshes.clear();
#else

#endif
		//for (int i = 0; i < meshes.size(); i++) {
		//	string file_name = "part_" + std::to_string(i) + ".obj";

		//	CGAL::IO::write_OBJ(file_name, meshes[i]);
		//}
		return meshes;
	}

	void hole_filling(SMesh& mesh) {
		std::vector<halfedge_descriptor> border_cycles;

		// collect one halfedge per boundary cycle
		PMP::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));
		printf("border_cycles.size() == %zd\n", border_cycles.size());
		for (halfedge_descriptor h : border_cycles)
		{
			vector<face_descriptor>  patch_facets;
			vector<vertex_descriptor> patch_vertices;
			PMP::triangulate_hole(mesh,
				h,
				std::back_inserter(patch_facets));
			std::cout << "* Number of facets in constructed patch: " << patch_facets.size() << std::endl;
			std::cout << "  Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
		}

	}





	typedef CGAL::Triangulation_2_projection_traits_3<K> P_traits;
	typedef CGAL::Triangulation_vertex_base_with_info_2<vertex_descriptor, P_traits> Vb;
	typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, P_traits>    Fbb;
	typedef CGAL::Constrained_triangulation_face_base_2<P_traits, Fbb>        Fb;
	typedef CGAL::Triangulation_data_structure_2<Vb, Fb>               TDS;
	typedef CGAL::Exact_predicates_tag                                Itag;
	typedef CGAL::Constrained_Delaunay_triangulation_2<P_traits, TDS, Itag>  CDT;
	typedef CDT::Point                                                CDTPoint;
	typedef CDT::Face_handle                                          Face_handle;

	map_int close_borders_between_two_meshes(SMesh& left_mesh, SMesh& right_mesh, K::Plane_3& cut_plane, map_int_pair& new_vertices, map_int& vertex_neighbours)
	{
		//printf("Left mesh: %zd, Right Mesh: %zd\n", left_mesh.faces().size(), right_mesh.faces().size());
		std::vector<halfedge_descriptor> border_cycles;
		auto& x_mesh = left_mesh;
		auto& y_mesh = right_mesh;

		// collect one halfedge per boundary cycle
		CGAL::Polygon_mesh_processing::extract_boundary_cycles(x_mesh, back_inserter(border_cycles));


		unsigned int nb_holes = 0;
		size_t nb_new_facets = 0;
		map_int neighbour_relations;

		vector<vector<std::pair<Point_3, vertex_descriptor>>> polylines;
		vector<std::pair<int, vector<int>>> polyline_nest;
		for (halfedge_descriptor h : border_cycles) {
			vector<std::pair<Point_3, vertex_descriptor>> polyline;;
			typedef CGAL::Halfedge_around_face_circulator<SMesh> Hedge_around_face_circulator;
			Hedge_around_face_circulator circ(h, x_mesh), done(circ);
			do {
				vertex_descriptor vd = x_mesh.target(*circ);
				Point_3 point = x_mesh.point(vd);
				polyline.push_back({ point, vd });
			} while (++circ != done);
			polylines.push_back(polyline);
		}
		P_traits traits(cut_plane.orthogonal_vector());
		sort(polylines.begin(), polylines.end(), [traits](vector<std::pair<Point_3, vertex_descriptor>>& a, vector<std::pair<Point_3, vertex_descriptor>>& b) {
			vector<Point_3> aa, bb;

			for (auto i : a) {
				aa.push_back(i.first);
			}
			for (auto i : b) {
				bb.push_back(i.first);
			}
			return CGAL::polygon_area_2(aa.begin(), aa.end(), traits) < CGAL::polygon_area_2(bb.begin(), bb.end(), traits);
			});

		vector<vector<Point_3>> polylines_points;

		for (auto x : polylines) {
			vector<Point_3> v;
			for (auto y : x) {
				v.push_back(y.first);
			}
			polylines_points.push_back(v);
		}

		if (polylines.size() != border_cycles.size()) {
			printf("%zd != %zd\n", polylines.size(), border_cycles.size());
			throw "polylines.size() != border_cycles.size()";
		}
		//printf("polylines.size() = %zd\n", polylines.size());
		for (int i = 0; i < polylines.size(); i++) {
			auto& polyline = polylines[i];
			//cout << "Polyline size: " << polyline.size() << endl;
			Point_3 point = polyline[0].first;
			int parent_index = -1;
			auto parent_it = polyline_nest.begin();
			for (auto it = polyline_nest.begin(); it != polyline_nest.end(); it++) {
				if (CGAL::bounded_side_2(polylines_points[it->first].begin(), polylines_points[it->first].begin(), point, traits)) {
					parent_index = it - polyline_nest.begin();
					parent_it = it;
				}
			}
			if (parent_index == -1) {
				polyline_nest.push_back({ i, {} });
			}
			else {
				parent_it->second.push_back(i);
			}
		}
		CDT cdt(traits);

		unordered_map<vertex_descriptor, typename CDT::Vertex_handle> vertices;

		for (auto it = polyline_nest.begin(); it != polyline_nest.end(); it++) {
			vector<int> cycle_ids = it->second;
			cycle_ids.insert(cycle_ids.begin(), it->first);
			for (int cycle_id : cycle_ids) {
				std::vector<std::pair<Point_3, vertex_descriptor>> polygon = polylines[cycle_id];
				cdt.insert(polygon.begin(), polygon.end());
			}
		}

		for (typename CDT::Vertex_handle v : cdt.finite_vertex_handles()) {
			vertices[v->info()] = v;
		}

		for (auto it = polyline_nest.begin(); it != polyline_nest.end(); it++) {
			vector<int> cycle_ids = it->second;
			cycle_ids.insert(cycle_ids.begin(), it->first);
			for (int cycle_id : cycle_ids) {
				auto& polygon = polylines[cycle_id];
				for (std::size_t i = 0; i < polygon.size(); ++i) {
					const std::size_t ip = (i + 1) % polygon.size();
					auto a = vertices[polygon[i].second];
					auto b = vertices[polygon[ip].second];
					if (a != b) {
						cdt.insert_constraint(a, b);
					}
				}

			}
		}

		//Mark facets that are inside the domain bounded by the polygon
		mark_domains(cdt);
		int count = 0;
		vector<face_descriptor> faces;
		for (Face_handle f : cdt.finite_face_handles())
		{
			if (f->info().in_domain()) {
				++count;
				auto face1 = x_mesh.add_face(f->vertex(0)->info(), f->vertex(2)->info(), f->vertex(1)->info());
				faces.push_back(face1);
				if (face1 == SMesh::null_face()) {
					throw "Unexpected null_face";
				}

			}

		}
		PMP::stitch_borders(x_mesh);
		vector<face_descriptor>  patch_facets;
		vector<vertex_descriptor>  patch_vertices;
		std::copy(faces.begin(), faces.end(), back_inserter(patch_facets));

		PMP::refine(x_mesh, faces, back_inserter(patch_facets), back_inserter(patch_vertices), CGAL::parameters::density_control_factor(density_control));
		//cout << "There are " << count << " facets in the domain. Refine: " << patch_facets.size() << endl;
		nb_new_facets += patch_facets.size();
		PMP::stitch_borders(x_mesh);

		for (face_descriptor fi : patch_facets) {
			halfedge_descriptor hf = x_mesh.halfedge(fi);

			vector<vertex_descriptor> vis;
			for (halfedge_descriptor hi : halfedges_around_face(hf, x_mesh))
			{
				vertex_descriptor vd = target(hi, x_mesh);
				vertex_descriptor vi;
				if (EXISTS(new_vertices, vd)) {
					vi = vd;
				}
				else {
					Point_3 p = x_mesh.point(vd);
					vi = y_mesh.add_vertex(p);
				}
				vis.push_back(vi);
			}

			// Reverse the face
			auto tmp = vis[1];
			vis[1] = vis[2];
			vis[2] = tmp;

			face_descriptor fi_2 = y_mesh.add_face(vis);
			if (fi_2 == SMesh::null_face()) {
				throw "Unexpected null_face";
			}
			//printf("Face: %d %d %d %f %d\n", fi.idx(), fi, fi.is_valid(), mesh1.point(mesh1.source(mesh1.halfedge(fi))).x(), *(mesh1.faces_begin() + fi.idx()));
			//neighbour_relations.push_back({ fi.idx(), fi_2.idx() });
			neighbour_relations[fi_2.idx()] = fi.idx();
		}

		PMP::stitch_borders(y_mesh);

		/*std::cout << std::endl;
		std::cout << nb_holes << " holes have been filled" << std::endl;
		std::cout << nb_new_facets << " facets have been created" << std::endl;*/

		if (nb_new_facets != neighbour_relations.size()) {
			throw "nb_new_facets and neighbour_relations.size() must be equal";
		}
		//find_intersections(left_mesh);
		//printf("----\n");
		//find_intersections(right_mesh);
		return neighbour_relations;

	}

	void intersect_and_insert_points(SMesh& mesh1, K::Plane_3& cut_plane, map_int_pair& new_vertices)
	{
		unsigned int new_pnt_count = 0;
		vector<vector<halfedge_descriptor>> face_vertex_map(mesh1.number_of_faces());
		int number_of_edges = mesh1.number_of_edges();
#if false && CPU_MODE
#pragma omp parallel for
#endif
		for (int i = 0; i < number_of_edges; i++) {

			Point_3 v1, v2;
			edge_descriptor edge;
			vertex_descriptor vd1, vd2;
#if false && CPU_MODE
#pragma omp critical
#endif	
			{
				edge = *(mesh1.edges_begin() + i);
				vd1 = source(edge, mesh1);
				vd2 = target(edge, mesh1);

				v1 = mesh1.point(vd1);
				v2 = mesh1.point(vd2);
			}
			K::Segment_3 seg(v1, v2);
			auto result = intersection(seg, cut_plane);
			if (result) {

				// Edge-edge intersection is handled  -20.4029 -212.357 -74.0186
				if (const K::Segment_3* s = boost::get<K::Segment_3>(&*result)) {
					continue;
				}

				const K::Point_3* p = boost::get<Point_3 >(&*result);
				double thresh = threshold;
				bool close_to_first_end = sqrt(CGAL::squared_distance(cut_plane, v1)) <= thresh;
				bool close_to_second_end = sqrt(CGAL::squared_distance(cut_plane, v2)) <= thresh;
				if (close_to_first_end || close_to_second_end) {
					auto closest_vd = close_to_first_end ? vd1 : vd2;
					new_vertices.insert({ closest_vd, {closest_vd, closest_vd} });
					continue;
				}
#if CPU_MODE
#pragma omp critical
#endif
				{
					halfedge_descriptor new_he = CGAL::Euler::split_edge(edge.halfedge(), mesh1);
					vertex_descriptor new_vd = target(new_he, mesh1);
					mesh1.point(new_vd) = *p;
					auto f1 = mesh1.face(new_he);
					auto f2 = mesh1.face(mesh1.opposite(new_he));
					new_vertices.insert({ new_vd, {vd1, vd2} });
					face_vertex_map[f1].push_back(new_he);
					face_vertex_map[f2].push_back(mesh1.opposite(mesh1.next(new_he)));
					new_pnt_count++;
				}
			}
		}

		printf("Number of new points inserted: %d\n", new_pnt_count);

		// Connect consecutive newly inserted points
		for (vector<halfedge_descriptor> halfedges : face_vertex_map) {
			size_t size = halfedges.size();
			if (size == 2) {
				CGAL::Euler::split_face(halfedges[0], halfedges[1], mesh1);
			}
			else if (size == 1) {
				halfedge_descriptor he = halfedges[0];
				halfedge_descriptor other_he = mesh1.next(mesh1.next(he));
				CGAL::Euler::split_face(he, other_he, mesh1);
			}
			else if (size != 0) {
				throw "Invalid halfedges size";
			}
		}

		vector<halfedge_descriptor> halfedges;
		halfedges.reserve(4);
		// Convert quads to triangles
		for (auto face : mesh1.faces()) {

			auto he = mesh1.halfedge(face);
			halfedges.clear();
			CGAL::Halfedge_around_face_circulator<SMesh> vcirc(he, mesh1), done(vcirc);
			do {
				halfedges.push_back(*vcirc);
			} while (++vcirc != done);
			if (halfedges.size() > 4) {
				cout << mesh1.point(mesh1.target(halfedges[0])) << endl;
				throw "Assumption violation: Encountered  a face with number of edges > 4 ";
			}
			if (halfedges.size() == 4) {
				CGAL::Euler::split_face(halfedges[0], halfedges[2], mesh1);
			}
		}
		// Donee
	}

	static void smesh_to_eigen(SMesh& mesh, Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
		V.resize(mesh.number_of_vertices(), 3);
		F.resize(mesh.number_of_faces(), 3);
		map_int vd_to_index;

		int v_i = 0;
		for (vertex_descriptor vd : mesh.vertices()) {
			auto point = mesh.point(vd);
			V.row(v_i) << point[0], point[1], point[2];
			vd_to_index[vd] = v_i;
			v_i++;
		}
		int f_i = 0;
		for (face_descriptor fd : mesh.faces()) {
			vector<int> indices;
			for (auto& he : mesh.halfedges_around_face(mesh.halfedge(fd))) {
				vertex_descriptor vd = mesh.source(he);
				indices.push_back(vd_to_index[vd]);
			}
			F.row(f_i) << indices[0], indices[1], indices[2];
			f_i++;
		}

	}

	map_int smesh_to_tetgen(SMesh& mesh, vector<vec3>& vertices, vector<int>& triangles) {
		map_int vd_to_index;
		int vertex_index = 0;
		vertices.reserve(mesh.number_of_vertices());
		triangles.reserve(3 * mesh.number_of_faces());
		int cnt = 0;
		for (auto& vd : mesh.vertices()) {
			//if (EXISTS(new_vertices, vd)) {
			//	cnt++;
			//	//modified_new_vertices.insert(vertex_index);
			//}
			//printf("%d\n", vd);
			K::Point_3 point = mesh.point(vd);
			vertices.push_back({ point[0], point[1], point[2] });
			vd_to_index[vd] = vertex_index;
			vertex_index++;
		}

		//printf("##########################   %d %d \n", cnt, new_vertices.size());
		for (auto& fd : mesh.faces()) {
			for (auto& he : mesh.halfedges_around_face(mesh.halfedge(fd))) {
				vertex_descriptor vd = mesh.source(he);
				triangles.push_back(vd_to_index[vd]);
			}
		}

		return vd_to_index;
	}



	void mark_domains(CDT& ct, Face_handle start, int index, std::list<CDT::Edge>& border)
	{
		if (start->info().nesting_level != -1) {
			return;
		}
		std::list<Face_handle> queue;
		queue.push_back(start);
		while (!queue.empty()) {
			Face_handle fh = queue.front();
			queue.pop_front();
			if (fh->info().nesting_level == -1) {
				fh->info().nesting_level = index;
				for (int i = 0; i < 3; i++) {
					CDT::Edge e(fh, i);
					Face_handle n = fh->neighbor(i);
					if (n->info().nesting_level == -1) {
						if (ct.is_constrained(e)) border.push_back(e);
						else queue.push_back(n);
					}
				}
			}
		}
	}

	void mark_domains(CDT& cdt)
	{
		for (CDT::Face_handle f : cdt.all_face_handles()) {
			f->info().nesting_level = -1;
		}
		std::list<CDT::Edge> border;
		mark_domains(cdt, cdt.infinite_face(), 0, border);
		while (!border.empty()) {
			CDT::Edge e = border.front();
			border.pop_front();
			Face_handle n = e.first->neighbor(e.second);
			if (n->info().nesting_level == -1) {
				mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
			}
		}
	}
};


#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/refine_mesh_3.h>

// Domain
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;
#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif
// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
using namespace CGAL::parameters;


//https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
bool hasEnding(string const& fullString, string const& ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}
//namespace igl_tetgen = igl::copyleft::tetgen;
//namespace igl_cgal = igl::copyleft::cgal;
int main(int argc, char* argv[])
{

	auto start = high_resolution_clock::now();
	auto time1 = high_resolution_clock::now();
	if (argc < 3) return -1;

	printf("Welcome!\n");

	llint division_depth = std::stoi(argv[3]);
	if (strlen(argv[2]) != 1) return -1;
	char mode = argv[2][0];

	if (mode == 'l') {
		//Eigen::MatrixXd V;
		//Eigen::MatrixXi F;
		//Eigen::MatrixXd B;

		//Eigen::MatrixXd TV;
		//Eigen::MatrixXi TT;
		//Eigen::MatrixXi TF;
		//// Load a surface mesh
		//igl::read_triangle_mesh(argv[1], V, F);
		//igl_cgal::RemeshSelfIntersectionsParam inters_param;
		//inters_param.detect_only = true;
		//Eigen::MatrixXd VV;
		//Eigen::MatrixXd FF;
		//Eigen::MatrixXd IF;
		//Eigen::VectorXd J;
		//Eigen::VectorXd IM;
		////igl_cgal::remesh_self_intersections(V, F, inters_param, VV, FF, IF, J, IM);

		//printf("%d, %d, %d, %d, %d\n", J.rows(), VV.rows(), FF.rows(), IF.rows(), IM.rows());

		//igl_tetgen::CDTParam param;
		//param.use_bounding_box = true;
		//param.bounding_box_scale = 1.01;
		////param.flags = "q5.000YnAfQ";
		//// Tetrahedralize the interior
		//igl_tetgen::cdt(V, F, param, TV, TT, TF);
		////igl::copyleft::tetgen::tetrahedralize(V, F, "q1.414YnAfQ", TV, TT, TF);
		//Eigen::MatrixXi TFF;
		//Eigen::VectorXd W;

		//igl::barycenter(TV, TT, B);

		//igl::FastWindingNumberBVH fwn_bvh;
		//igl::fast_winding_number(V.cast<float>(), F, 2, fwn_bvh);
		//igl::fast_winding_number(fwn_bvh, 2, B.cast<float>(), W);

		////igl::winding_number(V, F, B, W);

		//// Extract interior tets
		//Eigen::MatrixXi CT((W.array() > 0.5).count(), 4);
		//{
		//	size_t k = 0;
		//	for (size_t t = 0; t < TT.rows(); t++)
		//	{
		//		if (W(t) > 0.5)
		//		{
		//			CT.row(k) = TT.row(t);
		//			k++;
		//		}
		//	}
		//}


		////TT = CT;
		//// Print out tetrahedral mesh
		//{
		//	TFF.resize(TT.rows() * 4, 3);
		//	TFF.setZero();
		//	for (int i = 0; i < TT.rows(); i += 1) {
		//		TFF.row(4 * i) << TT(i, 0), TT(i, 1), TT(i, 2);
		//		TFF.row(4 * i + 1) << TT(i, 0), TT(i, 1), TT(i, 3);
		//		TFF.row(4 * i + 2) << TT(i, 0), TT(i, 2), TT(i, 3);
		//		TFF.row(4 * i + 3) << TT(i, 1), TT(i, 2), TT(i, 3);
		//	}

		//	igl::write_triangle_mesh("deneme.obj", TV, TFF);
		//}
		////igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Yg", TV, TT, TF);
		//return 0;
	}

	SMesh mesh;
	bool res;
	if (hasEnding(argv[1], ".stl")) {
		res = CGAL::IO::read_STL(argv[1], mesh);
	}
	else if (hasEnding(argv[1], ".obj")) {
		std::vector<Point_3> points;
		std::vector<std::vector<std::size_t> > faces;
		if (!CGAL::IO::read_polygon_soup(argv[1], points, faces))
		{
			std::cerr << "Warning: cannot read polygon soup" << std::endl;
			return -1;
		}

		PMP::repair_polygon_soup(points, faces, params::all_default());

		if (!PMP::orient_polygon_soup(points, faces))
		{
			std::cerr << "Some duplication happened during polygon soup orientation" << std::endl;
		}

		if (!PMP::is_polygon_soup_a_polygon_mesh(faces))
		{
			std::cerr << "Warning: polygon soup does not describe a polygon mesh" << std::endl;
			return -1;
		}

		PMP::polygon_soup_to_polygon_mesh(points, faces, mesh, CGAL::parameters::all_default());

		float bb_scale = 1.01;

		//CGAL::Bbox_3 bbox = PMP::bbox(mesh);
		//K::Vector_3 min_corner = { bbox.xmin(), bbox.ymin(), bbox.zmin() };
		//K::Vector_3 max_corner = { bbox.xmax(), bbox.ymax(), bbox.zmax() };
		//K::Vector_3 center = (max_corner + min_corner) / 2;

		//min_corner -= center;
		//max_corner -= center;

		//max_corner *= bb_scale;
		//min_corner *= bb_scale;
		//Eigen::MatrixXd V, BV;
		//Eigen::MatrixXi F, BF;
		//MetGen::smesh_to_eigen(mesh, V, F);
		//igl::bounding_box(V, BV, BF);

		//const auto mid = (BV.colwise().minCoeff() + BV.colwise().maxCoeff()).eval() * 0.5;
		//BV.rowwise() -= mid;
		//BV.array() *= bb_scale;
		//BV.rowwise() += mid;

		//int prev_size = mesh.number_of_vertices();

		//for (int i = 0; i < BV.rows(); i++) {
		//	mesh.add_vertex({BV(i, 0), BV(i, 1), BV(i, 2)});
		//}
		//for (int i = 0; i < BF.rows(); i++) {
		//	mesh.add_face( vertex_descriptor(BF(i, 0) + prev_size), vertex_descriptor(BF(i, 1) + prev_size), vertex_descriptor(BF(i, 2) + prev_size));
		//}


		res = 1;
		//CGAL::IO::write_OBJ("aa.obj", mesh);
	}
	else if (hasEnding(argv[1], ".ply")) {
		res = CGAL::IO::read_PLY(argv[1], mesh);
	}
	else {
		throw "Not supported input mesh type";
	}

	auto time2 = high_resolution_clock::now();
	std::vector<std::pair<face_descriptor, face_descriptor>> faces;
	//PMP::self_intersections(mesh, std::back_inserter(faces));

	printf("#V: %zd #F: %zd Intersection count: %d\n", mesh.number_of_vertices(), mesh.number_of_faces(), faces.size());
	auto time3 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(time3 - time1);
	auto reading_time = duration.count();
	printf("Reading took $%lld$ ms\n", duration.count());
	milliseconds writing_time;
	printf("Reading result: %s\n", res ? "OK" : "Fail");
	if (!res) {
		return 1;
	}
	/*for (int i = 0; i < faces.size(); i++) {
		mesh.remove_face(faces[i].first);
		mesh.remove_face(faces[i].second);
	}
	mesh.collect_garbage();
	printf("Self-intersection took $%lld$ ms\n", duration_cast<milliseconds>(time3 - time2).count());*/

	vector<map_int> boundary_maps;
	map_int_pair new_vertices;
	vector<map_int> vd_to_index_array;

	if (mode == 'x') { // Subdivision
		printf("Applying subdivision with #iterations=%lld\n", division_depth);
		printf("Previous #V: %zd, #F: %zd\n", mesh.number_of_vertices(), mesh.number_of_faces());
		CGAL::Subdivision_method_3::CatmullClark_subdivision(mesh, params::number_of_iterations(division_depth));
		PMP::triangulate_faces(mesh);
		printf("After #V: %zd, #F: %zd\n", mesh.number_of_vertices(), mesh.number_of_faces());
		string file_name = argv[1];
		file_name += "_subdivided_" + std::to_string(division_depth) + ".obj";
		CGAL::IO::write_OBJ(file_name, mesh);
	}
	else if (mode == 'c') {
		Polyhedron poly;
		CGAL::copy_face_graph(mesh, poly);
		// Create domain
		Mesh_domain domain(poly);
		// Mesh criteria (no cell_size set)
		Mesh_criteria criteria;
		// Mesh generation
		C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, features(domain));
		// Output
		std::ofstream medit_file("out_1.mesh");
		c3t3.output_to_medit(medit_file);
		medit_file.close();
		// Set tetrahedron size (keep cell_radius_edge_ratio), ignore facets
		Mesh_criteria new_criteria(cell_radius_edge_ratio = 3, cell_size = 0.03);

	}
	else if (mode == 'h') {

		if (argc != 7) {
			return -1;
		}

		double density_factor = std::stod(argv[4]);
		double threshold = std::stod(argv[5]);
		bool remove_extra = argv[6][0] == 'y';
		MetGen met(density_factor, threshold, remove_extra);
		//met.hole_filling(mesh);


		int order = 0;

		if (!CGAL::is_triangle_mesh(mesh)) {
			PMP::triangulate_faces(mesh);
		}
		time1 = high_resolution_clock::now();

		vector<K::Plane_3> cut_planes = met.apply_PCA(mesh, division_depth);

		vector<SMesh> meshes = met.clip(mesh, boundary_maps, cut_planes, new_vertices);
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - time1);
		printf("Clip took $%lld$ ms\n", duration.count());

		time1 = high_resolution_clock::now();
		vector<TetMeshOut*> outputs(cut_planes.size() + 1);
		vector<long long> output_times(cut_planes.size() + 1);
		vd_to_index_array.resize(cut_planes.size() + 1);
#if CPU_MODE
#pragma omp parallel for num_threads(division_depth + 1)
#endif
		for (int i = 0; i < outputs.size(); i++) {
			auto tetgen_begin_time = high_resolution_clock::now();
			vector<vec3> vertices;
			vector<int> triangles;


#if !CPU_MODE
			string file_name = "part_" + std::to_string(i) + ".obj";
			//met.prepareMesh(file_name, vertices, triangles);
			SMesh mesh;
			auto start_io = high_resolution_clock::now();
			auto res = CGAL::IO::read_OBJ(file_name, mesh);
			met.total_io += duration_cast<milliseconds>(high_resolution_clock::now() - start_io).count();

			if (!res) {
				return 1;
			}
			PMP::remove_isolated_vertices(mesh);
			//met.find_intersections(mesh);

			map_int vd_to_index = met.smesh_to_tetgen(mesh, vertices, triangles);
#else
			PMP::remove_isolated_vertices(meshes[i]);

			printf("Num threads: %d Input -- #V: %d #E: %d #F: %d\n", omp_get_num_threads(), meshes[i].number_of_vertices(), meshes[i].number_of_edges(), meshes[i].number_of_faces());
			//CGAL::IO::write_OBJ("part_" + std::to_string(i) + ".obj", meshes[i]);
			//meshes[i].clear();
			//CGAL::IO::read_OBJ("part_" + std::to_string(i) + ".obj", meshes[i]);
			printf("Num threads: %d Input -- #V: %d #E: %d #F: %d\n", omp_get_num_threads(), meshes[i].number_of_vertices(), meshes[i].number_of_edges(), meshes[i].number_of_faces());

			map_int vd_to_index = met.smesh_to_tetgen(meshes[i], vertices, triangles);
#endif

			TetMeshIn in = met.createTetMeshInput(vertices, triangles, new_vertices, vd_to_index, i, &(boundary_maps[0]));
			//in.write_mesh = false;
			outputs[i] = met.execute_tetgen(in, "part_" + std::to_string(i), true);
			vd_to_index_array[i] = vd_to_index;
			auto output_time = duration_cast<milliseconds>(high_resolution_clock::now() - tetgen_begin_time);
			output_times[i] = output_time.count();
		}
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - time1);
		printf("Per-thread timings: ");
		for (int i = 0; i < output_times.size(); i++) {
			printf("%d ms ", output_times[i]);
		}
		printf("\nOverall TetGen took $%lld$ ms\n", duration.count());

		string file_name1 = "part_0.mesh";
		string file_name2 = "part_1.mesh";
		string out_name = "out.mesh";
		time1 = high_resolution_clock::now();

		milliseconds merge_time;
		met.merge(meshes, outputs, boundary_maps, writing_time, merge_time, new_vertices, vd_to_index_array);

		for (auto* o : outputs) {
			delete o;
		}

		outputs.clear();

		duration = duration_cast<milliseconds>(high_resolution_clock::now() - time1);
		cout << "Total IO time (intermediate file saves): " << met.total_io << endl;
		printf("Merge took $%lld$ ms\n", duration.count() - writing_time.count());
		printf("Writing took $%lld$ ms\n", writing_time.count());
	}
	else {
		MetGen met;
		printf("Clip took $%lld$ ms\n", 0);

		//faces.clear();
		//PMP::self_intersections(mesh, std::back_inserter(faces));

		//printf("#V: %zd #F: %zd Intersection count: %d\n", mesh.number_of_vertices(), mesh.number_of_faces(), faces.size());
		//CGAL::IO::write_OBJ("repaired.obj", mesh);
		//met.hole_filling(mesh);
		//CGAL::IO::write_OBJ("repaired.obj", mesh);
		//faces.clear();
		//PMP::self_intersections(mesh, std::back_inserter(faces));

		printf("#V: %zd #F: %zd Intersection count: %d\n", mesh.number_of_vertices(), mesh.number_of_faces(), faces.size());
		map_int vertex_map;
		vector<glm::vec3> vertices;
		vector<int> triangles;
		time1 = high_resolution_clock::now();
		met.smesh_to_tetgen(mesh, vertices, triangles);
		TetMeshIn in = met.createTetMeshInput(vertices, triangles, new_vertices, vertex_map, 1);

		vector<TetMeshOut*> outputs(1);
		vector<SMesh> meshes(1);
		outputs[0] = met.execute_tetgen(in, "standalone_mesh", true);
		meshes[0] = mesh;
		duration = duration_cast<milliseconds>(high_resolution_clock::now() - time1);

		time1 = high_resolution_clock::now();
		milliseconds merge_time;
		met.merge(meshes, outputs, boundary_maps, writing_time, merge_time, new_vertices, vd_to_index_array);
		auto duration2 = duration_cast<milliseconds>(high_resolution_clock::now() - time1);

		printf("\nOverall TetGen took $%lld$ ms\n", duration.count());
		printf("Merge took: $%lld$ ms\n", duration2.count() - writing_time.count());
		printf("Writing duration: $%lld$ ms\n", writing_time.count());
	}

	auto stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	printf("Total duration: $%lld$ ms\n", duration.count());
	printf("Total duration w/o reading: $%lld$ ms\n", duration.count() - reading_time);
	printf("Total duration w/o reading, writing: $%lld$ ms\n", duration.count() - reading_time - writing_time.count());

	return 0;
}