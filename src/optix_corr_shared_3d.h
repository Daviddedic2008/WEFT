#pragma once
#include <optix.h>
#include <cuda_runtime.h>

#define NUM_3D_SUBSPACES 32
#define MAX_ENTRIES_PER_SUBSPACE 65536   // codewords per subspace (E) is a runtime value <= this (uint16 eids)
#define MAX_HITS_CAP 256                 // width of the per-ray hit buffers

// Stage-1 scene geometry
#define GEOM_FAN   0   // legacy: disc around each codeword; hits ~every codeword with q.c > 0 (half-space)
#define GEOM_POLAR 1   // exact threshold: plane x.c = 1, ray hits at t = 1/(d.c), tmax = 1/tau
#define GEOM_POLAR_ITER 2  // same scene, no any-hit: k nearest-hit traversals (nearest = best score)

struct CorrParams3D {
    float3* query_points;                                // [rays, 3], ray = token * 32 + subspace
    OptixTraversableHandle subspace_gas[NUM_3D_SUBSPACES];
    float   tmax[NUM_3D_SUBSPACES];                      // polar: 1 / tau_s for the chosen level
    float3* centroid_xyz;                                // fan: codewords [32 * E]
    const int* prim_eid;                                 // polar: [32 * E] primitive (norm-sorted index) -> eid (Stage 2 maps)
    int*    out_hit_centroid;
    float*  out_hit_value;
    int*    out_hit_count;
    int     max_hits;
    int     num_entries;                                 // E
    int     geometry;                                    // GEOM_FAN, GEOM_POLAR or GEOM_POLAR_ITER
    int     k_iter;                                      // GEOM_POLAR_ITER: nearest hits to collect per ray
};
