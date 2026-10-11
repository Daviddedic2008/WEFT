#include <optix.h>
#include <math.h>
#include "optix_corr_shared_3d.h"

extern "C" __constant__ CorrParams3D params;

// One ray per (query token, subspace). Launch index is subspace-major so neighbouring
// threads trace into the same GAS; outputs use the token-major logical ray id.
//
// Hit ids written to out_hit_centroid:
//   GEOM_FAN        : codeword id (eid)
//   GEOM_POLAR(_ITER): primitive index = position in the norm-sorted codebook; Stage 2 maps it
//                     to the eid (one coalesced lookup there instead of a dependent load per hit)
extern "C" __global__ void __raygen__corr_3d() {
    uint3 idx = optixGetLaunchIndex();
    int total_rays = optixGetLaunchDimensions().x;
    int nq = total_rays / NUM_3D_SUBSPACES;

    int subspace = idx.x / nq;
    int query_id = idx.x % nq;
    int logical_ray_id = query_id * NUM_3D_SUBSPACES + subspace;

    float3 q = params.query_points[logical_ray_id];
    float qn = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);
    float inv_norm = 1.0f / (qn + 1e-12f);

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 dir    = make_float3(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
    OptixTraversableHandle gas = params.subspace_gas[subspace];

    if (params.geometry == GEOM_POLAR_ITER) {
        // Nearest hit = largest d.c (t = 1/(d.c)). Repeatedly take the nearest hit beyond the
        // previous one: exact top-k in descending score order, no any-hit or closest-hit calls.
        int base = logical_ray_id * params.max_hits;
        int k = params.k_iter < params.max_hits ? params.k_iter : params.max_hits;
        float tmin = 0.0f, tmax = params.tmax[subspace];
        int n = 0;
        for (; n < k; ++n) {
#if OPTIX_VERSION >= 80000
            optixTraverse(gas, origin, dir, tmin, tmax, 0.0f, 255,
                          OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0);
            if (!optixHitObjectIsHit()) break;
            unsigned int prim = optixHitObjectGetPrimitiveIndex();
            float t = optixHitObjectGetRayTmax();
#else
            unsigned int p0 = 0xFFFFFFFFu, p1 = 0, p2 = 0;
            optixTrace(gas, origin, dir, tmin, tmax, 0.0f, 255,
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, p0, p1, p2);
            if (p0 == 0xFFFFFFFFu) break;
            unsigned int prim = p0;
            float t = __uint_as_float(p1);
#endif
            params.out_hit_centroid[base + n] = (int)prim;
            params.out_hit_value[base + n]    = qn / t;
            tmin = nextafterf(t, 3.0e38f);  // the next representable distance: skips only exact ties
        }
        params.out_hit_count[logical_ray_id] = n;
        return;
    }

    unsigned int p0 = (unsigned int)logical_ray_id;
    unsigned int p1 = 0;
    unsigned int p2 = __float_as_uint(qn);
    if (params.geometry == GEOM_POLAR) {
        // Codeword c is the plane x.c = 1. A ray from the origin along unit d meets it at
        // t = 1 / (d.c) (only when d.c > 0), so t <= tmax = 1/tau  <=>  d.c >= tau, exactly.
        optixTrace(gas, origin, dir, 0.0f, params.tmax[subspace], 0.0f, 255,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2);
    } else {
        optixTrace(gas, origin, dir, 0.0f, 2.0f, 0.0f, 255,
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1, p2);
    }
    params.out_hit_count[logical_ray_id] = (int)p1;
}

extern "C" __global__ void __anyhit__corr_3d() {
    int logical_ray_id = (int)optixGetPayload_0();
    int prim_idx = optixGetPrimitiveIndex();

    int entry_id;
    float value;
    if (params.geometry == GEOM_POLAR) {
        // One triangle per codeword, norm-sorted: record the primitive index (Stage 2 maps it).
        // q.c = |q| (d.c) = |q| / t. No memory read at all on this path.
        entry_id = prim_idx;
        value = __uint_as_float(optixGetPayload_2()) / optixGetRayTmax();
    } else {
        int subspace = logical_ray_id % NUM_3D_SUBSPACES;
        int E = params.num_entries;
        entry_id = (prim_idx / 4) % E;
        float3 q = params.query_points[logical_ray_id];
        float3 c = params.centroid_xyz[subspace * E + entry_id];
        value = q.x * c.x + q.y * c.y + q.z * c.z;
        if (value <= 0.0f) {
            optixIgnoreIntersection();
            return;
        }
    }

    unsigned int cnt = optixGetPayload_1();
    optixSetPayload_1(cnt + 1);

    int base = logical_ray_id * params.max_hits;
    if (cnt < (unsigned int)params.max_hits) {
        params.out_hit_centroid[base + cnt] = entry_id;
        params.out_hit_value[base + cnt]    = value;
    } else {
        // Buffer full: keep the best max_hits hits (any-hit order is arbitrary).
        int   min_j = 0;
        float min_v = params.out_hit_value[base];
        for (int j = 1; j < params.max_hits; ++j) {
            float v = params.out_hit_value[base + j];
            if (v < min_v) { min_v = v; min_j = j; }
        }
        if (value > min_v) {
            params.out_hit_centroid[base + min_j] = entry_id;
            params.out_hit_value[base + min_j]    = value;
        }
    }
    optixIgnoreIntersection();
}

// Only reached by GEOM_POLAR_ITER on OptiX < 8 (optixTrace fallback): report the nearest hit.
extern "C" __global__ void __closesthit__corr_3d() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __miss__corr_3d() {}
