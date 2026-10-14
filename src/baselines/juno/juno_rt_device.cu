// JUNO's RT-core selective L2-LUT construction (Liu et al., ASPLOS '24, Sec. 4.2 and Alg. 2):
// codebook entry e of 2D subspace s is a sphere of radius R at (e.x, e.y, 2s + 1); the residual
// projection of (query, probed cluster) in subspace s is a ray from (x, y, 2s) along +z with
// t_max = 1 - sqrt(R^2 - r^2), so it hits exactly the entries within distance r. The distance is
// recovered from the hit time, d = sqrt(R^2 - (1 - t_hit)^2), without reading the entry.
#include <optix.h>
#include "juno_shared.h"

extern "C" __constant__ JunoParams params;

extern "C" __global__ void __raygen__juno() {
    const unsigned int i = optixGetLaunchIndex().x;
    const int s = (int)(i / (unsigned)params.rays_per_sub);
    const float2 o = params.origins[i];
    unsigned int p0 = i, p1 = 0;
    optixTrace(params.handle, make_float3(o.x, o.y, 2.0f * s), make_float3(0.0f, 0.0f, 1.0f),
               0.0f, params.tmax[i], 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0, p1);
    params.counts[i] = (int)p1;
}

extern "C" __global__ void __anyhit__juno() {
    const unsigned int i = optixGetPayload_0();
    const unsigned int n = optixGetPayload_1();
    if (n < (unsigned)params.max_hits) {
        const float a = 1.0f - optixGetRayTmax();          // t of this candidate hit
        const size_t slot = (size_t)i * params.max_hits + n;
        params.hit_eid[slot] = (int)(optixGetPrimitiveIndex() % (unsigned)params.E);
        params.hit_dist[slot] = sqrtf(fmaxf(params.R2 - a * a, 0.0f));
    }
    optixSetPayload_1(n + 1);
    optixIgnoreIntersection();                              // keep traversing: collect every entry in range
}

extern "C" __global__ void __miss__juno() {}
