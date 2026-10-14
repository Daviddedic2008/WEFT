// Shared between the OptiX device programs (juno_rt_device.cu) and the host benchmark (juno_bench.cu).
#pragma once
#include <optix.h>

struct JunoParams {
    OptixTraversableHandle handle;
    const float2* origins;      // [rays] residual projection of (query, probed cluster) in one subspace
    const float*  tmax;         // [rays] JUNO's dynamic threshold as a maximum ray travel time
    int rays_per_sub;           // rays are subspace-major: ray i is in subspace i / rays_per_sub
    int E;                      // codebook entries per subspace (primitive id = s * E + e)
    int max_hits;               // output slots per ray
    float R2;                   // sphere radius squared
    int*   counts;              // [rays]
    int*   hit_eid;             // [rays * max_hits]
    float* hit_dist;            // [rays * max_hits] distance recovered from t_hit (JUNO Sec. 4.2)
};
