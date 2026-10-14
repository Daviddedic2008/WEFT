// JUNO's selective L2-LUT construction, RT cores vs CUDA cores, on the same GPU and the same inputs.
//
// JUNO (Liu et al., ASPLOS '24; JUNO++, TACO '25) maps the "which codebook entries lie within r of the
// query's residual projection" test of IVFPQ search onto RT cores (spheres at the entries, one ray per
// (query, probed cluster, 2D subspace), distance from the hit time). Its only no-RT data point runs the
// same OptiX program on an A100, where OptiX emulates traversal in software on CUDA cores, against a
// FAISS baseline on that GPU. This benchmark asks the direct question instead: on one RT-core GPU, is
// the RT version of this stage faster than a plain CUDA kernel computing the same selection?
//
//   rt     : OptiX built-in spheres, any-hit collects (entry, distance) per ray        (JUNO Alg. 2)
//   cuda   : one thread per ray, the subspace's E entries in shared memory, d^2 <= r^2 (same output)
//   dense  : FAISS-style full LUT, all E distances per ray (no selection), for reference
//
// Both selective versions produce the same hit lists (checked on the first batch), so everything
// downstream of this stage (distance accumulation over the selected entries) is identical; the time
// difference of this stage bounds the end-to-end difference RT cores can make to JUNO.
//
// Inputs (prepare_juno_data.py): <dir>/meta.txt "S E Q P R", codebook.bin float32 [S][E][2],
// origins.bin float32 [Q][P][S][2], thresholds.bin float32 [F][S], fracs.txt (F lines).
// Usage: juno_bench <dir> <juno_rt_device.ptx> [--nprobe 8,16,32,64] [--reps 5] [--out results.jsonl]
#include <cuda_runtime.h>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "juno_shared.h"

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define OPTIX_CHECK(x) do { OptixResult r_ = (x); if (r_ != OPTIX_SUCCESS) { \
    fprintf(stderr, "OptiX %s at %s:%d: %s\n", #x, __FILE__, __LINE__, optixGetErrorString(r_)); exit(1); } } while (0)

template <typename T> struct Rec { alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; T data; };
struct Empty {};

static void log_cb(unsigned int level, const char* tag, const char* msg, void*) {
    if (level <= 2) fprintf(stderr, "[optix %u][%s] %s\n", level, tag, msg);
}

template <typename T> static std::vector<T> read_bin(const std::string& path, size_t n) {
    std::vector<T> v(n);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    size_t got = fread(v.data(), sizeof(T), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "%s: expected %zu values, got %zu\n", path.c_str(), n, got); exit(1); }
    return v;
}

// ------------------------------------------------------------------------------------ CUDA versions
// Grid (ceil(rays_per_sub / block), S): every block works on one subspace, whose E entries it stages
// in shared memory; all threads read the same entry at once (a broadcast), one ray per thread.
__global__ void lut_select_cuda(const float2* __restrict__ origins, const float* __restrict__ r2,
                                const float2* __restrict__ entries, int rays_per_sub, int E, int max_hits,
                                int* __restrict__ counts, int* __restrict__ hit_eid, float* __restrict__ hit_dist) {
    extern __shared__ float2 sh[];
    const int s = blockIdx.y;
    for (int k = threadIdx.x; k < E; k += blockDim.x) sh[k] = entries[(size_t)s * E + k];
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rays_per_sub) return;
    const size_t i = (size_t)s * rays_per_sub + j;
    const float2 o = origins[i];
    const float rr = r2[i];
    const size_t base = i * max_hits;
    int n = 0;
    for (int e = 0; e < E; ++e) {
        const float dx = sh[e].x - o.x, dy = sh[e].y - o.y;
        const float d2 = dx * dx + dy * dy;
        if (d2 <= rr) {
            if (n < max_hits) { hit_eid[base + n] = e; hit_dist[base + n] = sqrtf(d2); }
            ++n;
        }
    }
    counts[i] = n;
}

__global__ void lut_dense_cuda(const float2* __restrict__ origins, const float2* __restrict__ entries,
                               int rays_per_sub, int E, float* __restrict__ lut) {
    extern __shared__ float2 sh[];
    const int s = blockIdx.y;
    for (int k = threadIdx.x; k < E; k += blockDim.x) sh[k] = entries[(size_t)s * E + k];
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rays_per_sub) return;
    const size_t i = (size_t)s * rays_per_sub + j;
    const float2 o = origins[i];
    float* out = lut + i * E;
    for (int e = 0; e < E; ++e) {
        const float dx = sh[e].x - o.x, dy = sh[e].y - o.y;
        out[e] = dx * dx + dy * dy;
    }
}

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <data_dir> <ptx> [--nprobe 8,16,32,64] [--reps 5] [--out f.jsonl]\n", argv[0]); return 1; }
    std::string dir = argv[1], ptx_path = argv[2], out_path, nprobe_arg = "8,16,32,64";
    int reps = 5;
    for (int a = 3; a + 1 < argc; a += 2) {
        if (!strcmp(argv[a], "--nprobe")) nprobe_arg = argv[a + 1];
        else if (!strcmp(argv[a], "--reps")) reps = atoi(argv[a + 1]);
        else if (!strcmp(argv[a], "--out")) out_path = argv[a + 1];
    }
    int S, E, Q, P; float R;
    { std::ifstream m(dir + "/meta.txt"); m >> S >> E >> Q >> P >> R; if (!m) { fprintf(stderr, "bad meta.txt\n"); return 1; } }
    std::vector<float> fracs;
    { std::ifstream f(dir + "/fracs.txt"); float x; while (f >> x) fracs.push_back(x); }
    const int F = (int)fracs.size();
    auto codebook = read_bin<float>(dir + "/codebook.bin", (size_t)S * E * 2);
    auto origins = read_bin<float>(dir + "/origins.bin", (size_t)Q * P * S * 2);
    auto thresholds = read_bin<float>(dir + "/thresholds.bin", (size_t)F * S);
    std::vector<int> nprobes;
    { std::stringstream ss(nprobe_arg); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) nprobes.push_back(std::min(atoi(t.c_str()), P)); }
    const float R2 = R * R;
    const int max_hits = E;

    // ---------------------------------------------------------------- OptiX setup
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext ctx;
    OptixDeviceContextOptions copts = {}; copts.logCallbackFunction = log_cb; copts.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(0, &copts, &ctx));

    std::string ptx;
    { std::ifstream f(ptx_path, std::ios::binary); if (!f) { fprintf(stderr, "cannot open %s\n", ptx_path.c_str()); return 1; }
      std::stringstream b; b << f.rdbuf(); ptx = b.str(); }
    const unsigned int build_flags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    OptixModuleCompileOptions mco = {}; mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    OptixPipelineCompileOptions pco = {};
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pco.numPayloadValues = 2;
    pco.numAttributeValues = 2;
    pco.pipelineLaunchParamsVariableName = "params";
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
    OptixModule mod, is_mod;
    OPTIX_CHECK(optixModuleCreate(ctx, &mco, &pco, ptx.data(), ptx.size(), nullptr, nullptr, &mod));
    OptixBuiltinISOptions bio = {};
    bio.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    bio.usesMotionBlur = 0;
    bio.buildFlags = build_flags;
    OPTIX_CHECK(optixBuiltinISModuleGet(ctx, &mco, &pco, &bio, &is_mod));

    OptixProgramGroupOptions gopts = {};
    OptixProgramGroup rg, ms, hg;
    OptixProgramGroupDesc d = {};
    d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module = mod; d.raygen.entryFunctionName = "__raygen__juno";
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &d, 1, &gopts, nullptr, nullptr, &rg));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module = mod; d.miss.entryFunctionName = "__miss__juno";
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &d, 1, &gopts, nullptr, nullptr, &ms));
    d = {}; d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    d.hitgroup.moduleAH = mod; d.hitgroup.entryFunctionNameAH = "__anyhit__juno";
    d.hitgroup.moduleIS = is_mod; d.hitgroup.entryFunctionNameIS = nullptr;
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &d, 1, &gopts, nullptr, nullptr, &hg));
    OptixProgramGroup groups[] = {rg, ms, hg};
    OptixPipelineLinkOptions plo = {}; plo.maxTraceDepth = 1;
    OptixPipeline pipe;
    OPTIX_CHECK(optixPipelineCreate(ctx, &pco, &plo, groups, 3, nullptr, nullptr, &pipe));
    OptixStackSizes ssz = {};
    for (auto g : groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &ssz, pipe));
    unsigned int dc_t, dc_s, cont;
    OPTIX_CHECK(optixUtilComputeStackSizes(&ssz, 1, 0, 0, &dc_t, &dc_s, &cont));
    OPTIX_CHECK(optixPipelineSetStackSize(pipe, dc_t, dc_s, cont, 1));

    Rec<Empty> r_rg, r_ms, r_hg;
    OPTIX_CHECK(optixSbtRecordPackHeader(rg, &r_rg));
    OPTIX_CHECK(optixSbtRecordPackHeader(ms, &r_ms));
    OPTIX_CHECK(optixSbtRecordPackHeader(hg, &r_hg));
    void *d_rg, *d_ms, *d_hg;
    CUDA_CHECK(cudaMalloc(&d_rg, sizeof(r_rg))); CUDA_CHECK(cudaMemcpy(d_rg, &r_rg, sizeof(r_rg), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ms, sizeof(r_ms))); CUDA_CHECK(cudaMemcpy(d_ms, &r_ms, sizeof(r_ms), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_hg, sizeof(r_hg))); CUDA_CHECK(cudaMemcpy(d_hg, &r_hg, sizeof(r_hg), cudaMemcpyHostToDevice));
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = (CUdeviceptr)d_rg;
    sbt.missRecordBase = (CUdeviceptr)d_ms; sbt.missRecordStrideInBytes = sizeof(r_ms); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = (CUdeviceptr)d_hg; sbt.hitgroupRecordStrideInBytes = sizeof(r_hg); sbt.hitgroupRecordCount = 1;

    // ---------------------------------------------------------------- scene: one sphere per entry
    std::vector<float3> centers((size_t)S * E);
    for (int s = 0; s < S; ++s)
        for (int e = 0; e < E; ++e) {
            const float* c = &codebook[((size_t)s * E + e) * 2];
            centers[(size_t)s * E + e] = make_float3(c[0], c[1], 2.0f * s + 1.0f);
        }
    float3* d_centers; float* d_radius;
    CUDA_CHECK(cudaMalloc(&d_centers, centers.size() * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_centers, centers.data(), centers.size() * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_radius, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_radius, &R, sizeof(float), cudaMemcpyHostToDevice));
    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    CUdeviceptr vb = (CUdeviceptr)d_centers, rb = (CUdeviceptr)d_radius;
    unsigned int gflags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    bi.sphereArray.vertexBuffers = &vb;
    bi.sphereArray.numVertices = (unsigned int)centers.size();
    bi.sphereArray.radiusBuffers = &rb;
    bi.sphereArray.singleRadius = 1;
    bi.sphereArray.flags = &gflags;
    bi.sphereArray.numSbtRecords = 1;
    OptixAccelBuildOptions ao = {}; ao.buildFlags = build_flags; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes bs;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
    void *d_tmp, *d_gas;
    CUDA_CHECK(cudaMalloc(&d_tmp, bs.tempSizeInBytes)); CUDA_CHECK(cudaMalloc(&d_gas, bs.outputSizeInBytes));
    OptixTraversableHandle gas;
    OPTIX_CHECK(optixAccelBuild(ctx, 0, &ao, &bi, 1, (CUdeviceptr)d_tmp, bs.tempSizeInBytes,
                                (CUdeviceptr)d_gas, bs.outputSizeInBytes, &gas, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaFree(d_tmp));

    float2* d_entries;
    CUDA_CHECK(cudaMalloc(&d_entries, codebook.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_entries, codebook.data(), codebook.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------------------------------------------- buffers (batches of <= ~1M rays)
    const size_t ray_budget = 1 << 20;
    size_t max_rays = 0;
    for (int p : nprobes) {
        int qb = std::max(1, (int)(ray_budget / ((size_t)p * S)));
        max_rays = std::max(max_rays, (size_t)std::min(qb, Q) * p * S);
    }
    float2 *d_orig; float *d_tmax, *d_r2, *d_dist_rt, *d_dist_cu; int *d_cnt_rt, *d_cnt_cu, *d_eid_rt, *d_eid_cu;
    CUDA_CHECK(cudaMalloc(&d_orig, max_rays * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_tmax, max_rays * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2, max_rays * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cnt_rt, max_rays * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cnt_cu, max_rays * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eid_rt, max_rays * max_hits * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eid_cu, max_rays * max_hits * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist_rt, max_rays * max_hits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist_cu, max_rays * max_hits * sizeof(float)));   // also the dense LUT
    JunoParams* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(JunoParams)));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    FILE* out = out_path.empty() ? nullptr : fopen(out_path.c_str(), "a");
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "GPU %s | S=%d E=%d Q=%d max nprobe=%d R=%.3f | fractions %d\n", prop.name, S, E, Q, P, R, F);

    for (int fi = 0; fi < F; ++fi) {
        for (int p : nprobes) {
            const int qb = std::min(Q, std::max(1, (int)(ray_budget / ((size_t)p * S))));
            double ms_rt = 0, ms_cu = 0, ms_dense = 0, hits = 0, rays_total = 0;
            long long checked = 0, matched = 0; double dist_err = 0; long long dist_n = 0;
            for (int q0 = 0; q0 < Q; q0 += qb) {
                const int nq = std::min(qb, Q - q0), rps = nq * p;
                const size_t rays = (size_t)S * rps;
                std::vector<float2> h_orig(rays); std::vector<float> h_tmax(rays), h_r2(rays);
                for (int s = 0; s < S; ++s) {
                    const float r = std::min(thresholds[(size_t)fi * S + s], 0.999f * R);
                    const float tm = 1.0f - sqrtf(R2 - r * r);
                    const float a = 1.0f - tm, reff2 = R2 - a * a;        // the radius the ray actually tests
                    for (int q = 0; q < nq; ++q)
                        for (int k = 0; k < p; ++k) {
                            const float* o = &origins[((((size_t)(q0 + q) * P) + k) * S + s) * 2];
                            const size_t i = (size_t)s * rps + (size_t)q * p + k;
                            h_orig[i] = make_float2(o[0], o[1]); h_tmax[i] = tm; h_r2[i] = reff2;
                        }
                }
                CUDA_CHECK(cudaMemcpy(d_orig, h_orig.data(), rays * sizeof(float2), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_tmax, h_tmax.data(), rays * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_r2, h_r2.data(), rays * sizeof(float), cudaMemcpyHostToDevice));
                JunoParams hp = {gas, d_orig, d_tmax, rps, E, max_hits, R2, d_cnt_rt, d_eid_rt, d_dist_rt};
                CUDA_CHECK(cudaMemcpy(d_params, &hp, sizeof(hp), cudaMemcpyHostToDevice));
                dim3 blk(256), grd((rps + 255) / 256, S);
                const size_t shm = (size_t)E * sizeof(float2);
                std::vector<float> trt, tcu, tde;
                for (int r = 0; r < reps + 1; ++r) {          // first repetition is warm-up
                    float ms;
                    CUDA_CHECK(cudaEventRecord(t0));
                    OPTIX_CHECK(optixLaunch(pipe, 0, (CUdeviceptr)d_params, sizeof(JunoParams), &sbt, (unsigned)rays, 1, 1));
                    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
                    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1)); if (r) trt.push_back(ms);
                    CUDA_CHECK(cudaEventRecord(t0));
                    lut_select_cuda<<<grd, blk, shm>>>(d_orig, d_r2, d_entries, rps, E, max_hits, d_cnt_cu, d_eid_cu, d_dist_cu);
                    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1)); if (r) tcu.push_back(ms);
                }
                // validate the selective outputs (first batch), before the dense LUT overwrites d_dist_cu
                if (q0 == 0) {
                    // check 20,000 rays spread over every subspace of the batch
                    const size_t nchk = std::min(rays, (size_t)20000);
                    const size_t stride = std::max<size_t>(1, rays / nchk);
                    std::vector<int> cnt_rt_all(rays), cnt_cu_all(rays);
                    CUDA_CHECK(cudaMemcpy(cnt_rt_all.data(), d_cnt_rt, rays * sizeof(int), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(cnt_cu_all.data(), d_cnt_cu, rays * sizeof(int), cudaMemcpyDeviceToHost));
                    for (size_t c = 0; c < nchk && c * stride < rays; ++c) {
                        const size_t i = c * stride;
                        std::vector<int> a(max_hits), b(max_hits); std::vector<float> da(max_hits), db(max_hits);
                        const int na = std::min(cnt_rt_all[i], max_hits), nb = std::min(cnt_cu_all[i], max_hits);
                        CUDA_CHECK(cudaMemcpy(a.data(), d_eid_rt + i * max_hits, na * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(b.data(), d_eid_cu + i * max_hits, nb * sizeof(int), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(da.data(), d_dist_rt + i * max_hits, na * sizeof(float), cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaMemcpy(db.data(), d_dist_cu + i * max_hits, nb * sizeof(float), cudaMemcpyDeviceToHost));
                        std::vector<std::pair<int, float>> pa, pb;
                        for (int k = 0; k < na; ++k) pa.push_back({a[k], da[k]});
                        for (int k = 0; k < nb; ++k) pb.push_back({b[k], db[k]});
                        std::sort(pa.begin(), pa.end()); std::sort(pb.begin(), pb.end());
                        bool same = pa.size() == pb.size();
                        for (size_t k = 0; same && k < pa.size(); ++k) {
                            same = pa[k].first == pb[k].first;
                            dist_err += fabs(pa[k].second - pb[k].second); ++dist_n;
                        }
                        ++checked; matched += same;
                    }
                }
                for (int r = 0; r < reps + 1; ++r) {
                    float ms;
                    CUDA_CHECK(cudaEventRecord(t0));
                    lut_dense_cuda<<<grd, blk, shm>>>(d_orig, d_entries, rps, E, d_dist_cu);
                    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1)); if (r) tde.push_back(ms);
                }
                std::vector<int> cnt(rays);
                CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt_cu, rays * sizeof(int), cudaMemcpyDeviceToHost));
                for (int c : cnt) hits += c;
                ms_rt += median(trt); ms_cu += median(tcu); ms_dense += median(tde); rays_total += rays;
            }
            char line[1024];
            snprintf(line, sizeof(line),
                     "{\"gpu\": \"%s\", \"frac\": %.3f, \"nprobe\": %d, \"queries\": %d, \"rays\": %.0f, "
                     "\"mean_hits_per_ray\": %.2f, \"rt_ms\": %.4f, \"cuda_ms\": %.4f, \"dense_lut_ms\": %.4f, "
                     "\"rt_over_cuda\": %.3f, \"set_match\": %.5f, \"checked_rays\": %lld, \"mean_dist_abs_err\": %.3g}",
                     prop.name, fracs[fi], p, Q, rays_total, hits / rays_total, ms_rt, ms_cu, ms_dense,
                     ms_rt / ms_cu, checked ? (double)matched / checked : 0.0, checked, dist_n ? dist_err / dist_n : 0.0);
            printf("%s\n", line); fflush(stdout);
            if (out) { fprintf(out, "%s\n", line); fflush(out); }
        }
    }
    if (out) fclose(out);
    return 0;
}
