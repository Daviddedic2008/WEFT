#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include "optix_corr_shared_3d.h"

#define CUDA_CHECK(call) do { cudaError_t e=(call); TORCH_CHECK(e==cudaSuccess, "CUDA: ", cudaGetErrorString(e)); } while(0)
#define OPTIX_CHECK(call) do { OptixResult r=(call); TORCH_CHECK(r==OPTIX_SUCCESS, "OptiX: ", optixGetErrorString(r)); } while(0)

static void ctx_log(unsigned int lvl, const char* tag, const char* msg, void*) {}

static bool loadPTX(const std::string& fn, std::vector<char>& buf) {
    FILE* f = fopen(fn.c_str(), "rb"); if (!f) return false;
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    buf.resize(sz+1);
    if (fread(buf.data(),1,sz,f) != (size_t)sz){ fclose(f); return false; }
    buf[sz]='\0'; fclose(f); return true;
}

struct RG {}; struct MS {}; struct HG {};
template<typename T> struct SbtRecord { alignas(OPTIX_SBT_RECORD_HEADER_SIZE) char header[OPTIX_SBT_RECORD_HEADER_SIZE]; T data; };
typedef SbtRecord<RG> RayGenSbt;
typedef SbtRecord<MS> MissSbt;
typedef SbtRecord<HG> HitSbt;

// ragrt_fused_kernel.cu
extern "C" void launch_mask_failing_candidates(const int* d_cand_pids, int n, const uint32_t* d_doc_predicates,
                                               uint32_t query_mask, float* d_scores, cudaStream_t stream);
extern "C" void launch_bruteforce_stage1(const float* d_q_points, const float* d_codewords, int total_rays,
                                         int E, int k, int max_hits, int* d_out_c, float* d_out_v,
                                         int* d_out_n, cudaStream_t stream);
extern "C" void launch_threshold_stage1(const float* d_q_points, const float* d_codewords, const float* d_tau,
                                        int total_rays, int E, int max_hits, int* d_out_c, float* d_out_v,
                                        int* d_out_n, cudaStream_t stream);
extern "C" void launch_all_codewords_stage1(const float* d_q_points, const float* d_codewords, int total_rays, int E,
                                            int max_hits, int* d_out_c, float* d_out_v, int* d_out_n, cudaStream_t stream);
extern "C" void ragrt_stage2(const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
                             const int* d_topc, const float* d_scores,
                             int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
                             const int64_t* d_csr_row_ptrs, const void* d_csr_col_eids, int eid_bytes,
                             const int64_t* d_csr_block_sums, const uint16_t* d_csr_lengths,
                             const int* d_prim_eid, int E,
                             int buf_idx, cudaStream_t stream);
extern "C" void ragrt_stage3_dense(const uint8_t* d_map_packed_24, float* d_passage_scores, int num_passages,
                                   const uint32_t* d_doc_predicates, uint32_t query_mask, int ntok,
                                   int buf_idx, cudaStream_t stream);
extern "C" long long ragrt_stage3_sparse(const uint8_t* d_map_packed_24, int num_passages,
                                         const uint32_t* d_doc_predicates, uint32_t query_mask, long long min_len,
                                         int buf_idx, cudaStream_t stream, int** d_pids, float** d_sums,
                                         long long* n_valid);
extern "C" long long ragrt_stage3_pidsort(const uint8_t* d_map_packed_24, int num_passages,
                                          const uint32_t* d_doc_predicates, uint32_t query_mask, long long min_len,
                                          int buf_idx, cudaStream_t stream, int** d_pids, float** d_sums);
extern "C" long long ragrt_last_total_postings(int buf_idx);
extern "C" long long ragrt_stage3_sparse_static(const uint8_t* d_map_packed_24, int num_passages, const uint32_t* d_doc_predicates,
                                                const uint32_t* d_mask, long long cap, int buf_idx, cudaStream_t stream,
                                                int** d_pids, float** d_sums);
extern "C" long long ragrt_stage3_pidsort_static(const uint8_t* d_map_packed_24, int num_passages, const uint32_t* d_doc_predicates,
                                                 const uint32_t* d_mask, long long cap, int buf_idx, cudaStream_t stream,
                                                 int** d_pids, float** d_sums);
extern "C" long long ragrt_reserve_sparse(long long cap, int buf_idx, cudaStream_t stream);
extern "C" long long ragrt_sparse_capacity(int buf_idx);
extern "C" unsigned long long* ragrt_total_postings_ptr(int buf_idx);
extern "C" void launch_mask_failing_candidates_dev(const int* d_cand_pids, int n, const uint32_t* d_doc_predicates,
                                                   const uint32_t* d_mask, float* d_scores, cudaStream_t stream);
extern "C" void ragrt_set_drop_stats(int enabled);
extern "C" void ragrt_reset_drop_stats();
extern "C" void ragrt_drop_stats(unsigned long long* out);
// tile_maxsim_fused_decomp.cu
extern "C" void launch_tile_maxsim_fused_decomp(
    const void* d_Q, const int* d_pids, const int64_t* d_doc_offsets, const int* d_doc_lens,
    int num_cands, const int* d_codes, const uint8_t* d_residuals, const void* d_centroids,
    const void* d_bucket_weights, const uint8_t* d_reversed_bit_map, const uint8_t* d_decomp_table,
    float* d_out_scores, cudaStream_t stream);
extern "C" void ragrt_set_rerank_mode(int mode);
extern "C" int  ragrt_get_rerank_mode();
extern "C" void ragrt_set_codeword_weight(float w);
extern "C" float ragrt_get_codeword_weight();
extern "C" void ragrt_set_stage3_hash(int on);
extern "C" void ragrt_set_stage3_mode(int mode);
extern "C" int ragrt_get_stage3_mode();
extern "C" int ragrt_get_stage3_hash();
extern "C" void ragrt_set_compact_scores(int on);
extern "C" int ragrt_get_compact_scores();
extern "C" void ragrt_set_residual_scoring(const uint32_t* tok_ids, const uint8_t* residuals, const float* lut);
extern "C" int ragrt_residual_scoring_on();

static const char* DROP_STAT_NAMES[] = {
    "launches", "rays", "rays_overflow", "hits_lost_to_cap", "combos", "combos_found",
    "min_score_drops", "tasks", "tasks_dropped", "hits", "rays_short", "postings"
};
static const int NUM_DROP_STATS = 12;

static const int MAX_RAYS = 32 * NUM_3D_SUBSPACES;   // 32 query tokens x 32 subspaces
static const int FAN_MAX_HITS = 128;                  // legacy buffer width for the fan geometry

struct Gas {
    OptixTraversableHandle handle = 0;
    void* out = nullptr; void* verts = nullptr; void* idx = nullptr;
};

class CorrIndex3D {
public:
    OptixDeviceContext ctx = nullptr;
    OptixModule mod = nullptr;
    OptixPipeline pipe = nullptr;
    OptixProgramGroup rg_grp=nullptr, ms_grp=nullptr, hg_grp=nullptr;
    OptixShaderBindingTable sbt = {};
    void *d_rg=nullptr,*d_ms=nullptr,*d_hg=nullptr;

    // Scene
    int E = 0;
    std::vector<float> h_cb;                    // [32 * E * 3]
    float3* d_centroid_xyz = nullptr;           // [32 * E]
    float3* d_sorted_xyz = nullptr;             // [32 * E], each subspace sorted by norm (primitive order)
    torch::Tensor t_prim_eid;                   // same map as d_prim_eid, for stage1_hits
    int* d_prim_eid = nullptr;                  // [32 * E] norm-sorted index -> eid
    std::vector<int> h_prim_eid;
    std::vector<Gas> fan_gas;                   // [32]
    std::vector<std::vector<Gas>> level_gas;    // [L][32]
    std::vector<std::array<float, NUM_3D_SUBSPACES>> level_tau;
    std::vector<int> level_k;
    std::vector<std::array<int, NUM_3D_SUBSPACES>> level_prims;
    float* d_level_tau = nullptr;               // [L * 32]

    int geometry = GEOM_FAN;
    int stage1_mode = 0;    // 0: OptiX RT cores, 1: CUDA cores (fan: exact top-k, polar: threshold scan),
                            // 2: every codeword (ablation: codebook filter off)
    int stage3_sparse = 1;

    // Buffers
    torch::Tensor buf_out_c[2], buf_out_v[2], buf_out_n[2];
    torch::Tensor d_passage_scores[2], d_final_scores[2], d_candidate_pids[2], d_cand_approx[2];
    CorrParams3D* d_p_single = nullptr;
    int max_allocated_cands = 0;

    cudaStream_t stream_cand = nullptr;
    cudaStream_t stream_score = nullptr;
    cudaEvent_t event_cand_ready[2];
    cudaEvent_t event_tms_done[2];

    bool is_bound = false;
    torch::Tensor b_csr_row_ptrs, b_csr_col_eids, b_csr_block_sums, b_csr_lengths, b_map_packed;
    torch::Tensor b_doc_offsets, b_doc_lens, b_codes, b_residuals;
    torch::Tensor b_centroids_128, b_bucket_weights, b_reversed_bit_map, b_decomp_table;
    torch::Tensor b_token_ids, b_res_lut;   // token-level plain-IVF index + per-query residual LUT
    torch::Tensor b_doc_predicates;
    int64_t b_num_passages = 0;
    int b_num_centroids = 0;
    int b_eid_bytes = 1;

    CorrIndex3D(std::string ptx_path) {
        CUDA_CHECK(cudaFree(0)); OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions copts={}; copts.logCallbackFunction=ctx_log; copts.logCallbackLevel=4;
        CUcontext cu=nullptr; cuCtxGetCurrent(&cu); OPTIX_CHECK(optixDeviceContextCreate(cu,&copts,&ctx));

        std::vector<char> ptx; TORCH_CHECK(loadPTX(ptx_path, ptx), "failed to load candidate PTX ", ptx_path);
        OptixModuleCompileOptions mco={}; mco.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        OptixPipelineCompileOptions pco={}; pco.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pco.numPayloadValues=3;
        pco.numAttributeValues=2;
        pco.pipelineLaunchParamsVariableName="params";
        pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        OPTIX_CHECK(optixModuleCreate(ctx,&mco,&pco,ptx.data(),ptx.size(),nullptr,nullptr,&mod));

        OptixProgramGroupOptions gopts={}; OptixProgramGroupDesc d={};
        d.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN; d.raygen.module=mod; d.raygen.entryFunctionName="__raygen__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&rg_grp));
        d={}; d.kind=OPTIX_PROGRAM_GROUP_KIND_MISS; d.miss.module=mod; d.miss.entryFunctionName="__miss__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&ms_grp));
        d={}; d.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP; d.hitgroup.moduleAH=mod; d.hitgroup.entryFunctionNameAH="__anyhit__corr_3d";
        d.hitgroup.moduleCH=mod; d.hitgroup.entryFunctionNameCH="__closesthit__corr_3d";
        OPTIX_CHECK(optixProgramGroupCreate(ctx,&d,1,&gopts,nullptr,nullptr,&hg_grp));

        OptixPipelineLinkOptions plo={}; plo.maxTraceDepth=1; OptixProgramGroup groups[]={rg_grp,ms_grp,hg_grp};
        OPTIX_CHECK(optixPipelineCreate(ctx,&pco,&plo,groups,3,nullptr,nullptr,&pipe));
        OptixStackSizes ss={}; for(auto g:groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g,&ss,pipe));
        uint32_t dc_t=0,dc_s=0,cont=0; OPTIX_CHECK(optixUtilComputeStackSizes(&ss,1,0,0,&dc_t,&dc_s,&cont));
        OPTIX_CHECK(optixPipelineSetStackSize(pipe,dc_t,dc_s,cont,1));

        RayGenSbt rg={}; MissSbt ms={}; HitSbt hg={};
        OPTIX_CHECK(optixSbtRecordPackHeader(rg_grp,&rg)); OPTIX_CHECK(optixSbtRecordPackHeader(ms_grp,&ms)); OPTIX_CHECK(optixSbtRecordPackHeader(hg_grp,&hg));
        CUDA_CHECK(cudaMalloc(&d_rg,sizeof(rg))); CUDA_CHECK(cudaMemcpy(d_rg,&rg,sizeof(rg),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_ms,sizeof(ms))); CUDA_CHECK(cudaMemcpy(d_ms,&ms,sizeof(ms),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_hg,sizeof(hg))); CUDA_CHECK(cudaMemcpy(d_hg,&hg,sizeof(hg),cudaMemcpyHostToDevice));
        sbt.raygenRecord=(CUdeviceptr)d_rg; sbt.missRecordBase=(CUdeviceptr)d_ms; sbt.missRecordStrideInBytes=sizeof(ms); sbt.missRecordCount=1;
        sbt.hitgroupRecordBase=(CUdeviceptr)d_hg; sbt.hitgroupRecordStrideInBytes=sizeof(hg); sbt.hitgroupRecordCount=1;

        CUDA_CHECK(cudaMalloc(&d_p_single, sizeof(CorrParams3D)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_cand, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_score, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_cand_ready[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&event_tms_done[b], cudaEventDisableTiming));
        }
        auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        for (int b = 0; b < 2; ++b) {
            buf_out_c[b] = torch::zeros({MAX_RAYS, MAX_HITS_CAP}, oi);
            buf_out_v[b] = torch::zeros({MAX_RAYS, MAX_HITS_CAP}, of);
            buf_out_n[b] = torch::zeros({MAX_RAYS}, oi);
        }
    }

    ~CorrIndex3D() {
        free_scene();
        if (d_p_single) cudaFree(d_p_single);
        if (stream_cand) cudaStreamDestroy(stream_cand);
        if (stream_score) cudaStreamDestroy(stream_score);
        for (int b = 0; b < 2; ++b) {
            if (event_cand_ready[b]) cudaEventDestroy(event_cand_ready[b]);
            if (event_tms_done[b]) cudaEventDestroy(event_tms_done[b]);
        }
    }

    // ------------------------------------------------------------------ scene
    static void free_gas(Gas& g) {
        if (g.out) cudaFree(g.out);
        if (g.verts) cudaFree(g.verts);
        if (g.idx) cudaFree(g.idx);
        g = Gas();
    }

    void free_levels() {
        for (auto& lv : level_gas) for (auto& g : lv) free_gas(g);
        level_gas.clear(); level_tau.clear(); level_k.clear(); level_prims.clear();
        if (d_level_tau) cudaFree(d_level_tau);
        d_level_tau = nullptr;
    }

    void free_scene() {
        for (auto& g : fan_gas) free_gas(g);
        fan_gas.clear();
        free_levels();
        if (d_centroid_xyz) cudaFree(d_centroid_xyz);
        if (d_prim_eid) cudaFree(d_prim_eid);
        if (d_sorted_xyz) cudaFree(d_sorted_xyz);
        d_centroid_xyz = nullptr; d_prim_eid = nullptr; d_sorted_xyz = nullptr;
    }

    Gas build_gas(const std::vector<float3>& verts, const std::vector<uint3>& indices, unsigned int geom_flags) {
        Gas g;
        CUDA_CHECK(cudaMalloc(&g.verts, verts.size() * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&g.idx, indices.size() * sizeof(uint3)));
        CUDA_CHECK(cudaMemcpy(g.verts, verts.data(), verts.size() * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.idx, indices.data(), indices.size() * sizeof(uint3), cudaMemcpyHostToDevice));

        OptixBuildInput bi = {}; bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        CUdeviceptr dp_v = (CUdeviceptr)g.verts;
        bi.triangleArray.vertexBuffers = &dp_v;
        bi.triangleArray.numVertices = (unsigned)verts.size();
        bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.indexBuffer = (CUdeviceptr)g.idx;
        bi.triangleArray.numIndexTriplets = (unsigned)indices.size();
        bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        unsigned int fl = geom_flags;
        bi.triangleArray.flags = &fl; bi.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions ao = {}; ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE; ao.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes bs; OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &ao, &bi, 1, &bs));
        void* tmp = nullptr;
        CUDA_CHECK(cudaMalloc(&tmp, bs.tempSizeInBytes)); CUDA_CHECK(cudaMalloc(&g.out, bs.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(ctx, nullptr, &ao, &bi, 1, (CUdeviceptr)tmp, bs.tempSizeInBytes,
                                    (CUdeviceptr)g.out, bs.outputSizeInBytes, &g.handle, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaFree(tmp));
        return g;
    }

    static void plane_basis(float3 n, float3& u, float3& v) {
        u = (fabsf(n.x) > 0.1f) ? make_float3(-n.y, n.x, 0.0f) : make_float3(0.0f, -n.z, n.y);
        float un = sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
        u.x /= un; u.y /= un; u.z /= un;
        v = make_float3(n.y*u.z - n.z*u.y, n.z*u.x - n.x*u.z, n.x*u.y - n.y*u.x);
    }

    // Codebooks [32, E, 3]. Builds the legacy fan scene and the norm-sorted primitive map
    // used by the polar levels (set_polar_levels). E is taken from the codebooks.
    void build(torch::Tensor codebooks, double base_radius = 0.95, int64_t num_tris = 4) {
        TORCH_CHECK(num_tris == 4, "build() requires num_tris == 4; got ", num_tris);
        TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) == NUM_3D_SUBSPACES && codebooks.size(2) == 3,
                    "codebooks must be [32, E, 3]");
        free_scene();
        auto cb = codebooks.to(torch::kCPU).to(torch::kFloat32).contiguous();
        int S = NUM_3D_SUBSPACES;
        E = (int)cb.size(1);
        TORCH_CHECK(E >= 1 && E <= MAX_ENTRIES_PER_SUBSPACE, "E=", E, " out of range");
        h_cb.assign(cb.data_ptr<float>(), cb.data_ptr<float>() + (size_t)S * E * 3);

        std::vector<float3> h_xyz(S * E);
        for (int i = 0; i < S * E; ++i) h_xyz[i] = make_float3(h_cb[3*i], h_cb[3*i+1], h_cb[3*i+2]);
        CUDA_CHECK(cudaMalloc(&d_centroid_xyz, h_xyz.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_centroid_xyz, h_xyz.data(), h_xyz.size() * sizeof(float3), cudaMemcpyHostToDevice));

        h_prim_eid.resize(S * E);
        for (int s = 0; s < S; ++s) {
            int* o = h_prim_eid.data() + s * E;
            std::iota(o, o + E, 0);
            std::stable_sort(o, o + E, [&](int a, int b) { return norm(s, a) > norm(s, b); });
        }
        CUDA_CHECK(cudaMalloc(&d_prim_eid, h_prim_eid.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prim_eid, h_prim_eid.data(), h_prim_eid.size() * sizeof(int), cudaMemcpyHostToDevice));
        t_prim_eid = torch::from_blob(h_prim_eid.data(), {(int64_t)h_prim_eid.size()}, torch::kInt32).to(torch::kInt64).to(torch::kCUDA);
        std::vector<float3> h_sorted(S * E);
        for (int s = 0; s < S; ++s)
            for (int i = 0; i < E; ++i) h_sorted[s * E + i] = h_xyz[s * E + h_prim_eid[s * E + i]];
        CUDA_CHECK(cudaMalloc(&d_sorted_xyz, h_sorted.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(d_sorted_xyz, h_sorted.data(), h_sorted.size() * sizeof(float3), cudaMemcpyHostToDevice));

        constexpr float PI = 3.14159265358979323846f;
        fan_gas.resize(S);
        for (int s = 0; s < S; ++s) {
            std::vector<float3> verts; std::vector<uint3> indices;
            verts.reserve(E * 5); indices.reserve(E * 4);
            for (int e = 0; e < E; ++e) {
                float3 c = h_xyz[s * E + e];
                float nrm = sqrtf(c.x*c.x + c.y*c.y + c.z*c.z);
                float radius = sqrtf((float)(base_radius * base_radius) + nrm * nrm);
                float3 n = (nrm > 1e-5f) ? make_float3(c.x/nrm, c.y/nrm, c.z/nrm) : make_float3(0, 0, 1);
                float3 u, v; plane_basis(n, u, v);
                uint32_t b = (uint32_t)verts.size();
                int P = (int)num_tris;
                verts.push_back(c);
                for (int k = 0; k < P; ++k) {
                    float th = k * (2.0f * PI / (float)P);
                    verts.push_back(make_float3(c.x + radius * (cosf(th)*u.x + sinf(th)*v.x),
                                                c.y + radius * (cosf(th)*u.y + sinf(th)*v.y),
                                                c.z + radius * (cosf(th)*u.z + sinf(th)*v.z)));
                }
                for (int k = 0; k < P; ++k) indices.push_back(make_uint3(b, b + 1 + k, b + 1 + ((k + 1) % P)));
            }
            fan_gas[s] = build_gas(verts, indices, OPTIX_GEOMETRY_FLAG_NONE);
        }
    }

    float norm(int s, int e) const {
        const float* c = &h_cb[((size_t)s * E + e) * 3];
        return sqrtf(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
    }

    // Exact-threshold scenes. taus [L, 32]: level l returns, for every ray of subspace s,
    // exactly the codewords with d.c >= taus[l][s] (d = unit query slice). ks[l] is the
    // k_eids the level was calibrated for; a search with k_eids uses the level with the
    // smallest ks[l] >= k_eids (the lowest threshold if none).
    // Codeword c -> one triangle in the plane x.c = 1, centred on the foot point c/|c|^2,
    // circumscribing the disc of radius sqrt(1/tau^2 - 1/|c|^2): every point of the plane
    // with |x| <= 1/tau lies in that disc, and tmax = 1/tau cuts everything else.
    // Codewords with |c| <= tau can never reach tau and are left out.
    void set_polar_levels(torch::Tensor taus, std::vector<int64_t> ks) {
        TORCH_CHECK(E > 0, "build() first");
        auto t = taus.to(torch::kCPU).to(torch::kFloat32).contiguous();
        TORCH_CHECK(t.dim() == 2 && t.size(1) == NUM_3D_SUBSPACES && t.size(0) == (int64_t)ks.size() && ks.size() >= 1,
                    "taus must be [L, 32] with L == len(ks)");
        free_levels();
        int L = (int)ks.size();
        const float* tp = t.data_ptr<float>();
        level_gas.resize(L);
        for (int l = 0; l < L; ++l) {
            std::array<float, NUM_3D_SUBSPACES> tau; std::array<int, NUM_3D_SUBSPACES> nprim;
            level_gas[l].resize(NUM_3D_SUBSPACES);
            for (int s = 0; s < NUM_3D_SUBSPACES; ++s) {
                float ts = tp[l * NUM_3D_SUBSPACES + s];
                TORCH_CHECK(ts > 0.0f && std::isfinite(ts), "tau must be positive, got ", ts, " (level ", l, ", subspace ", s, ")");
                tau[s] = ts;
                std::vector<float3> verts; std::vector<uint3> indices;
                const int* order = h_prim_eid.data() + s * E;
                for (int i = 0; i < E; ++i) {
                    int e = order[i];
                    float cn = norm(s, e);
                    if (!(cn > ts * 1.0001f)) break;              // sorted by norm: the rest are smaller
                    const float* c = &h_cb[((size_t)s * E + e) * 3];
                    float3 n = make_float3(c[0]/cn, c[1]/cn, c[2]/cn);
                    float3 f = make_float3(n.x / cn, n.y / cn, n.z / cn);   // foot point c / |c|^2
                    float r = sqrtf(std::max(1.0f/(ts*ts) - 1.0f/(cn*cn), 0.0f)) * 1.01f + 1e-6f;
                    float3 u, v; plane_basis(n, u, v);
                    uint32_t b = (uint32_t)verts.size();
                    for (int k = 0; k < 3; ++k) {                 // circumradius 2r -> inradius r
                        float th = 1.5707963f + k * 2.0943951f;
                        float cx = 2.0f * r * cosf(th), cy = 2.0f * r * sinf(th);
                        verts.push_back(make_float3(f.x + cx*u.x + cy*v.x, f.y + cx*u.y + cy*v.y, f.z + cx*u.z + cy*v.z));
                    }
                    indices.push_back(make_uint3(b, b + 1, b + 2));
                }
                nprim[s] = (int)indices.size();
                if (indices.empty()) {   // nothing can reach tau: one degenerate triangle (never hit)
                    verts.assign(3, make_float3(0.0f, 0.0f, 1e6f));
                    indices.push_back(make_uint3(0, 1, 2));
                }
                level_gas[l][s] = build_gas(verts, indices, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
            }
            level_tau.push_back(tau);
            level_prims.push_back(nprim);
            level_k.push_back((int)ks[l]);
        }
        std::vector<float> flat;
        for (auto& a : level_tau) flat.insert(flat.end(), a.begin(), a.end());
        CUDA_CHECK(cudaMalloc(&d_level_tau, flat.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_level_tau, flat.data(), flat.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    int level_for(int k_eids) const {
        TORCH_CHECK(!level_k.empty(), "polar geometry needs set_polar_levels() (calibrated thresholds) first");
        int best = -1;
        for (int l = 0; l < (int)level_k.size(); ++l)
            if (level_k[l] >= k_eids && (best < 0 || level_k[l] < level_k[best])) best = l;
        if (best < 0) {
            best = 0;
            for (int l = 1; l < (int)level_k.size(); ++l) if (level_k[l] > level_k[best]) best = l;
        }
        return best;
    }

    void set_geometry(int g) {
        TORCH_CHECK(g == GEOM_FAN || g == GEOM_POLAR || g == GEOM_POLAR_ITER,
                    "geometry must be 0 (fan), 1 (polar) or 2 (polar_iter)");
        geometry = g;
    }
    bool polar_scene() const { return geometry == GEOM_POLAR || geometry == GEOM_POLAR_ITER; }
    // Stage-1 hit ids are norm-sorted primitive indices (mapped to eids in Stage 2) when they come
    // from the polar scene or its threshold-scan twin; the fan scene and the exact top-k scan give eids.
    bool hits_are_prims() const {
        if (stage1_mode == 2) return false;
        return geometry == GEOM_POLAR || (geometry == GEOM_POLAR_ITER && stage1_mode == 0);
    }
    void set_stage1_mode(int mode) {
        TORCH_CHECK(mode >= 0 && mode <= 2, "stage1 mode must be 0 (RT), 1 (CUDA cores) or 2 (all codewords)");
        stage1_mode = mode;
    }
    void set_stage3_sparse(bool on) { stage3_sparse = on ? 1 : 0; }
    int max_hits() const { return polar_scene() ? MAX_HITS_CAP : FAN_MAX_HITS; }

    // ------------------------------------------------------------------ binding
    void ensure_cand_capacity(int required_cands) {
        if (required_cands > max_allocated_cands) {
            int new_capacity = std::max(16384, required_cands * 2);
            auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            for (int b = 0; b < 2; ++b) {
                d_candidate_pids[b] = torch::full({new_capacity}, -1, oi);
                d_cand_approx[b]    = torch::empty({new_capacity}, of);
                d_final_scores[b]   = torch::empty({new_capacity}, of);
            }
            max_allocated_cands = new_capacity;
        }
    }

    void bind_index(
        torch::Tensor csr_row_ptrs, torch::Tensor csr_col_eids,
        torch::Tensor csr_block_sums, torch::Tensor csr_lengths, torch::Tensor map_packed,
        torch::Tensor doc_offsets, torch::Tensor doc_lens,
        torch::Tensor codes, torch::Tensor residuals, torch::Tensor centroids_128,
        torch::Tensor bucket_weights, torch::Tensor reversed_bit_map, torch::Tensor decomp_table,
        torch::Tensor doc_predicates, int64_t num_passages, int num_centroids
    ) {
        TORCH_CHECK(csr_col_eids.element_size() == 1 || csr_col_eids.element_size() == 2, "csr_col_eids must be 8- or 16-bit");
        TORCH_CHECK(csr_lengths.element_size() == 2, "csr_lengths must be 16-bit");
        TORCH_CHECK(bucket_weights.numel() == 4, "rerank kernels assume 2-bit residuals (4 bucket weights)");
        TORCH_CHECK(E == 0 || csr_col_eids.element_size() == 2 || E <= 256, "E > 256 needs 16-bit csr_col_eids");
        b_csr_row_ptrs = csr_row_ptrs.to(torch::kInt64).contiguous();
        b_csr_col_eids = csr_col_eids.contiguous();
        b_eid_bytes = (int)csr_col_eids.element_size();
        b_csr_block_sums = csr_block_sums.to(torch::kInt64).contiguous();
        b_csr_lengths = csr_lengths.contiguous();
        b_map_packed = map_packed.contiguous();
        b_doc_offsets = doc_offsets.contiguous();
        b_doc_lens = doc_lens.contiguous();
        b_codes = codes.contiguous();
        b_residuals = residuals.contiguous();
        b_centroids_128 = centroids_128.to(torch::kFloat16).contiguous();
        b_bucket_weights = bucket_weights.contiguous();
        b_reversed_bit_map = reversed_bit_map.contiguous();
        b_decomp_table = decomp_table.contiguous();
        b_doc_predicates = doc_predicates.to(torch::kUInt32).contiguous();
        b_num_passages = doc_predicates.size(0);
        b_num_centroids = num_centroids;
        TORCH_CHECK(b_csr_row_ptrs.numel() == (int64_t)NUM_3D_SUBSPACES * (num_centroids + 1),
                    "csr_row_ptrs must be [32, num_centroids + 1]");
        for (int b = 0; b < 2; ++b)
            d_passage_scores[b] = torch::zeros({b_num_passages}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        ensure_cand_capacity(16384);
        is_bound = true;
    }

    // Token-level plain-IVF index: global token id of every posting (same order as csr_packed_24).
    void bind_token_ids(torch::Tensor token_ids) {
        TORCH_CHECK(is_bound, "bind_index first");
        TORCH_CHECK(token_ids.numel() * 3 == b_map_packed.numel(), "one token id per posting");
        b_token_ids = token_ids.to(torch::kInt32).contiguous();
    }

    // Per-query residual LUT [ntok, 32, 256] (float32, on the GPU); an empty tensor turns residual
    // scoring off. Used by the next search; not valid for search_batch_pipelined (one LUT per query).
    void set_residual_lut(torch::Tensor lut) {
        if (lut.numel() == 0) {
            b_res_lut = torch::Tensor();
            ragrt_set_residual_scoring(nullptr, nullptr, nullptr);
            return;
        }
        TORCH_CHECK(b_token_ids.defined(), "residual scoring needs bind_token_ids (token-level plain-IVF index)");
        TORCH_CHECK(lut.dim() == 3 && lut.size(1) == 32 && lut.size(2) == 256 && lut.size(0) <= 32, "lut must be [ntok<=32, 32, 256]");
        b_res_lut = lut.to(torch::kFloat32).contiguous();
        ragrt_set_residual_scoring((const uint32_t*)b_token_ids.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(),
                                   b_res_lut.data_ptr<float>());
    }

    // ------------------------------------------------------------------ stages
    void fill_params(CorrParams3D& p, const float* qp, int buf, int level, int k_eids) {
        p = {};
        p.k_iter = k_eids;
        p.query_points = (float3*)qp;
        p.geometry = geometry;
        p.num_entries = E;
        p.max_hits = max_hits();
        p.centroid_xyz = d_centroid_xyz;
        p.prim_eid = d_prim_eid;
        for (int s = 0; s < NUM_3D_SUBSPACES; ++s) {
            if (polar_scene()) {
                p.subspace_gas[s] = level_gas[level][s].handle;
                p.tmax[s] = 1.0f / level_tau[level][s];
            } else {
                p.subspace_gas[s] = fan_gas[s].handle;
                p.tmax[s] = 2.0f;
            }
        }
        p.out_hit_centroid = buf_out_c[buf].data_ptr<int>();
        p.out_hit_value = buf_out_v[buf].data_ptr<float>();
        p.out_hit_count = buf_out_n[buf].data_ptr<int>();
    }

    // d_params: device copy of p (only read for the RT path).
    void run_stage1(const CorrParams3D& p, CUdeviceptr d_params, const float* qp, int total_rays, int k_eids,
                    int level, cudaStream_t stream) {
        if (stage1_mode == 2) {
            TORCH_CHECK(E <= p.max_hits, "all-codewords mode needs E <= ", p.max_hits, " (use a polar geometry, E <= 256)");
            launch_all_codewords_stage1(qp, (const float*)d_centroid_xyz, total_rays, E, p.max_hits,
                                        p.out_hit_centroid, p.out_hit_value, p.out_hit_count, stream);
        } else if (stage1_mode == 0) {
            OPTIX_CHECK(optixLaunch(pipe, stream, d_params, sizeof(CorrParams3D), &sbt, total_rays, 1, 1));
        } else if (geometry == GEOM_POLAR) {
            // Same threshold over the norm-sorted codebook, so it also returns primitive indices.
            launch_threshold_stage1(qp, (const float*)d_sorted_xyz, d_level_tau + level * NUM_3D_SUBSPACES,
                                    total_rays, E, p.max_hits, p.out_hit_centroid, p.out_hit_value,
                                    p.out_hit_count, stream);
        } else {
            launch_bruteforce_stage1(qp, (const float*)d_centroid_xyz, total_rays, E, k_eids, p.max_hits,
                                     p.out_hit_centroid, p.out_hit_value, p.out_hit_count, stream);
        }
    }

    // Stages 2 and 3: k_candidates pids into d_candidate_pids[buf] (-1 = padding) and their
    // approximate scores into d_cand_approx[buf]. Torch ops run on the current stream (== stream).
    void stages23(const int* topc, const float* scores, int nq, int k_cent, int k_candidates,
                  uint32_t query_mask, int k_eids, int buf, cudaStream_t stream, cudaEvent_t ev_after_s2) {
        int total_rays = nq * NUM_3D_SUBSPACES;
        ragrt_stage2(buf_out_c[buf].data_ptr<int>(), buf_out_v[buf].data_ptr<float>(), buf_out_n[buf].data_ptr<int>(),
                     topc, scores, total_rays, max_hits(), k_eids, k_cent, nq, b_num_centroids,
                     b_csr_row_ptrs.data_ptr<int64_t>(), b_csr_col_eids.data_ptr(), b_eid_bytes,
                     b_csr_block_sums.data_ptr<int64_t>(), (const uint16_t*)b_csr_lengths.data_ptr(),
                     hits_are_prims() ? d_prim_eid : nullptr, E, buf, stream);
        if (ev_after_s2) CUDA_CHECK(cudaEventRecord(ev_after_s2, stream));
        auto cand = d_candidate_pids[buf].slice(0, 0, k_candidates);
        auto approx = d_cand_approx[buf].slice(0, 0, k_candidates);
        if (stage3_sparse && ragrt_get_stage3_mode() == 3) {     // pid-only sort: L entries aligned with the postings
            int* d_pids = nullptr; float* d_sums = nullptr;
            long long L = ragrt_stage3_pidsort(b_map_packed.data_ptr<uint8_t>(), (int)b_num_passages,
                                               (const uint32_t*)b_doc_predicates.data_ptr(), query_mask,
                                               (long long)k_candidates, buf, stream, &d_pids, &d_sums);
            auto sums = torch::from_blob(d_sums, {L}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            auto pids = torch::from_blob(d_pids, {L}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            if (L == k_candidates) {
                approx.copy_(sums);
                cand.copy_(pids);
            } else {
                auto top = torch::topk(sums, k_candidates);
                approx.copy_(std::get<0>(top));
                cand.copy_(pids.index({std::get<1>(top)}));
            }
        } else if (stage3_sparse) {
            int* d_pids = nullptr; float* d_sums = nullptr; long long U = 0;
            long long L = ragrt_stage3_sparse(b_map_packed.data_ptr<uint8_t>(), (int)b_num_passages,
                                              (const uint32_t*)b_doc_predicates.data_ptr(), query_mask,
                                              (long long)k_candidates, buf, stream, &d_pids, &d_sums, &U);
            auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            auto sums = torch::from_blob(d_sums, {L}, of);
            auto pids = torch::from_blob(d_pids, {L}, oi);
            if (L == k_candidates) {            // U <= k: every touched passage is a candidate
                approx.copy_(sums);
                cand.copy_(pids);
            } else {
                auto top = torch::topk(sums, k_candidates);
                approx.copy_(std::get<0>(top));
                cand.copy_(pids.index({std::get<1>(top)}));
            }
        } else {
            ragrt_stage3_dense(b_map_packed.data_ptr<uint8_t>(), d_passage_scores[buf].data_ptr<float>(),
                               (int)b_num_passages, (const uint32_t*)b_doc_predicates.data_ptr(), query_mask,
                               nq, buf, stream);
            auto top = torch::topk(d_passage_scores[buf], k_candidates);
            approx.copy_(std::get<0>(top));
            cand.copy_(std::get<1>(top));
        }
    }

    torch::Tensor rerank(const torch::Tensor& Q_fp16, int k_candidates, int k_top, uint32_t query_mask,
                         int buf, cudaStream_t stream) {
        launch_tile_maxsim_fused_decomp(
            Q_fp16.data_ptr(), d_candidate_pids[buf].data_ptr<int>(), b_doc_offsets.data_ptr<int64_t>(),
            b_doc_lens.data_ptr<int>(), k_candidates, b_codes.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(),
            b_centroids_128.data_ptr(), b_bucket_weights.data_ptr(), b_reversed_bit_map.data_ptr<uint8_t>(),
            b_decomp_table.data_ptr<uint8_t>(), d_final_scores[buf].data_ptr<float>(), stream);
        launch_mask_failing_candidates(d_candidate_pids[buf].data_ptr<int>(), k_candidates,
            (const uint32_t*)b_doc_predicates.data_ptr(), query_mask, d_final_scores[buf].data_ptr<float>(), stream);
        auto top = torch::topk(d_final_scores[buf].slice(0, 0, k_candidates), std::min(k_top, k_candidates));
        return d_candidate_pids[buf].slice(0, 0, k_candidates).index({std::get<1>(top)});
    }

    void check_query(const torch::Tensor& Q_sub3d, int k_candidates, int k_eids) {
        TORCH_CHECK(is_bound, "Index must be bound before search!");
        TORCH_CHECK(E > 0, "build() first");
        TORCH_CHECK(Q_sub3d.dim() == 3 && Q_sub3d.size(1) == NUM_3D_SUBSPACES && Q_sub3d.size(2) == 3, "Q_sub3d must be [ntok, 32, 3]");
        TORCH_CHECK(Q_sub3d.size(0) >= 1 && Q_sub3d.size(0) <= 32, "1..32 query tokens");
        TORCH_CHECK(k_candidates >= 1 && k_candidates <= b_num_passages, "bad k_candidates");
        TORCH_CHECK(k_eids >= 1 && k_eids <= 256, "k_eids must be in [1, 256]");
        ensure_cand_capacity(k_candidates);
    }

    int level_or_zero(int k_eids) { return polar_scene() ? level_for(k_eids) : 0; }

    torch::Tensor search_single_query_native(
        torch::Tensor Q_full_128_fp32, torch::Tensor Q_full_fp16, torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores,
        int k_cent, int k_candidates, int k_top, uint32_t query_mask = 0, int k_eids = 16
    ) {
        check_query(Q_sub3d, k_candidates, k_eids);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int nq = (int)Q_sub3d.size(0);
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.reshape({total_rays, 3}).contiguous();
        int level = level_or_zero(k_eids);
        CorrParams3D p; fill_params(p, qp.data_ptr<float>(), 0, level, k_eids);
        if (stage1_mode == 0) CUDA_CHECK(cudaMemcpyAsync(d_p_single, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        run_stage1(p, (CUdeviceptr)d_p_single, qp.data_ptr<float>(), total_rays, k_eids, level, stream);
        stages23(topc.data_ptr<int>(), scores.data_ptr<float>(), nq, k_cent, k_candidates, query_mask, k_eids, 0, stream, nullptr);
        return rerank(Q_full_fp16, k_candidates, k_top, query_mask, 0, stream);
    }

    std::tuple<torch::Tensor, std::vector<float>> search_single_query_profiled(
        torch::Tensor Q_full_128_fp32, torch::Tensor Q_full_fp16, torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores,
        int k_cent, int k_candidates, int k_top, uint32_t query_mask = 0, int k_eids = 16
    ) {
        check_query(Q_sub3d, k_candidates, k_eids);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int nq = (int)Q_sub3d.size(0);
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.reshape({total_rays, 3}).contiguous();
        int level = level_or_zero(k_eids);
        CorrParams3D p; fill_params(p, qp.data_ptr<float>(), 0, level, k_eids);
        if (stage1_mode == 0) CUDA_CHECK(cudaMemcpyAsync(d_p_single, &p, sizeof(p), cudaMemcpyHostToDevice, stream));

        cudaEvent_t ev[5];
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
        run_stage1(p, (CUdeviceptr)d_p_single, qp.data_ptr<float>(), total_rays, k_eids, level, stream);
        cudaEventRecord(ev[1], stream);
        stages23(topc.data_ptr<int>(), scores.data_ptr<float>(), nq, k_cent, k_candidates, query_mask, k_eids, 0, stream, ev[2]);
        cudaEventRecord(ev[3], stream);
        auto out = rerank(Q_full_fp16, k_candidates, k_top, query_mask, 0, stream);
        cudaEventRecord(ev[4], stream);
        cudaEventSynchronize(ev[4]);
        std::vector<float> t(4, 0.0f);
        for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]);
        for (auto& e : ev) cudaEventDestroy(e);
        return std::make_tuple(out, t);
    }

    torch::Tensor search_batch_pipelined(
        std::vector<torch::Tensor> Q_full_128_list,
        std::vector<torch::Tensor> Q_full_fp16_list,
        std::vector<torch::Tensor> Q_sub3d_list,
        std::vector<torch::Tensor> topc_list,
        std::vector<torch::Tensor> scores_list,
        int k_cent, int k_candidates, int k_top,
        uint32_t query_mask = 0, int k_eids = 16
    ) {
        int num_queries = (int)Q_full_fp16_list.size();
        TORCH_CHECK(num_queries >= 1, "empty batch");
        for (auto& q : Q_sub3d_list) check_query(q, k_candidates, k_eids);
        auto out_results = torch::full({num_queries, k_top}, -1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int level = level_or_zero(k_eids);

        std::vector<torch::Tensor> qps(num_queries);
        std::vector<CorrParams3D> h_params_all(num_queries);
        for (int i = 0; i < num_queries; ++i) {
            qps[i] = Q_sub3d_list[i].reshape({-1, 3}).contiguous();
            fill_params(h_params_all[i], qps[i].data_ptr<float>(), i % 2, level, k_eids);
        }
        auto d_params_batch = torch::empty({num_queries * (int64_t)sizeof(CorrParams3D)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        CUDA_CHECK(cudaMemcpy(d_params_batch.data_ptr(), h_params_all.data(),
            num_queries * sizeof(CorrParams3D), cudaMemcpyHostToDevice));
        // A pageable H2D cudaMemcpy can return before the DMA lands, and stream_cand is
        // non-blocking (not ordered after the legacy default stream): wait explicitly.
        CUDA_CHECK(cudaDeviceSynchronize());

        c10::cuda::CUDAStream c_stream_cand  = c10::cuda::getStreamFromExternal(stream_cand, 0);
        c10::cuda::CUDAStream c_stream_score = c10::cuda::getStreamFromExternal(stream_score, 0);

        for (int i = 0; i < num_queries; ++i) {
            int buf = i % 2;
            if (i >= 2) CUDA_CHECK(cudaStreamWaitEvent(stream_cand, event_tms_done[buf], 0));
            int nq = (int)Q_sub3d_list[i].size(0);
            int total_rays = nq * NUM_3D_SUBSPACES;
            CUdeviceptr param_ptr = (CUdeviceptr)(d_params_batch.data_ptr<uint8_t>() + i * sizeof(CorrParams3D));
            {
                c10::cuda::CUDAStreamGuard guard(c_stream_cand);
                run_stage1(h_params_all[i], param_ptr, qps[i].data_ptr<float>(), total_rays, k_eids, level, stream_cand);
                stages23(topc_list[i].data_ptr<int>(), scores_list[i].data_ptr<float>(), nq, k_cent, k_candidates,
                         query_mask, k_eids, buf, stream_cand, nullptr);
                CUDA_CHECK(cudaEventRecord(event_cand_ready[buf], stream_cand));
            }
            {
                c10::cuda::CUDAStreamGuard guard(c_stream_score);
                CUDA_CHECK(cudaStreamWaitEvent(stream_score, event_cand_ready[buf], 0));
                auto ranked = rerank(Q_full_fp16_list[i], k_candidates, k_top, query_mask, buf, stream_score);
                out_results[i].slice(0, 0, ranked.size(0)).copy_(ranked);
                CUDA_CHECK(cudaEventRecord(event_tms_done[buf], stream_score));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_cand));
        CUDA_CHECK(cudaStreamSynchronize(stream_score));
        return out_results;
    }

    // ------------------------------------------------------------------ diagnostics / tests
    // Stage 1 only: (eids [rays, max_hits], values, counts) for ray = token * 32 + subspace.
    std::vector<torch::Tensor> stage1_hits(torch::Tensor Q_sub3d, int k_eids) {
        TORCH_CHECK(E > 0, "build() first");
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int nq = (int)Q_sub3d.size(0);
        TORCH_CHECK(nq >= 1 && nq <= 32, "1..32 tokens");
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.reshape({total_rays, 3}).to(torch::kFloat32).contiguous();
        int level = level_or_zero(k_eids);
        CorrParams3D p; fill_params(p, qp.data_ptr<float>(), 0, level, k_eids);
        buf_out_c[0].fill_(-1); buf_out_v[0].zero_();
        if (stage1_mode == 0) CUDA_CHECK(cudaMemcpyAsync(d_p_single, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        run_stage1(p, (CUdeviceptr)d_p_single, qp.data_ptr<float>(), total_rays, k_eids, level, stream);
        int mh = max_hits();
        auto c = buf_out_c[0].view({-1}).slice(0, 0, (int64_t)total_rays * mh).view({total_rays, mh}).clone();
        if (hits_are_prims()) {   // report eids, like every other mode
            auto sub = (torch::arange(total_rays, c.options().dtype(torch::kInt64)) % NUM_3D_SUBSPACES).unsqueeze(1) * E;
            auto valid = c >= 0;
            auto mapped = t_prim_eid.index({(sub + c.clamp_min(0).to(torch::kInt64)).view({-1})}).view({total_rays, mh}).to(torch::kInt32);
            c = torch::where(valid, mapped, c);
        }
        auto v = buf_out_v[0].view({-1}).slice(0, 0, (int64_t)total_rays * mh).view({total_rays, mh}).clone();
        auto n = buf_out_n[0].slice(0, 0, total_rays).clone();
        return {c, v, n};
    }

    // Stages 1-3 only: (candidate pids [k_candidates] with -1 padding, approximate scores).
    std::vector<torch::Tensor> candidates(torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores,
                                          int k_cent, int k_candidates, uint32_t query_mask, int k_eids) {
        check_query(Q_sub3d, k_candidates, k_eids);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int nq = (int)Q_sub3d.size(0);
        int total_rays = nq * NUM_3D_SUBSPACES;
        auto qp = Q_sub3d.reshape({total_rays, 3}).contiguous();
        int level = level_or_zero(k_eids);
        CorrParams3D p; fill_params(p, qp.data_ptr<float>(), 0, level, k_eids);
        if (stage1_mode == 0) CUDA_CHECK(cudaMemcpyAsync(d_p_single, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        run_stage1(p, (CUdeviceptr)d_p_single, qp.data_ptr<float>(), total_rays, k_eids, level, stream);
        stages23(topc.data_ptr<int>(), scores.data_ptr<float>(), nq, k_cent, k_candidates, query_mask, k_eids, 0, stream, nullptr);
        return {d_candidate_pids[0].slice(0, 0, k_candidates).clone(), d_cand_approx[0].slice(0, 0, k_candidates).clone()};
    }

    // ------------------------------------------------------------------ static (CUDA graph) path
    // A query = phase A (Stage 1 + Stage 2) -> host reads the posting count P (one sync) ->
    // phase B (sync-free sparse Stage 3 over a fixed capacity, top-k, rerank, filter, top-k).
    // Neither phase synchronizes or allocates driver memory, so Python can capture each one in a
    // CUDA graph (engines.GraphSearch). Call reserve_static() first, outside any capture.
    long long reserve_static(long long cap, int k_candidates) {
        TORCH_CHECK(is_bound, "bind_index first");
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        ensure_cand_capacity(k_candidates);
        return ragrt_reserve_sparse(cap, 0, stream);
    }
    long long sparse_capacity() { return ragrt_sparse_capacity(0); }
    torch::Tensor total_postings_tensor() {
        TORCH_CHECK(is_bound, "bind_index first");
        return torch::from_blob((void*)ragrt_total_postings_ptr(0), {1},
                                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    }

    void static_phase_a(torch::Tensor Q_sub3d, torch::Tensor topc, torch::Tensor scores, int k_cent, int k_eids) {
        TORCH_CHECK(stage1_mode != 0, "the static path needs a CUDA-core Stage 1 (modes 1, 2)");
        TORCH_CHECK(stage3_sparse, "the static path needs the sparse Stage 3");
        TORCH_CHECK(!ragrt_residual_scoring_on(), "the static path does not support residual scoring");
        TORCH_CHECK(is_bound && E > 0, "bind_index and build first");
        TORCH_CHECK(Q_sub3d.dim() == 3 && Q_sub3d.size(1) == NUM_3D_SUBSPACES && Q_sub3d.size(0) <= 32, "Q_sub3d [ntok<=32, 32, 3]");
        TORCH_CHECK(k_eids >= 1 && k_eids <= 256, "k_eids must be in [1, 256]");
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int nq = (int)Q_sub3d.size(0);
        int total_rays = nq * NUM_3D_SUBSPACES;
        const float* qp = Q_sub3d.data_ptr<float>();
        TORCH_CHECK(Q_sub3d.is_contiguous() && Q_sub3d.scalar_type() == torch::kFloat32, "Q_sub3d contiguous float32");
        int level = level_or_zero(k_eids);
        CorrParams3D p; fill_params(p, qp, 0, level, k_eids);
        run_stage1(p, (CUdeviceptr)d_p_single, qp, total_rays, k_eids, level, stream);
        ragrt_stage2(buf_out_c[0].data_ptr<int>(), buf_out_v[0].data_ptr<float>(), buf_out_n[0].data_ptr<int>(),
                     topc.data_ptr<int>(), scores.data_ptr<float>(), total_rays, max_hits(), k_eids, k_cent, nq,
                     b_num_centroids, b_csr_row_ptrs.data_ptr<int64_t>(), b_csr_col_eids.data_ptr(), b_eid_bytes,
                     b_csr_block_sums.data_ptr<int64_t>(), (const uint16_t*)b_csr_lengths.data_ptr(),
                     hits_are_prims() ? d_prim_eid : nullptr, E, 0, stream);
    }

    // Returns the top k_top pids (-1 = padding). mask_dev: one int32 on the GPU holding the filter bits.
    torch::Tensor static_phase_b(torch::Tensor Q_full_fp16, long long cap, int k_candidates, int k_top, torch::Tensor mask_dev) {
        TORCH_CHECK(k_candidates >= 1 && k_candidates <= cap && k_candidates <= max_allocated_cands, "bad k_candidates");
        TORCH_CHECK(mask_dev.is_cuda() && mask_dev.numel() == 1 && mask_dev.element_size() == 4, "mask_dev: one 32-bit int on the GPU");
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        int* d_pids = nullptr; float* d_sums = nullptr;
        // Stage 3 mode at capture time picks the kernels the graph holds: 3 = pid sort, else the 64-bit sort.
        auto stage3_static = ragrt_get_stage3_mode() == 3 ? ragrt_stage3_pidsort_static : ragrt_stage3_sparse_static;
        long long L = stage3_static(b_map_packed.data_ptr<uint8_t>(), (int)b_num_passages,
                                    (const uint32_t*)b_doc_predicates.data_ptr(), (const uint32_t*)mask_dev.data_ptr(),
                                    cap, 0, stream, &d_pids, &d_sums);
        TORCH_CHECK(L == cap, "sparse capacity ", sparse_capacity(), " < cap ", cap, ": call reserve_static first");
        auto of = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        auto sums = torch::from_blob(d_sums, {L}, of);
        auto pids = torch::from_blob(d_pids, {L}, oi);
        auto top = torch::topk(sums, k_candidates);
        auto cand = d_candidate_pids[0].slice(0, 0, k_candidates);
        cand.copy_(pids.index({std::get<1>(top)}));
        d_cand_approx[0].slice(0, 0, k_candidates).copy_(std::get<0>(top));
        launch_tile_maxsim_fused_decomp(
            Q_full_fp16.data_ptr(), d_candidate_pids[0].data_ptr<int>(), b_doc_offsets.data_ptr<int64_t>(),
            b_doc_lens.data_ptr<int>(), k_candidates, b_codes.data_ptr<int>(), b_residuals.data_ptr<uint8_t>(),
            b_centroids_128.data_ptr(), b_bucket_weights.data_ptr(), b_reversed_bit_map.data_ptr<uint8_t>(),
            b_decomp_table.data_ptr<uint8_t>(), d_final_scores[0].data_ptr<float>(), stream);
        launch_mask_failing_candidates_dev(d_candidate_pids[0].data_ptr<int>(), k_candidates,
            (const uint32_t*)b_doc_predicates.data_ptr(), (const uint32_t*)mask_dev.data_ptr(),
            d_final_scores[0].data_ptr<float>(), stream);
        auto fin = torch::topk(d_final_scores[0].slice(0, 0, k_candidates), std::min(k_top, k_candidates));
        return cand.index({std::get<1>(fin)});
    }

    pybind11::dict info() {
        pybind11::dict d;
        d["num_entries"] = E;
        d["geometry"] = geometry == GEOM_POLAR ? "polar" : (geometry == GEOM_POLAR_ITER ? "polar_iter" : "fan");
        d["stage1"] = stage1_mode == 0 ? "rt" : (stage1_mode == 1 ? "cuda_cores" : "all_codewords");
        d["stage3"] = stage3_sparse ? "sparse" : "dense";
        d["rerank"] = ragrt_get_rerank_mode() == 1 ? "wmma" : "simt";
        d["eid_bytes"] = b_eid_bytes;
        d["codeword_weight"] = ragrt_get_codeword_weight();
        d["compact_scores"] = ragrt_get_compact_scores();
        d["stage3_hash"] = ragrt_get_stage3_hash();
        d["stage3_mode"] = ragrt_get_stage3_mode();
        d["residual_scoring"] = ragrt_residual_scoring_on();
        d["token_level_index"] = b_token_ids.defined();
        pybind11::list lv;
        for (size_t l = 0; l < level_k.size(); ++l) {
            pybind11::dict x;
            x["k"] = level_k[l];
            x["tau"] = std::vector<float>(level_tau[l].begin(), level_tau[l].end());
            x["prims"] = std::vector<int>(level_prims[l].begin(), level_prims[l].end());
            lv.append(x);
        }
        d["levels"] = lv;
        return d;
    }
};

torch::Tensor native_tile_maxsim_fused_decomp(
    torch::Tensor Q, torch::Tensor pids, torch::Tensor doc_offsets, torch::Tensor doc_lens,
    torch::Tensor codes, torch::Tensor residuals, torch::Tensor centroids,
    torch::Tensor bucket_weights, torch::Tensor reversed_bit_map, torch::Tensor decomp_table
) {
    TORCH_CHECK(Q.dim() == 2 && Q.size(0) == 32 && Q.size(1) == 128,
        "fused MaxSim kernel reads Q as [32, 128]; got [", Q.size(0), ", ", Q.size(1), "]");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 && Q.is_contiguous(), "Q must be contiguous fp16");
    TORCH_CHECK(bucket_weights.numel() == 4, "2-bit residuals expected");
    int num_cands = pids.size(0);
    auto out_scores = torch::empty({num_cands}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    launch_tile_maxsim_fused_decomp(
        Q.data_ptr(), pids.data_ptr<int>(), doc_offsets.data_ptr<int64_t>(), doc_lens.data_ptr<int>(),
        num_cands, codes.data_ptr<int>(), residuals.data_ptr<uint8_t>(), centroids.data_ptr(),
        bucket_weights.data_ptr(), reversed_bit_map.data_ptr<uint8_t>(), decomp_table.data_ptr<uint8_t>(),
        out_scores.data_ptr<float>(), c10::cuda::getCurrentCUDAStream().stream()
    );
    return out_scores;
}

// Standalone entry point for testing the exact brute-force Stage 1 against torch.
std::vector<torch::Tensor> bruteforce_stage1_standalone(torch::Tensor q_points, torch::Tensor codewords, int64_t k, int64_t max_hits) {
    TORCH_CHECK(q_points.is_cuda() && q_points.scalar_type() == torch::kFloat32 && q_points.dim() == 2 && q_points.size(1) == 3);
    TORCH_CHECK(codewords.is_cuda() && codewords.scalar_type() == torch::kFloat32 && codewords.dim() == 2 && codewords.size(1) == 3);
    TORCH_CHECK(codewords.size(0) % NUM_3D_SUBSPACES == 0, "codewords must be [32 * E, 3]");
    auto q = q_points.contiguous(); auto cw = codewords.contiguous();
    int R = (int)q.size(0), E = (int)(cw.size(0) / NUM_3D_SUBSPACES);
    auto oi = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto out_c = torch::full({R, max_hits}, -1, oi);
    auto out_v = torch::zeros({R, max_hits}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto out_n = torch::zeros({R}, oi);
    launch_bruteforce_stage1(q.data_ptr<float>(), cw.data_ptr<float>(), R, E, (int)k, (int)max_hits,
                             out_c.data_ptr<int>(), out_v.data_ptr<float>(), out_n.data_ptr<int>(),
                             c10::cuda::getCurrentCUDAStream().stream());
    return {out_c, out_v, out_n};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<CorrIndex3D>(m, "CorrIndex3D")
        .def(pybind11::init<std::string>(), pybind11::arg("ptx_path"))
        .def("build", &CorrIndex3D::build, pybind11::arg("codebooks"), pybind11::arg("base_radius") = 0.95, pybind11::arg("num_tris") = 4)
        .def("set_polar_levels", &CorrIndex3D::set_polar_levels, pybind11::arg("taus"), pybind11::arg("ks"),
             "Exact-threshold scenes: taus [L, 32] on unit-query . codeword, ks[l] = k_eids level l serves")
        .def("level_for", &CorrIndex3D::level_for)
        .def("bind_index", &CorrIndex3D::bind_index)
        .def("bind_token_ids", &CorrIndex3D::bind_token_ids, pybind11::arg("token_ids"),
             "token-level plain-IVF index: global token id per posting (enables residual scoring)")
        .def("set_residual_lut", &CorrIndex3D::set_residual_lut, pybind11::arg("lut"),
             "per-query residual LUT [ntok, 32, 256] float32 (empty tensor = residual scoring off)")
        .def("set_geometry", &CorrIndex3D::set_geometry, pybind11::arg("geometry"), "0 = fan (legacy), 1 = polar (exact threshold, any-hit), 2 = polar_iter (k nearest-hit traversals, no hit shaders)")
        .def("set_stage1_mode", &CorrIndex3D::set_stage1_mode, pybind11::arg("mode"),
             "0 = OptiX RT cores, 1 = CUDA cores (fan: exact top-k; polar: same threshold, linear scan)")
        .def("set_stage3_sparse", &CorrIndex3D::set_stage3_sparse, pybind11::arg("sparse"))
        .def("info", &CorrIndex3D::info)
        .def("stage1_hits", &CorrIndex3D::stage1_hits, pybind11::arg("Q_sub3d"), pybind11::arg("k_eids") = 16)
        .def("candidates", &CorrIndex3D::candidates,
             pybind11::arg("Q_sub3d"), pybind11::arg("topc"), pybind11::arg("scores"), pybind11::arg("k_cent"),
             pybind11::arg("k_candidates"), pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16)
        .def("search_single_query_profiled", &CorrIndex3D::search_single_query_profiled,
             pybind11::arg("Q_full_128_fp32"), pybind11::arg("Q_full_fp16"), pybind11::arg("Q_sub3d"), pybind11::arg("topc"), pybind11::arg("scores"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16)
        .def("search_single_query_native", &CorrIndex3D::search_single_query_native,
             pybind11::arg("Q_full_128_fp32"), pybind11::arg("Q_full_fp16"), pybind11::arg("Q_sub3d"), pybind11::arg("topc"), pybind11::arg("scores"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16)
        .def("reserve_static", &CorrIndex3D::reserve_static, pybind11::arg("cap"), pybind11::arg("k_candidates"),
             "allocate task/sparse buffers for the static (CUDA graph) path; returns the sparse capacity")
        .def("sparse_capacity", &CorrIndex3D::sparse_capacity)
        .def("total_postings_tensor", &CorrIndex3D::total_postings_tensor,
             "int64 [1] view of the posting count written by Stage 2 (read with .item() between the phases)")
        .def("static_phase_a", &CorrIndex3D::static_phase_a, pybind11::arg("Q_sub3d"), pybind11::arg("topc"),
             pybind11::arg("scores"), pybind11::arg("k_cent"), pybind11::arg("k_eids"))
        .def("static_phase_b", &CorrIndex3D::static_phase_b, pybind11::arg("Q_full_fp16"), pybind11::arg("cap"),
             pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("mask_dev"))
        .def("search_batch_pipelined", &CorrIndex3D::search_batch_pipelined,
             pybind11::arg("Q_full_128_list"), pybind11::arg("Q_full_fp16_list"), pybind11::arg("Q_sub3d_list"), pybind11::arg("topc_list"), pybind11::arg("scores_list"),
             pybind11::arg("k_cent"), pybind11::arg("k_candidates"), pybind11::arg("k_top"), pybind11::arg("query_mask") = 0, pybind11::arg("k_eids") = 16);
    m.def("native_tile_maxsim_fused_decomp", &native_tile_maxsim_fused_decomp);
    m.def("set_rerank_mode", [](const std::string& mode) {
        TORCH_CHECK(mode == "wmma" || mode == "simt", "rerank mode must be 'wmma' or 'simt'");
        ragrt_set_rerank_mode(mode == "wmma" ? 1 : 0);
    }, "Stage-4 / TMS kernel: 'wmma' (tensor cores, default) or 'simt' (original)");
    m.def("set_codeword_weight", [](double w) { ragrt_set_codeword_weight((float)w); },
          "Weight of the codeword term q_s.w in each posting's Stage-2 score: 1 (default) or 0 (centroid-only ablation)");
    m.def("get_codeword_weight", []() { return (double)ragrt_get_codeword_weight(); });
    m.def("set_compact_scores", [](bool on) { ragrt_set_compact_scores(on ? 1 : 0); },
          "Stage 2 reads centroid scores as [ntok, k_cent] aligned with topc (true) or [ntok, num_centroids]");
    m.def("get_compact_scores", []() { return ragrt_get_compact_scores() != 0; });
    m.def("set_stage3_hash", [](bool on) { ragrt_set_stage3_hash(on ? 1 : 0); },
          "sparse Stage 3 by hash tables (true) instead of a radix sort (false, default)");
    m.def("get_stage3_hash", []() { return ragrt_get_stage3_hash() != 0; });
    m.def("set_stage3_mode", [](int mode) { ragrt_set_stage3_mode(mode); },
          "sparse Stage 3: 0 = 64-bit key radix sort (default), 1 = hash tables, 2 = 32-bit key / 16-bit value pair sort, "
          "3 = pid-only key sort with a per-passage segment reduce");
    m.def("get_stage3_mode", []() { return ragrt_get_stage3_mode(); });
    m.def("get_rerank_mode", []() { return std::string(ragrt_get_rerank_mode() == 1 ? "wmma" : "simt"); });
    m.def("bruteforce_stage1", &bruteforce_stage1_standalone,
          "Exact top-k positive codewords per ray on CUDA cores: (q [R,3], codewords [32*E,3], k, max_hits) -> (ids, vals, counts)");
    m.def("set_drop_stats", [](bool enabled) { ragrt_set_drop_stats(enabled ? 1 : 0); },
          "Enable/disable the Stage-2 diagnostic counters (off by default; leave off for timing).");
    m.def("reset_drop_stats", []() { ragrt_reset_drop_stats(); });
    m.def("get_drop_stats", []() {
        unsigned long long v[NUM_DROP_STATS];
        ragrt_drop_stats(v);
        pybind11::dict d;
        for (int i = 0; i < NUM_DROP_STATS; ++i) d[DROP_STAT_NAMES[i]] = v[i];
        return d;
    }, "Cumulative counters since the last reset (all zero unless set_drop_stats(True)).");
}
