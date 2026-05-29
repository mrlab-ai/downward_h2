#include "hm2_torch_cuda.h"

#include <cuda_runtime.h>

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <chrono>

namespace hm2_torch {

namespace {

struct HM2CudaProfileStats {
    uint64_t direct_calls = 0;
    uint64_t graph_calls = 0;
    uint64_t graph_recaptures = 0;
    double direct_max_add_ms = 0.0;
    double direct_amin_ms = 0.0;
    double graph_launch_ms = 0.0;
    double graph_capture_ms = 0.0;
};

HM2CudaProfileStats g_hm2_cuda_profile_stats;

bool hm2_cuda_profile_enabled() {
    static bool enabled = []() {
        const char *value = std::getenv("HM2_TORCH_PROFILE");
        return value && std::string(value) != "0";
    }();
    return enabled;
}

void maybe_log_hm2_cuda_profile() {
    HM2CudaProfileStats &s = g_hm2_cuda_profile_stats;
    const uint64_t total_calls = s.direct_calls + s.graph_calls;
    if (total_calls == 0 || (total_calls % 100) != 0) {
        return;
    }

    const double direct_denom = s.direct_calls > 0 ? static_cast<double>(s.direct_calls) : 1.0;
    const double graph_denom = s.graph_calls > 0 ? static_cast<double>(s.graph_calls) : 1.0;
    std::printf(
        "[hm2_cuda_profile] direct_calls=%llu graph_calls=%llu recaptures=%llu avg_direct_max_add_ms=%.6f avg_direct_amin_ms=%.6f avg_graph_launch_ms=%.6f avg_graph_capture_ms=%.6f\n",
        static_cast<unsigned long long>(s.direct_calls),
        static_cast<unsigned long long>(s.graph_calls),
        static_cast<unsigned long long>(s.graph_recaptures),
        s.direct_max_add_ms / direct_denom,
        s.direct_amin_ms / direct_denom,
        s.graph_launch_ms / graph_denom,
        s.graph_capture_ms / graph_denom);
    std::fflush(stdout);
}

struct FusedGraphCache {
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t graph = nullptr;
    const int64_t *hm_ptr = nullptr;
    int64_t *max_ptr = nullptr;
    const int64_t *cost_ptr = nullptr;
    const int64_t *select_ptr = nullptr;
    const int32_t *offset_ptr = nullptr;
    const int64_t *head_ptr = nullptr;
    const int64_t *grouped_head_ptr = nullptr;
    const int32_t *grouped_head_offsets_ptr = nullptr;
    const int64_t *grouped_edge_ptr = nullptr;
    int32_t *changed_ptr = nullptr;
    int n_cp = -1;
    int batch = -1;
    int table = -1;
    int edges = -1;
    int grouped_heads = -1;
};

FusedGraphCache g_fused_graph_cache;

inline void clear_graph_cache(FusedGraphCache &cache) {
    if (cache.exec) {
        cudaGraphExecDestroy(cache.exec);
        cache.exec = nullptr;
    }
    if (cache.graph) {
        cudaGraphDestroy(cache.graph);
        cache.graph = nullptr;
    }
}

__global__ void hm2_fused_max_add_kernel(
    const int64_t *hm_table,
    int64_t *max_values,
    const int64_t *edge_costs_table_last,
    const int64_t *edge_select_indices,
    const int32_t *edge_offsets,
    int n_cp,
    int batch_size,
    int table_size,
    int num_edges) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_cp * batch_size * num_edges;
    if (global_idx >= total) {
        return;
    }

    int edge = global_idx % num_edges;
    int tmp = global_idx / num_edges;
    int batch = tmp % batch_size;
    int cp = tmp / batch_size;

    int start = edge_offsets[edge];
    int end = edge_offsets[edge + 1];

    int64_t max_val = LLONG_MIN;
    int64_t hm_base = static_cast<int64_t>(cp) * batch_size * table_size + static_cast<int64_t>(batch) * table_size;

    for (int i = start; i < end; ++i) {
        int64_t table_index = edge_select_indices[i];
        int64_t candidate = hm_table[hm_base + table_index];
        if (candidate > max_val) {
            max_val = candidate;
        }
    }

    max_val += edge_costs_table_last[(static_cast<int64_t>(cp) * num_edges) + edge];
    max_values[(static_cast<int64_t>(cp) * batch_size + batch) * num_edges + edge] = max_val;
}

__global__ void hm2_fused_grouped_amin_kernel(
    int64_t *hm_table,
    const int64_t *max_values,
    const int64_t *grouped_head_indices,
    const int32_t *grouped_head_offsets,
    const int64_t *grouped_edge_indices,
    int n_cp,
    int batch_size,
    int table_size,
    int num_edges,
    int num_grouped_heads,
    int32_t *changed_flag) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_cp * batch_size * num_grouped_heads;
    if (global_idx >= total) {
        return;
    }

    int grouped_head_index = global_idx % num_grouped_heads;
    int tmp = global_idx / num_grouped_heads;
    int batch = tmp % batch_size;
    int cp = tmp / batch_size;

    int64_t head = grouped_head_indices[grouped_head_index];
    int start = grouped_head_offsets[grouped_head_index];
    int end = grouped_head_offsets[grouped_head_index + 1];

    int64_t value = LLONG_MAX;
    int64_t base = (static_cast<int64_t>(cp) * batch_size + batch) * num_edges;
    for (int idx = start; idx < end; ++idx) {
        int64_t edge = grouped_edge_indices[idx];
        int64_t candidate = max_values[base + edge];
        if (candidate < value) {
            value = candidate;
        }
    }

    int64_t hm_offset = (static_cast<int64_t>(cp) * batch_size + batch) * table_size + head;
    int64_t old = hm_table[hm_offset];
    if (value < old) {
        hm_table[hm_offset] = value;
        atomicExch(changed_flag, 1);
    }
}

__global__ void hm2_scatter_zero_kernel(
    int64_t *hm_table,
    const int64_t *batch_indices,
    const int64_t *table_indices,
    int n_cp,
    int batch_size,
    int table_size,
    int num_points) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_cp * num_points;
    if (global_idx >= total) {
        return;
    }

    int point = global_idx % num_points;
    int cp = global_idx / num_points;

    int64_t batch = batch_indices[point];
    int64_t table = table_indices[point];
    int64_t offset = (static_cast<int64_t>(cp) * batch_size + batch) * table_size + table;
    hm_table[offset] = 0;
}

}  // namespace

bool hm2_fused_available() {
    return true;
}

void hm2_scatter_zero_cuda(
    torch::Tensor &hm_table,
    const torch::Tensor &batch_indices,
    const torch::Tensor &table_indices) {
    TORCH_CHECK(hm_table.is_cuda(), "hm_table must be CUDA tensor");
    TORCH_CHECK(batch_indices.is_cuda(), "batch_indices must be CUDA tensor");
    TORCH_CHECK(table_indices.is_cuda(), "table_indices must be CUDA tensor");

    TORCH_CHECK(hm_table.dtype() == torch::kInt64, "hm_table must be int64");
    TORCH_CHECK(batch_indices.dtype() == torch::kInt64, "batch_indices must be int64");
    TORCH_CHECK(table_indices.dtype() == torch::kInt64, "table_indices must be int64");

    TORCH_CHECK(hm_table.is_contiguous(), "hm_table must be contiguous");
    TORCH_CHECK(batch_indices.is_contiguous(), "batch_indices must be contiguous");
    TORCH_CHECK(table_indices.is_contiguous(), "table_indices must be contiguous");
    TORCH_CHECK(batch_indices.size(0) == table_indices.size(0), "index tensors must have same length");

    const int n_cp = static_cast<int>(hm_table.size(0));
    const int batch_size = static_cast<int>(hm_table.size(1));
    const int table_size = static_cast<int>(hm_table.size(2));
    const int num_points = static_cast<int>(batch_indices.size(0));

    if (num_points == 0 || n_cp == 0) {
        return;
    }

    const int threads = 256;
    const int total = n_cp * num_points;
    const int blocks = (total + threads - 1) / threads;

    hm2_scatter_zero_kernel<<<blocks, threads>>>(
        hm_table.data_ptr<int64_t>(),
        batch_indices.data_ptr<int64_t>(),
        table_indices.data_ptr<int64_t>(),
        n_cp,
        batch_size,
        table_size,
        num_points);
}

void hm2_fused_iteration_cuda(
    torch::Tensor &hm_table,
    torch::Tensor &max_values,
    const torch::Tensor &edge_costs_table_last,
    const torch::Tensor &edge_select_indices,
    const torch::Tensor &edge_offsets,
    const torch::Tensor &min_indices,
    const torch::Tensor &grouped_head_indices,
    const torch::Tensor &grouped_head_offsets,
    const torch::Tensor &grouped_edge_indices,
    torch::Tensor &changed_flag) {
    TORCH_CHECK(hm_table.is_cuda(), "hm_table must be CUDA tensor");
    TORCH_CHECK(max_values.is_cuda(), "max_values must be CUDA tensor");
    TORCH_CHECK(edge_costs_table_last.is_cuda(), "edge_costs_table_last must be CUDA tensor");
    TORCH_CHECK(edge_select_indices.is_cuda(), "edge_select_indices must be CUDA tensor");
    TORCH_CHECK(edge_offsets.is_cuda(), "edge_offsets must be CUDA tensor");
    TORCH_CHECK(min_indices.is_cuda(), "min_indices must be CUDA tensor");
    TORCH_CHECK(grouped_head_indices.is_cuda(), "grouped_head_indices must be CUDA tensor");
    TORCH_CHECK(grouped_head_offsets.is_cuda(), "grouped_head_offsets must be CUDA tensor");
    TORCH_CHECK(grouped_edge_indices.is_cuda(), "grouped_edge_indices must be CUDA tensor");
    TORCH_CHECK(changed_flag.is_cuda(), "changed_flag must be CUDA tensor");

    TORCH_CHECK(hm_table.dtype() == torch::kInt64, "hm_table must be int64");
    TORCH_CHECK(max_values.dtype() == torch::kInt64, "max_values must be int64");
    TORCH_CHECK(edge_costs_table_last.dtype() == torch::kInt64, "edge_costs_table_last must be int64");
    TORCH_CHECK(edge_select_indices.dtype() == torch::kInt64, "edge_select_indices must be int64");
    TORCH_CHECK(min_indices.dtype() == torch::kInt64, "min_indices must be int64");
    TORCH_CHECK(grouped_head_indices.dtype() == torch::kInt64, "grouped_head_indices must be int64");
    TORCH_CHECK(grouped_edge_indices.dtype() == torch::kInt64, "grouped_edge_indices must be int64");
    TORCH_CHECK(changed_flag.dtype() == torch::kInt32, "changed_flag must be int32");

    TORCH_CHECK(hm_table.is_contiguous(), "hm_table must be contiguous");
    TORCH_CHECK(max_values.is_contiguous(), "max_values must be contiguous");
    TORCH_CHECK(edge_costs_table_last.is_contiguous(), "edge_costs_table_last must be contiguous");
    TORCH_CHECK(edge_select_indices.is_contiguous(), "edge_select_indices must be contiguous");
    TORCH_CHECK(edge_offsets.is_contiguous(), "edge_offsets must be contiguous");
    TORCH_CHECK(min_indices.is_contiguous(), "min_indices must be contiguous");
    TORCH_CHECK(grouped_head_indices.is_contiguous(), "grouped_head_indices must be contiguous");
    TORCH_CHECK(grouped_head_offsets.is_contiguous(), "grouped_head_offsets must be contiguous");
    TORCH_CHECK(grouped_edge_indices.is_contiguous(), "grouped_edge_indices must be contiguous");
    TORCH_CHECK(changed_flag.is_contiguous(), "changed_flag must be contiguous");

    int n_cp = static_cast<int>(hm_table.size(0));
    int batch_size = static_cast<int>(hm_table.size(1));
    int table_size = static_cast<int>(hm_table.size(2));
    int num_edges = static_cast<int>(max_values.size(2));
    int num_grouped_heads = static_cast<int>(grouped_head_indices.size(0));

    int total = n_cp * batch_size * num_edges;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const bool profile_enabled = hm2_cuda_profile_enabled();
    cudaEvent_t start_event = nullptr;
    cudaEvent_t mid_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    if (profile_enabled) {
        cudaEventCreate(&start_event);
        cudaEventCreate(&mid_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event);
    }

    hm2_fused_max_add_kernel<<<blocks, threads>>>(
        hm_table.data_ptr<int64_t>(),
        max_values.data_ptr<int64_t>(),
        edge_costs_table_last.data_ptr<int64_t>(),
        edge_select_indices.data_ptr<int64_t>(),
        edge_offsets.data_ptr<int32_t>(),
        n_cp,
        batch_size,
        table_size,
        num_edges);

    if (profile_enabled) {
        cudaEventRecord(mid_event);
    }

    int grouped_total = n_cp * batch_size * num_grouped_heads;
    int grouped_blocks = (grouped_total + threads - 1) / threads;

    hm2_fused_grouped_amin_kernel<<<grouped_blocks, threads>>>(
        hm_table.data_ptr<int64_t>(),
        max_values.data_ptr<int64_t>(),
        grouped_head_indices.data_ptr<int64_t>(),
        grouped_head_offsets.data_ptr<int32_t>(),
        grouped_edge_indices.data_ptr<int64_t>(),
        n_cp,
        batch_size,
        table_size,
        num_edges,
        num_grouped_heads,
        changed_flag.data_ptr<int32_t>());

    if (profile_enabled) {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        float max_add_ms = 0.0f;
        float amin_ms = 0.0f;
        cudaEventElapsedTime(&max_add_ms, start_event, mid_event);
        cudaEventElapsedTime(&amin_ms, mid_event, stop_event);

        HM2CudaProfileStats &s = g_hm2_cuda_profile_stats;
        s.direct_calls++;
        s.direct_max_add_ms += static_cast<double>(max_add_ms);
        s.direct_amin_ms += static_cast<double>(amin_ms);
        maybe_log_hm2_cuda_profile();

        cudaEventDestroy(start_event);
        cudaEventDestroy(mid_event);
        cudaEventDestroy(stop_event);
    }
}

void hm2_fused_iteration_cuda_graph(
    torch::Tensor &hm_table,
    torch::Tensor &max_values,
    const torch::Tensor &edge_costs_table_last,
    const torch::Tensor &edge_select_indices,
    const torch::Tensor &edge_offsets,
    const torch::Tensor &min_indices,
    const torch::Tensor &grouped_head_indices,
    const torch::Tensor &grouped_head_offsets,
    const torch::Tensor &grouped_edge_indices,
    torch::Tensor &changed_flag) {
    TORCH_CHECK(hm_table.is_cuda(), "hm_table must be CUDA tensor");

    int n_cp = static_cast<int>(hm_table.size(0));
    int batch_size = static_cast<int>(hm_table.size(1));
    int table_size = static_cast<int>(hm_table.size(2));
    int num_edges = static_cast<int>(max_values.size(2));
    int num_grouped_heads = static_cast<int>(grouped_head_indices.size(0));

    int64_t *hm_ptr = hm_table.data_ptr<int64_t>();
    int64_t *max_ptr = max_values.data_ptr<int64_t>();
    int64_t *cost_ptr = edge_costs_table_last.data_ptr<int64_t>();
    int64_t *select_ptr = edge_select_indices.data_ptr<int64_t>();
    int32_t *offset_ptr = edge_offsets.data_ptr<int32_t>();
    int64_t *head_ptr = min_indices.data_ptr<int64_t>();
    int64_t *grouped_head_ptr = grouped_head_indices.data_ptr<int64_t>();
    int32_t *grouped_head_offsets_ptr = grouped_head_offsets.data_ptr<int32_t>();
    int64_t *grouped_edge_ptr = grouped_edge_indices.data_ptr<int64_t>();
    int32_t *changed_ptr = changed_flag.data_ptr<int32_t>();

    bool same_signature =
        g_fused_graph_cache.exec &&
        g_fused_graph_cache.hm_ptr == hm_ptr &&
        g_fused_graph_cache.max_ptr == max_ptr &&
        g_fused_graph_cache.cost_ptr == cost_ptr &&
        g_fused_graph_cache.select_ptr == select_ptr &&
        g_fused_graph_cache.offset_ptr == offset_ptr &&
        g_fused_graph_cache.head_ptr == head_ptr &&
        g_fused_graph_cache.grouped_head_ptr == grouped_head_ptr &&
        g_fused_graph_cache.grouped_head_offsets_ptr == grouped_head_offsets_ptr &&
        g_fused_graph_cache.grouped_edge_ptr == grouped_edge_ptr &&
        g_fused_graph_cache.changed_ptr == changed_ptr &&
        g_fused_graph_cache.n_cp == n_cp &&
        g_fused_graph_cache.batch == batch_size &&
        g_fused_graph_cache.table == table_size &&
        g_fused_graph_cache.edges == num_edges &&
        g_fused_graph_cache.grouped_heads == num_grouped_heads;

    int total = n_cp * batch_size * num_edges;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    int grouped_total = n_cp * batch_size * num_grouped_heads;
    int grouped_blocks = (grouped_total + threads - 1) / threads;

    const bool profile_enabled = hm2_cuda_profile_enabled();

    if (!same_signature) {
        const auto capture_start = std::chrono::steady_clock::now();
        clear_graph_cache(g_fused_graph_cache);

        cudaStream_t stream = cudaStreamPerThread;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        hm2_fused_max_add_kernel<<<blocks, threads, 0, stream>>>(
            hm_ptr,
            max_ptr,
            cost_ptr,
            select_ptr,
            offset_ptr,
            n_cp,
            batch_size,
            table_size,
            num_edges);

        hm2_fused_grouped_amin_kernel<<<grouped_blocks, threads, 0, stream>>>(
            hm_ptr,
            max_ptr,
            grouped_head_ptr,
            grouped_head_offsets_ptr,
            grouped_edge_ptr,
            n_cp,
            batch_size,
            table_size,
            num_edges,
            num_grouped_heads,
            changed_ptr);

        cudaStreamEndCapture(stream, &g_fused_graph_cache.graph);
        cudaGraphInstantiate(&g_fused_graph_cache.exec, g_fused_graph_cache.graph, nullptr, nullptr, 0);

        g_fused_graph_cache.hm_ptr = hm_ptr;
        g_fused_graph_cache.max_ptr = max_ptr;
        g_fused_graph_cache.cost_ptr = cost_ptr;
        g_fused_graph_cache.select_ptr = select_ptr;
        g_fused_graph_cache.offset_ptr = offset_ptr;
        g_fused_graph_cache.head_ptr = head_ptr;
        g_fused_graph_cache.grouped_head_ptr = grouped_head_ptr;
        g_fused_graph_cache.grouped_head_offsets_ptr = grouped_head_offsets_ptr;
        g_fused_graph_cache.grouped_edge_ptr = grouped_edge_ptr;
        g_fused_graph_cache.changed_ptr = changed_ptr;
        g_fused_graph_cache.n_cp = n_cp;
        g_fused_graph_cache.batch = batch_size;
        g_fused_graph_cache.table = table_size;
        g_fused_graph_cache.edges = num_edges;
        g_fused_graph_cache.grouped_heads = num_grouped_heads;

        if (profile_enabled) {
            const auto capture_end = std::chrono::steady_clock::now();
            HM2CudaProfileStats &s = g_hm2_cuda_profile_stats;
            s.graph_recaptures++;
            s.graph_capture_ms += std::chrono::duration<double, std::milli>(capture_end - capture_start).count();
        }
    }

    if (profile_enabled) {
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, cudaStreamPerThread);
        cudaGraphLaunch(g_fused_graph_cache.exec, cudaStreamPerThread);
        cudaEventRecord(stop_event, cudaStreamPerThread);
        cudaEventSynchronize(stop_event);
        float launch_ms = 0.0f;
        cudaEventElapsedTime(&launch_ms, start_event, stop_event);

        HM2CudaProfileStats &s = g_hm2_cuda_profile_stats;
        s.graph_calls++;
        s.graph_launch_ms += static_cast<double>(launch_ms);
        maybe_log_hm2_cuda_profile();

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    } else {
        cudaGraphLaunch(g_fused_graph_cache.exec, cudaStreamPerThread);
    }
}

}  // namespace hm2_torch
