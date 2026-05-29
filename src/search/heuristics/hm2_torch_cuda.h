#pragma once

#include <torch/torch.h>

namespace hm2_torch {

bool hm2_fused_available();

void hm2_scatter_zero_cuda(
    torch::Tensor &hm_table,
    const torch::Tensor &batch_indices,
    const torch::Tensor &table_indices);

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
    torch::Tensor &changed_flag);

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
    torch::Tensor &changed_flag);

}  // namespace hm2_torch
