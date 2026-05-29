#include "hm2_torch.h"

#include "../plugins/plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cassert>
#include <limits>
#include <set>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <map>

using namespace std;
using namespace torch::indexing;

#include <stdexcept>
#include <string>

namespace hm2_torch {
namespace {

struct HM2PhaseProfileStats {
    uint64_t calls = 0;
    uint64_t sampled_calls = 0;
    double total_ms = 0.0;
    double init_ms = 0.0;
    double update_ms = 0.0;
    double eval_ms = 0.0;
};

HM2PhaseProfileStats g_hm2_phase_profile_stats;

bool hm2_phase_profile_enabled() {
    static bool enabled = []() {
        const char *value = std::getenv("HM2_TORCH_PROFILE");
        return value && std::string(value) != "0";
    }();
    return enabled;
}

void maybe_log_hm2_phase_profile(const size_t batch_size, const int table_size, const int n_cp) {
    HM2PhaseProfileStats &s = g_hm2_phase_profile_stats;
    if (s.sampled_calls == 0 || (s.sampled_calls % 100) != 0) {
        return;
    }
    const double denom = static_cast<double>(s.sampled_calls);
    utils::g_log << "[hm2_profile] calls=" << s.calls
                 << " sampled=" << s.sampled_calls
                 << " avg_total_ms=" << (s.total_ms / denom)
                 << " avg_init_ms=" << (s.init_ms / denom)
                 << " avg_update_ms=" << (s.update_ms / denom)
                 << " avg_eval_ms=" << (s.eval_ms / denom)
                 << " batch=" << batch_size
                 << " table=" << table_size
                 << " n_cp=" << n_cp
                 << endl;
}

}  // namespace

HM2Torch::HM2Torch(
    int m,
    int cp,
    int n_cp,
    bool use_gpu,
    int max_batch_size_cap,
    const shared_ptr<AbstractTask> &transform,
    bool cache_estimates,
    const string &description,
    utils::Verbosity verbosity
    )
    : HM2Base(m, cp, n_cp, use_gpu, transform, cache_estimates, description, verbosity),
      max_indices_(),
      edge_costs_(),
            edge_costs_table_last_(),
            edge_offsets_(),
            edge_select_indices_(),
      min_indices_(),
    grouped_head_indices_(),
    grouped_head_offsets_(),
    grouped_edge_indices_(),
      goal_indices_(),
      max_batch_size_(1),
            max_batch_size_cap_(max_batch_size_cap),
      iteration_count_(0),
    init_buffer_slot_(0),
      hyperedges_(get_hyperedges(m)),
      device_(get_device(use_gpu)),
      int_options_(torch::TensorOptions().dtype(torch::kLong).requires_grad(false).device(get_device(use_gpu))),
      bool_options_(torch::TensorOptions().dtype(torch::kBool).requires_grad(false).device(get_device(use_gpu))) {
        static std::once_flag torch_threads_once;
        std::call_once(torch_threads_once, []() {
                torch::set_num_threads(1);
                torch::set_num_interop_threads(1);
        });

    vector<int> select_indices_vector;
    vector<int> max_indices_vector;
    vector<int> min_indices_vector;
    vector<int> grouped_head_indices_vector;
    vector<int> grouped_head_offsets_vector;
    vector<int64_t> grouped_edge_indices_vector;
    vector<int> goal_indices_vector;
    vector<int> edge_offsets_vector;
    vector<int64_t> edge_select_indices_vector;
    edge_offsets_vector.push_back(0);



    for (size_t edge_index = 0; edge_index < hyperedges_.size(); ++edge_index) {
        const auto &edge = hyperedges_[edge_index];
        select_indices_vector.insert(select_indices_vector.end(), edge.tail.begin(), edge.tail.end());
        max_indices_vector.insert(max_indices_vector.end(), edge.tail.size(), static_cast<int>(edge_index));
        for (uint32_t idx : edge.tail) {
            edge_select_indices_vector.push_back(static_cast<int64_t>(idx));
        }
        edge_offsets_vector.push_back(static_cast<int>(edge_select_indices_vector.size()));
        min_indices_vector.push_back(edge.head);
    }

    grouped_head_offsets_vector.push_back(0);
    if (!min_indices_vector.empty()) {
        map<int, vector<int>> head_to_edges;
        for (size_t edge_index = 0; edge_index < min_indices_vector.size(); ++edge_index) {
            head_to_edges[min_indices_vector[edge_index]].push_back(static_cast<int>(edge_index));
        }
        for (const auto &entry : head_to_edges) {
            grouped_head_indices_vector.push_back(entry.first);
            for (int edge_index : entry.second) {
                grouped_edge_indices_vector.push_back(static_cast<int64_t>(edge_index));
            }
            grouped_head_offsets_vector.push_back(static_cast<int>(grouped_edge_indices_vector.size()));
        }
    }

    for (const auto &goal_index : get_goal_indices()) {
        goal_indices_vector.emplace_back(static_cast<int>(goal_index));
    }
    select_indices_ = torch::tensor(select_indices_vector, int_options_);

    max_indices_ = torch::tensor(max_indices_vector, int_options_);
    min_indices_ = torch::tensor(min_indices_vector, int_options_);
    grouped_head_indices_ = torch::tensor(grouped_head_indices_vector, int_options_);
    grouped_edge_indices_ = torch::tensor(grouped_edge_indices_vector, int_options_);
    grouped_head_offsets_ = torch::tensor(grouped_head_offsets_vector, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false).device(device_));
    goal_indices_ = torch::tensor(goal_indices_vector, int_options_);
    edge_select_indices_ = torch::tensor(edge_select_indices_vector, int_options_);
    edge_offsets_ = torch::tensor(edge_offsets_vector, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false).device(device_));
    edge_costs_vector_ = edge_costs_edge_major_;
    if (cp_ != hm2_base::CostPartition::GOAL_FACTS_INTERNALLY) {
        if (edge_costs_edge_major_.empty()) {
            edge_costs_ = torch::tensor(edge_costs_edge_major_, int_options_).view({0, n_cp_, 1});
            edge_costs_table_last_ = torch::tensor(edge_costs_cp_major_, int_options_).view({n_cp_, 1, 0});
        } else {
            int num_edges = static_cast<int>(edge_costs_edge_major_.size()) / n_cp_;
            edge_costs_ = torch::tensor(edge_costs_edge_major_, int_options_).view({num_edges, n_cp_, 1});
            edge_costs_table_last_ = torch::tensor(edge_costs_cp_major_, int_options_).view({n_cp_, 1, num_edges});
        }
    } else {
        edge_costs_vector_ = edge_costs_edge_major_;
    }
    const bool goal_facts_internal = (cp_ == hm2_base::CostPartition::GOAL_FACTS_INTERNALLY);
    int num_edges = 0;
    if (goal_facts_internal) {
        num_edges = static_cast<int>(hyperedges_.size());
    } else {
        num_edges = edge_costs_edge_major_.size() / n_cp_;
    }

    // For GOAL_FACTS_INTERNALLY, n_cp_ can be as large as num_operators_ + 1.
    // Avoid allocating O(n_cp_ * table_size_) buffers here; we allocate per-chunk
    // buffers inside get_goal_facts_h_values_per_cost().
    const int alloc_n_cp = goal_facts_internal ? 1 : n_cp_;

    hm_table_ = torch::zeros({alloc_n_cp, max_batch_size_, table_size_}, int_options_);

    all_values_ = torch::zeros({alloc_n_cp, max_batch_size_, select_indices_.size(0)}, int_options_);
    max_values_ = torch::zeros({alloc_n_cp, max_batch_size_, num_edges}, int_options_);
    last_heuristic_table_ = torch::zeros_like(hm_table_);
    less_than_table_ = torch::zeros_like(hm_table_, bool_options_);
    any_scalar_ = torch::scalar_tensor(true, bool_options_);
    device_changed_flag_ = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).requires_grad(false).device(device_));

    if (cp == 3) {
        State initial_state = task_proxy.get_initial_state();
        optimize_cp(initial_state);
    }
}

#ifdef DEBUG
static void print_tensor_shape(torch::Tensor t) {
    utils::g_log << "shape: ";
    for (int i = 0; i < t.dim(); i++) {
        utils::g_log << t.size(i) << ", ";
    }
    utils::g_log << endl;
}
#endif

torch::DeviceType HM2Torch::get_device(bool use_gpu) const {
    if (use_gpu) {
        cout << "Using GPU" << torch::kCUDA << endl;
        return torch::kCUDA;
    } else {
        return torch::kCPU;
    }
}

vector<uint32_t> HM2Torch::get_goal_indices() {
    vector<uint32_t> goal_indices;
    if (m >= 1) {
        for (size_t i = 0; i < goals.size(); ++i) {
            FactPair goal = goals[i];
            int idx1 = variable_offsets[goal.var] + goal.value;
            goal_indices.push_back(idx1);
            if (m >= 2) {
                for (size_t j = i + 1; j < goals.size(); ++j) {
                    FactPair goal2 = goals[j];
                    goal_indices.push_back(getGlobalIndex(goal, goal2));
                }
            }
        }
    }

    return goal_indices;
}

unique_ptr<hm2_base::HM2Base> HM2Torch::get_base_heuristic(hm2_base::CostPartition &cp, int num_cost_partitions) {
    return make_unique<HM2Torch>(m, static_cast<int>(cp), num_cost_partitions, use_gpu_, max_batch_size_cap_, task, does_cache_estimates(), "test?", utils::Verbosity::SILENT);
}



void HM2Torch::print_torch_vector(torch::Tensor& t) {
    Tuple all_facts = iterate_atoms(0);
    for (size_t i = 0; i < all_facts.size(); i++) {
        FactPair fact = all_facts[i];
        int idx = variable_offsets[fact.var] + fact.value;

        auto v = t[0][0][idx].item<int>();
        cout << "h([" << fact.var << "=" << fact.value << "]) = " << v << endl;
    }
    for (size_t i = 0; i < all_facts.size(); i++) {
        FactPair fact1 = all_facts[i];
        Tuple all_facts = iterate_atoms(i + 1);
        for (size_t j = 0; j < all_facts.size(); j++) {
            FactPair fact2 = all_facts[j];
            int pair_idx = getGlobalIndex(fact1, fact2);
            auto v = t[0][0][pair_idx].item<int>();
            cout << "h([" << fact1.var << "=" << fact1.value << ", " << fact2.var << "=" << fact2.value << "]) = " << v << endl;
        }
    }
    cout << endl;
}

vector<int> HM2Torch::compute_heuristic(const vector<State> &ancestor_states) {
    //batched version
    // const State &state = ancestor_states[0];

    if (max_batch_size_cap_ > 0 && ancestor_states.size() > static_cast<size_t>(max_batch_size_cap_)) {
        vector<int> result;
        result.reserve(ancestor_states.size());
        const size_t cap = static_cast<size_t>(max_batch_size_cap_);
        for (size_t start = 0; start < ancestor_states.size(); start += cap) {
            size_t end = std::min(start + cap, ancestor_states.size());
            vector<State> chunk;
            chunk.reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                chunk.push_back(ancestor_states[i]);
            }
            vector<int> chunk_result = HM2Torch::compute_heuristic(chunk);
            result.insert(result.end(), chunk_result.begin(), chunk_result.end());
        }
        return result;
    }

    torch::NoGradGuard no_grad;
    const bool profile_enabled = hm2_phase_profile_enabled();
    const auto total_start = std::chrono::steady_clock::now();

    size_t batch_size = ancestor_states.size();
    vector<bool> is_goal;
    vector<Tuple> tuples;
    is_goal.reserve(batch_size);
    tuples.reserve(batch_size);
    for (const State &ancestor_state : ancestor_states) {
        State state = convert_ancestor_state(ancestor_state);
        is_goal.emplace_back(task_properties::is_goal_state(task_proxy, state));
        tuples.emplace_back(task_properties::get_fact_pairs(state));
    }
    if (device_ == torch::kCUDA) {
        // GPU-native initialization happens below directly into hm_slice.
    } else {
        init_hm_table(tuples);
    }

    int table_size = table_size_;

    const long bs = static_cast<long>(batch_size);
    const long num_edges = edge_costs_table_last_.defined() ? edge_costs_table_last_.size(2)
                                                           : static_cast<long>(hyperedges_.size());

    if (device_ == torch::kCUDA) {
        // CUDA kernels require contiguous tensors. Narrowing a "capacity" tensor along
        // the batch dimension produces a non-contiguous view unless batch == capacity.
        // Allocate working tensors to the *exact* batch size on CUDA.
        if (max_batch_size_ != static_cast<int>(batch_size) || !hm_table_.defined() || hm_table_.size(1) != bs) {
            max_batch_size_ = static_cast<int>(batch_size);
            hm_table_ = torch::zeros({n_cp_, bs, table_size}, int_options_);
            all_values_ = torch::zeros({n_cp_, bs, select_indices_.size(0)}, int_options_);
            max_values_ = torch::zeros({n_cp_, bs, num_edges}, int_options_);
            last_heuristic_table_ = torch::zeros_like(hm_table_);
            less_than_table_ = torch::zeros_like(hm_table_, bool_options_);
        }
    } else {
        if (static_cast<int>(batch_size) > max_batch_size_) {
            int bucket_size = 1;
            while (bucket_size < static_cast<int>(batch_size)) {
                bucket_size <<= 1;
            }
            max_batch_size_ = bucket_size;
            hm_table_ = torch::zeros({n_cp_, max_batch_size_, table_size}, int_options_);
            all_values_ = torch::zeros({n_cp_, max_batch_size_, select_indices_.size(0)}, int_options_);
            max_values_ = torch::zeros({n_cp_, max_batch_size_, num_edges}, int_options_);
            last_heuristic_table_ = torch::zeros({n_cp_, max_batch_size_, table_size}, int_options_);
            less_than_table_ = torch::zeros({n_cp_, max_batch_size_, table_size}, bool_options_);
        }
    }

    torch::Tensor hm_slice = (device_ == torch::kCUDA)
        ? hm_table_
        : hm_table_.narrow(1, 0, bs);

    const auto init_start = std::chrono::steady_clock::now();
    if (device_ == torch::kCUDA) {
        init_hm_table_cuda(tuples, hm_slice);
    } else {
        const auto cpu_int_options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false).device(torch::kCPU);
        torch::Tensor hm_base = torch::tensor(hm_table, cpu_int_options).view({table_size, static_cast<long>(batch_size)});
        hm_slice.copy_(hm_base.transpose(0, 1).unsqueeze(0).expand({n_cp_, static_cast<long>(batch_size), table_size}));
    }
    const auto init_end = std::chrono::steady_clock::now();

    torch::Tensor all_values = (device_ == torch::kCUDA)
        ? all_values_
        : all_values_.narrow(1, 0, bs);
    torch::Tensor max_values = (device_ == torch::kCUDA)
        ? max_values_
        : max_values_.narrow(1, 0, bs);
    torch::Tensor last_heuristic_table = (device_ == torch::kCUDA)
        ? last_heuristic_table_
        : last_heuristic_table_.narrow(1, 0, bs);
    torch::Tensor less_than_table = (device_ == torch::kCUDA)
        ? less_than_table_
        : less_than_table_.narrow(1, 0, bs);

    const auto update_start = std::chrono::steady_clock::now();
    update_hm_table(
        hm_slice,
        all_values,
        max_values,
        last_heuristic_table,
        less_than_table,
        any_scalar_
        );
    const auto update_end = std::chrono::steady_clock::now();



    const auto eval_start = std::chrono::steady_clock::now();
    vector<int> h_list = eval_tensor(hm_slice);
    const auto eval_end = std::chrono::steady_clock::now();
    for (size_t i = 0; i < h_list.size(); i++) {
        int h = h_list[i];
        if (h == numeric_limits<int>::max()) {
            h_list[i] = DEAD_END;
        }
        if (is_goal[i]) {
            h_list[i] = 0; //TODO: ...
            // 1) heuristic should be 0 if state is goal state
            // 2) do this check initially, store indices and avoid computation.
        }
    }

    if (profile_enabled) {
        HM2PhaseProfileStats &s = g_hm2_phase_profile_stats;
        s.calls++;
        s.sampled_calls++;
        s.total_ms += std::chrono::duration<double, std::milli>(eval_end - total_start).count();
        s.init_ms += std::chrono::duration<double, std::milli>(init_end - init_start).count();
        s.update_ms += std::chrono::duration<double, std::milli>(update_end - update_start).count();
        s.eval_ms += std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
        maybe_log_hm2_phase_profile(batch_size, table_size, n_cp_);
    }

    return h_list;

    //if (h == numeric_limits<int>::max())
    //    return DEAD_END;
    //return h;
}

torch::Tensor HM2Torch::optimize_cp(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);

    int batch_size = 1;
    int table_size = table_size_;

    Tuple s_tup = task_properties::get_fact_pairs(state);
    init_hm_table(s_tup);

    torch::Tensor original_edge_costs = edge_costs_.sum(1).view({1, 1, -1});
    torch::Tensor edge_costs = torch::rand_like(edge_costs_, int_options_);
    torch::optim::Adam optimizer({edge_costs}, torch::optim::AdamOptions(0.01));
    int h = compute_heuristic(ancestor_state);
    cout << "h: " << h << endl;

    for (int batch_id = 0; batch_id < 100; batch_id++) {
        //torch::Tensor costs = edge_costs.softmax(1) * original_edge_costs; // NOTE: softmax is used to ensure that sum is 1 for all cp's.
        torch::Tensor costs = edge_costs * original_edge_costs; // NOTE: softmax is used to ensure that sum is 1 for all cp's.
        torch::Tensor hm_slice = torch::tensor(hm_table, int_options_).view({batch_size, 1, table_size}).repeat({1, n_cp_, 1});   //TODO: I switched bs and table_size dim. Compare perfromance!
        last_heuristic_table_ = torch::zeros_like(hm_slice);
        torch::Tensor all_values = all_values_.index({Slice(0, 1), Slice(), Slice()}).to(torch::kDouble);
        torch::Tensor max_values = max_values_.index({Slice(0, 1), Slice(), Slice()}).to(torch::kDouble);
        torch::Tensor last_heuristic_table = last_heuristic_table_.index({Slice(0, 1), Slice(), Slice()});
        torch::Tensor less_than_table = less_than_table_.index({Slice(0, 1), Slice(), Slice()});
        was_updated = true;
        int round = 0;
        do {
            ++round;
            was_updated = false;
            bool check_fixpoint = round >= iteration_count_;

            if (check_fixpoint) {
                torch::copy_out(last_heuristic_table, last_heuristic_table, hm_slice);  // Used to detect whether we have converged.
            }
            all_values = hm_slice.index_select(2, select_indices_.clone());
            max_values = max_values.index_reduce(2, max_indices_.clone(), all_values, "amax", false);
            max_values = max_values + costs * 1;
            hm_slice = hm_slice.index_reduce(2, min_indices_.clone(), max_values, "amin", true);

            if (check_fixpoint) {
                last_heuristic_table = last_heuristic_table - hm_slice;
                less_than_table = last_heuristic_table < -1E-8;
                any_scalar_ = less_than_table.any();
                was_updated = any_scalar_.detach().item<bool>();
            } else {
                was_updated = true;
            }
        } while (was_updated);

        torch::Tensor goal_tensor = get<0>(hm_slice.index_select(2, goal_indices_).max(2));
        goal_tensor = goal_tensor * -1;
        goal_tensor = goal_tensor.sum();

        goal_tensor = goal_tensor + costs.index({0, 0, -1});
        cout << "current h(init): " << goal_tensor << endl;

        optimizer.zero_grad();
        goal_tensor.backward();
        optimizer.step();
    }
    edge_costs_ = (1 * edge_costs.softmax(1) * original_edge_costs).detach().to(torch::kInt64).view({1, n_cp_, -1});

    return edge_costs_;
}

int HM2Torch::compute_heuristic(const State &ancestor_state) {
    return compute_heuristic(vector<State>{ancestor_state})[0];
}

void HM2Torch::update_hm_table(
    torch::Tensor& hm_table,
    torch::Tensor& all_values,
    torch::Tensor& max_values,
    torch::Tensor& last_heuristic_table,
    torch::Tensor& less_than_table,
    torch::Tensor& any_scalar
    ) {
    const int fixpoint_sync_stride = 4;
    int round = 0;
    do {
        ++round;
        was_updated = false;
        bool check_fixpoint = round >= iteration_count_;
        bool should_sync_fixpoint = check_fixpoint &&
            ((round - iteration_count_) % fixpoint_sync_stride == 0);

        bool used_fused = false;
#ifdef HM2_TORCH_HAS_CUDA_FUSED_KERNEL
        if (device_ == torch::kCUDA && hm2_fused_available() && edge_offsets_.defined() && edge_select_indices_.defined()) {
            if (check_fixpoint) {
                device_changed_flag_.fill_(0);
            }
            // Defensive: fused CUDA kernels require contiguous tensors.
            torch::Tensor hm_table_contig = hm_table;
            torch::Tensor max_values_contig = max_values;
            const bool need_copy_back = !hm_table_contig.is_contiguous() || !max_values_contig.is_contiguous();
            if (!hm_table_contig.is_contiguous()) {
                hm_table_contig = hm_table.contiguous();
            }
            if (!max_values_contig.is_contiguous()) {
                max_values_contig = max_values.contiguous();
            }
            hm2_fused_iteration_cuda_graph(
                hm_table_contig,
                max_values_contig,
                edge_costs_table_last_,
                edge_select_indices_,
                edge_offsets_,
                min_indices_,
                grouped_head_indices_,
                grouped_head_offsets_,
                grouped_edge_indices_,
                device_changed_flag_);
            if (need_copy_back) {
                hm_table.copy_(hm_table_contig);
                max_values.copy_(max_values_contig);
            }
            used_fused = true;
        }
#endif
        if (!used_fused) {
            if (should_sync_fixpoint) {
                torch::copy_out(last_heuristic_table, last_heuristic_table, hm_table);  // Used to detect whether we have converged.
            }
            torch::index_select_out(all_values, hm_table, 2, select_indices_);

            torch::index_reduce_out(max_values, max_values, 2, max_indices_, all_values, "amax",
                                    false);  // No need to reset 'max_values' since we exclude self.

            assert(edge_costs_table_last_.defined());
            torch::add_out(max_values, max_values, edge_costs_table_last_);

            torch::index_reduce_out(hm_table, hm_table, 2, min_indices_, max_values, "amin", true);
        }

        if (used_fused) {
            if (check_fixpoint) {
                was_updated = device_changed_flag_.item<int>() != 0;
                if (!was_updated) {
                    iteration_count_ = round;
                }
            } else {
                was_updated = true;
            }
        } else {
            if (should_sync_fixpoint) {
                torch::sub_out(last_heuristic_table, hm_table, last_heuristic_table);
                torch::lt_out(less_than_table, last_heuristic_table, -1E-8);
                torch::any_out(any_scalar, less_than_table);
                was_updated = any_scalar.item<bool>();
                if (!was_updated) {
                    iteration_count_ = round;
                }
            } else {
                was_updated = true;
            }
        }
    } while (was_updated);

    if (cp_ == hm2_base::CostPartition::GOAL_FACTS_INTERNALLY) {
        hm_table_ = hm_table;
    }
}

void HM2Torch::init_hm_table_cuda(const vector<Tuple> &t_list, torch::Tensor &hm_slice) {
    // CUDA init kernel requires contiguous hm_table.
    const bool hm_is_contiguous = hm_slice.is_contiguous();
    torch::Tensor hm_target = hm_is_contiguous ? hm_slice : hm_slice.contiguous();
    hm_target.fill_(numeric_limits<int>::max());

    vector<int64_t> batch_indices;
    vector<int64_t> table_indices;
    batch_indices.reserve(t_list.size() * 64);
    table_indices.reserve(t_list.size() * 64);

    for (size_t tuple_idx = 0; tuple_idx < t_list.size(); ++tuple_idx) {
        if (has_op_with_empty_preconds) {
            batch_indices.push_back(static_cast<int64_t>(tuple_idx));
            table_indices.push_back(static_cast<int64_t>(dummy_precond_node_id));
        }

        const Tuple &t = t_list[tuple_idx];
        for (size_t i = 0; i < t.size(); ++i) {
            FactPair atom = t[i];
            int atom_idx1 = variable_offsets[atom.var] + atom.value;
            batch_indices.push_back(static_cast<int64_t>(tuple_idx));
            table_indices.push_back(static_cast<int64_t>(atom_idx1));
            if (m >= 2) {
                for (size_t j = i + 1; j < t.size(); ++j) {
                    FactPair atom2 = t[j];
                    int pair_idx = getGlobalIndex(atom, atom2);
                    batch_indices.push_back(static_cast<int64_t>(tuple_idx));
                    table_indices.push_back(static_cast<int64_t>(pair_idx));
                }
            }
        }
    }

    if (!batch_indices.empty()) {
        int slot = init_buffer_slot_;
        init_buffer_slot_ = 1 - init_buffer_slot_;

        long needed_size = static_cast<long>(batch_indices.size());
        auto pinned_cpu_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

        if (!init_batch_idx_pinned_[slot].defined() || init_batch_idx_pinned_[slot].size(0) < needed_size) {
            init_batch_idx_pinned_[slot] = torch::empty({needed_size}, pinned_cpu_opts);
        }
        if (!init_table_idx_pinned_[slot].defined() || init_table_idx_pinned_[slot].size(0) < needed_size) {
            init_table_idx_pinned_[slot] = torch::empty({needed_size}, pinned_cpu_opts);
        }

        torch::Tensor batch_idx_cpu = init_batch_idx_pinned_[slot].narrow(0, 0, needed_size);
        torch::Tensor table_idx_cpu = init_table_idx_pinned_[slot].narrow(0, 0, needed_size);

        std::memcpy(batch_idx_cpu.data_ptr<int64_t>(), batch_indices.data(), batch_indices.size() * sizeof(int64_t));
        std::memcpy(table_idx_cpu.data_ptr<int64_t>(), table_indices.data(), table_indices.size() * sizeof(int64_t));

        if (!init_batch_idx_device_[slot].defined() || init_batch_idx_device_[slot].size(0) < needed_size) {
            init_batch_idx_device_[slot] = torch::empty({needed_size}, int_options_);
        }
        if (!init_table_idx_device_[slot].defined() || init_table_idx_device_[slot].size(0) < needed_size) {
            init_table_idx_device_[slot] = torch::empty({needed_size}, int_options_);
        }

        torch::Tensor batch_idx = init_batch_idx_device_[slot].narrow(0, 0, needed_size);
        torch::Tensor table_idx = init_table_idx_device_[slot].narrow(0, 0, needed_size);
        batch_idx.copy_(batch_idx_cpu, true);
        table_idx.copy_(table_idx_cpu, true);

    #ifdef HM2_TORCH_HAS_CUDA_FUSED_KERNEL
            hm2_scatter_zero_cuda(hm_target, batch_idx, table_idx);
    #else
            hm_target.index_put_({Slice(), batch_idx, table_idx}, 0);
    #endif
    }

        if (!hm_is_contiguous) {
            hm_slice.copy_(hm_target);
        }
}



vector<int> HM2Torch::eval_tensor(torch::Tensor& hm_slice) const {
    #ifdef DEBUG
    print_tensor_shape(hm_slice);
    #endif
    const torch::Tensor original_cost = hm_slice.index({-1, Slice(), Slice()});
    const torch::Tensor original_max_goal_tensor =
        get<0>(original_cost.index_select(1, goal_indices_).max(1));

    size_t batch_size = hm_slice.size(1);

    torch::Tensor final_goal_tensor;
    if (n_cp_ == 1) {
        final_goal_tensor = original_max_goal_tensor;
    } else {
        torch::Tensor tmp = get<0>(hm_slice.index({Slice(0, -1), Slice(), Slice()}).index_select(2, goal_indices_).max(2));
        torch::Tensor max_goal_tensor = tmp.sum(0);
        final_goal_tensor = torch::maximum(max_goal_tensor, original_max_goal_tensor);
    }
    torch::Tensor final_goal_tensor_cpu = final_goal_tensor.to(torch::kCPU);

    vector<int> max_goals(batch_size);
    auto final_goal_tensor_accessor = final_goal_tensor_cpu.accessor<long, 1>();
    for (size_t i = 0; i < batch_size; i++) {
        long value = final_goal_tensor_accessor[i];
        max_goals[i] = (value <= numeric_limits<int>::max()) ? value : numeric_limits<int>::max();
    }

    return max_goals;
}

vector<vector<int>> HM2Torch::get_goal_facts_h_values_per_cost(State &state) {

    const int original_n_cp = n_cp_;

    const int num_edges_total = static_cast<int>(hyperedges_.size());
    const int64_t select_count = select_indices_.defined() ? select_indices_.size(0) : 0;
    const int64_t table_size = table_size_;

    // Estimate bytes per cost function for the per-chunk working set.
    // We allocate (int64) hm_table + all_values + max_values + edge_costs_table_last.
    // all_values dominates when select_count is large.
    const int64_t bytes_per_cp = 8 * (table_size + select_count + 2LL * num_edges_total);

    // Cap memory used by these per-chunk tensors. Default: 512 MiB.
    int64_t cap_mb = 512;
    if (const char *env = std::getenv("HM2_TORCH_GOALFACTS_PRECOMP_MB")) {
        try {
            cap_mb = std::stoll(env);
        } catch (...) {
        }
    }
    if (cap_mb < 64) {
        cap_mb = 64;
    }
    const int64_t cap_bytes = cap_mb * 1024LL * 1024LL;

    int amount = 300;
    if (const char *env = std::getenv("HM2_TORCH_GOALFACTS_CHUNK")) {
        try {
            amount = std::stoi(env);
        } catch (...) {
        }
    } else if (bytes_per_cp > 0) {
        const int64_t max_cp_by_cap = cap_bytes / bytes_per_cp;
        if (max_cp_by_cap > 0) {
            amount = static_cast<int>(std::min<int64_t>(amount, max_cp_by_cap));
        } else {
            amount = 1;
        }
    }
    amount = std::max(1, amount);
    const int num_iterations = (original_n_cp + amount - 1) / amount;

    torch::DeviceType device = get_device(use_gpu_);
    const auto int_options_ = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false).device(device);

    const torch::Tensor hm_table_saved = hm_table_;
    const torch::Tensor all_values_saved = all_values_;
    const torch::Tensor max_values_saved = max_values_;
    const torch::Tensor last_heuristic_table_saved = last_heuristic_table_;
    const torch::Tensor less_than_table_saved = less_than_table_;
    const torch::Tensor edge_costs_saved = edge_costs_;
    const torch::Tensor edge_costs_table_last_saved = edge_costs_table_last_;

    vector<torch::Tensor> tensors;
    tensors.reserve(num_iterations);

    const bool implicit_goal_facts_internal =
        (cp_ == hm2_base::CostPartition::GOAL_FACTS_INTERNALLY) && edge_costs_vector_.empty();

    for (int i = 0; i < num_iterations; i++) {
        const int start = min(i * amount, original_n_cp);
        const int end = min(start + amount, original_n_cp);
        n_cp_ = end - start;
        if (n_cp_ <= 0) {
            continue;
        }

        const int num_edges = num_edges_total;
        vector<int> edge_costs_vector;
        vector<int> edge_costs_table_last_vector;
        edge_costs_table_last_vector.reserve(static_cast<size_t>(num_edges) * n_cp_);

        if (implicit_goal_facts_internal) {
            // Cost function indices correspond to operator ids, with the last one as the baseline.
            const int baseline_cp_index = original_n_cp - 1;
            // We only need edge_costs_table_last_ for update_hm_table().
            for (int local_cp = 0; local_cp < n_cp_; ++local_cp) {
                const int global_cp = start + local_cp;
                for (int edge_index = 0; edge_index < num_edges; ++edge_index) {
                    const auto &edge = hyperedges_[edge_index];
                    int cost = edge.cost;
                    if (global_cp != baseline_cp_index && edge.operator_id == global_cp) {
                        cost = 0;
                    }
                    edge_costs_table_last_vector.push_back(cost);
                }
            }
        } else {
            // Legacy path: slice prebuilt edge-cost layouts.
            const int cp_major_stride = static_cast<int>(edge_costs_cp_major_.size()) / original_n_cp;
            for (int cp_index = start; cp_index < end; ++cp_index) {
                int base = cp_index * cp_major_stride;
                edge_costs_table_last_vector.insert(
                    edge_costs_table_last_vector.end(),
                    edge_costs_cp_major_.begin() + base,
                    edge_costs_cp_major_.begin() + base + cp_major_stride);
            }
        }

        // update_hm_table() uses edge_costs_table_last_ only.
        edge_costs_ = torch::tensor({}, int_options_).view({0, n_cp_, 1});
        edge_costs_table_last_ = edge_costs_table_last_vector.empty()
            ? torch::tensor(edge_costs_table_last_vector, int_options_).view({n_cp_, 1, 0})
            : torch::tensor(edge_costs_table_last_vector, int_options_).view({n_cp_, 1, num_edges});

        // Allocate working buffers per chunk to keep peak memory bounded.
        hm_table_ = torch::zeros({n_cp_, 1, table_size_}, int_options_);
        all_values_ = torch::zeros({n_cp_, 1, select_indices_.size(0)}, int_options_);
        max_values_ = torch::zeros({n_cp_, 1, num_edges}, int_options_);
        last_heuristic_table_ = torch::zeros_like(hm_table_);
        less_than_table_ = torch::zeros_like(hm_table_, bool_options_);

        compute_heuristic(vector<State>{state});
        torch::Tensor tensor = hm_table_.index_select(2, goal_indices_).permute({2, 0, 1}).contiguous();
        tensors.push_back(tensor);
    }

    n_cp_ = original_n_cp;
    edge_costs_ = edge_costs_saved;
    edge_costs_table_last_ = edge_costs_table_last_saved;
    hm_table_ = hm_table_saved;
    all_values_ = all_values_saved;
    max_values_ = max_values_saved;
    last_heuristic_table_ = last_heuristic_table_saved;
    less_than_table_ = less_than_table_saved;

    if (tensors.empty()) {
        return {};
    }

    torch::Tensor tensor = torch::cat(tensors, 1);

    vector<vector<int>> result;
    //compute_heuristic(state);
    //torch::Tensor tensor = hm_table_.index_select(2, goal_indices_);

    auto sizes = tensor.sizes();
    int num_goals = sizes[0];
    int num_cp = sizes[1];
    int batch_size = sizes[2];

    assert(batch_size == 1);
    (void)batch_size;

    torch::Tensor tensor_cpu = tensor.to(torch::kCPU);

    for (int cp_idx = 0; cp_idx < num_cp; ++cp_idx) {
        vector<int> row;
        row.reserve(num_goals);
        for (int goal_idx = 0; goal_idx < num_goals; ++goal_idx) {
            row.push_back(tensor_cpu[goal_idx][cp_idx][0].item<int>());
        }
        result.push_back(row);
    }
    assert((int)result.size() == n_cp_);
    if (!result.empty()) {
        assert(result.at(0).size() == get_goal_indices().size());
    }

    return result;
}



class HM2TorchHeuristicFeature
    : public plugins::TypedFeature<Evaluator, HM2Torch> {
public:
    HM2TorchHeuristicFeature() : TypedFeature("hm2torch") {
        document_title("h^m heuristic parallel implementation using PyTorch");

        add_option<int>("m", "subset size", "2", plugins::Bounds("1", "infinity"));
        // TODO: transform to enums or modules. 
        add_option<int>("cp", "cost partitioning mode: [none=0, random=1, goal_facts=2]", "0", plugins::Bounds("0", "3"));
        add_option<int>("n_cp", "number of cost functions", "1", plugins::Bounds("1", "infinity"));
        add_option<bool>("use_gpu", "cpu opr gpu", "false");
        add_option<int>(
            "max_batch_size",
            "Cap batched hm2torch evaluation. If > 0, split larger batches into chunks to limit persistent GPU/pinned allocations. Use -1 for uncapped.",
            "-1",
            plugins::Bounds("-1", "infinity"));
        add_heuristic_options_to_feature(*this, "hm2torch");

        document_language_support("action costs", "supported");
        document_language_support("conditional effects", "ignored");
        document_language_support("axioms", "ignored");

        document_property(
            "admissible",
            "yes for tasks without conditional effects or axioms");
        document_property(
            "consistent",
            "yes for tasks without conditional effects or axioms");
        document_property(
            "safe",
            "yes for tasks without conditional effects or axioms");
        document_property("preferred operators", "no");
    }

    virtual shared_ptr<HM2Torch> create_component(
        const plugins::Options &opts,
        const utils::Context &) const override {
        return plugins::make_shared_from_arg_tuples<HM2Torch>(
            opts.get<int>("m"),
            opts.get<int>("cp"),
            opts.get<int>("n_cp"),
            opts.get<bool>("use_gpu"),
            opts.get<int>("max_batch_size"),
            get_heuristic_arguments_from_options(opts)
            );
    }
};

static plugins::FeaturePlugin<HM2TorchHeuristicFeature> _plugin;
}
