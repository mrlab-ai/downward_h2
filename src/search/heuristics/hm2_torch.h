#include "../heuristic.h"
#include "hm2_base.h"

#include "hm2_torch_cuda.h"

#include <array>
#include <torch/torch.h>

namespace plugins {
    class Options;
}

namespace hm2_torch {
    class HM2Torch : public hm2_base::HM2Base {
        using Tuple = std::vector<FactPair>;

        // attributes
        mutable torch::Tensor hm_table_;
        torch::Tensor select_indices_;
        torch::Tensor max_indices_;
        torch::Tensor edge_costs_;
        torch::Tensor edge_costs_table_last_;
        torch::Tensor edge_offsets_;
        torch::Tensor edge_select_indices_;
        torch::Tensor min_indices_;
        torch::Tensor grouped_head_indices_;
        torch::Tensor grouped_head_offsets_;
        torch::Tensor grouped_edge_indices_;
        torch::Tensor goal_indices_;
        int max_batch_size_;
        int max_batch_size_cap_;
        int iteration_count_;
        std::vector<Hyperedge> hyperedges_;
        std::vector<int> edge_costs_vector_;

        torch::Tensor all_values_;
        torch::Tensor max_values_;
        torch::Tensor last_heuristic_table_;
        torch::Tensor less_than_table_;
        torch::Tensor any_scalar_;
        torch::Tensor device_changed_flag_;
        torch::Tensor hm_table_staging_pinned_;
        std::array<torch::Tensor, 2> init_batch_idx_pinned_;
        std::array<torch::Tensor, 2> init_table_idx_pinned_;
        std::array<torch::Tensor, 2> init_batch_idx_device_;
        std::array<torch::Tensor, 2> init_table_idx_device_;
        int init_buffer_slot_;

        const torch::DeviceType device_;
        const c10::TensorOptions int_options_;
        const c10::TensorOptions bool_options_;

        void init_hm_table_cuda(const std::vector<Tuple> &t_list, torch::Tensor &hm_slice);



        // auxiliary methods
        std::vector<int> eval_tensor(torch::Tensor& hm_slice) const;
        void print_torch_vector(torch::Tensor& t);
        void update_hm_table(
            torch::Tensor& hm_table,
            torch::Tensor& all_values,
            torch::Tensor& max_values,
            torch::Tensor& last_heuristic_table,
            torch::Tensor& less_than_table,
            torch::Tensor& any_scalar
            );
        int update_hm_entry(int t, int val);
        void extend_tuple(const Tuple &t, const OperatorProxy &op);

        int check_tuple_in_tuple(const Tuple &tuple, const Tuple &big_tuple) const;
        std::vector<uint32_t> get_goal_indices() override;

        int get_operator_pre_value(const OperatorProxy &op, int var) const;
        bool contradict_effect_of(const OperatorProxy &op, int var, int val) const;

        void generate_all_tuples();
        void generate_all_tuples_aux(int var, int sz, const Tuple &base);

        void generate_all_partial_tuples(const Tuple &base_tuple,
                                         std::vector<Tuple> &res) const;
        void generate_all_partial_tuples_aux(const Tuple &base_tuple, const Tuple &t, int index,
                                             int sz, std::vector<Tuple> &res) const;


        void dump_table() const;
        void dump_atom(const FactPair &atom);
        void dump_tuple(std::vector<FactPair> &t);

        torch::DeviceType get_device(bool use_gpu) const;
        std::vector<std::vector<int>> get_goal_facts_h_values_per_cost(State& state) override;

        std::unique_ptr<HM2Base> get_base_heuristic(hm2_base::CostPartition &cp, int n_cp) override;

    protected:
        virtual int compute_heuristic(const State &ancestor_state) override;
        virtual std::vector<int> compute_heuristic(const std::vector<State> &ancestor_states) override;
        torch::Tensor optimize_cp(const State &ancestor_state) override;


    public:
    HM2Torch(
        int m, int cp, int n_cp, bool use_gpu, int max_batch_size_cap, const std::shared_ptr<AbstractTask> &transform,
        bool cache_estimates, const std::string &description,
        utils::Verbosity verbosity);

    };

}