#ifndef HEURISTICS_HM2_HEURISTIC_H
#define HEURISTICS_HM2_HEURISTIC_H

#include "../heuristic.h"
#include "hm2_base.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace plugins {
    class Options;
}

namespace hm2_heuristic {
    /*
    Haslum's h^m heuristic family ("critical path heuristics").

    This is a very slow implementation and should not be used for
    speed benchmarks.
    */

    class HM2Heuristic : public hm2_base::HM2Base {
        using Tuple = std::vector<FactPair>;
    
        // auxiliary methods
        void update_hm_table();
        int update_hm_entry(int t, int val);
        void extend_tuple(const Tuple &t, const OperatorProxy &op);

        int check_tuple_in_tuple(const Tuple &tuple, const Tuple &big_tuple) const;

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

        std::vector<uint32_t> get_goal_indices() override;

        std::vector<std::vector<int>> get_goal_facts_h_values_per_cost(State& state) override;
        std::unique_ptr<HM2Base> get_base_heuristic(hm2_base::CostPartition &cp, int n_cp) override;

    protected:
        virtual int compute_heuristic(const State &ancestor_state) override;

        [[noreturn]] torch::Tensor optimize_cp(const State &/*ancestor_state*/) override {
            utils::exit_with(utils::ExitCode::SEARCH_UNSUPPORTED);
        };

    public:
        HM2Heuristic(
            int m, int cp, int n_cp, const std::shared_ptr<AbstractTask> &transform,
            bool cache_estimates, const std::string &description,
            utils::Verbosity verbosity);

        
    };
}

#endif
