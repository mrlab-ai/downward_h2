#include "hyperedge.h"
#include "../heuristic.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace plugins {
class Options;
}
namespace hm2_base  {

    enum class CostPartition
    {
        NONE,
        RANDOM,
        GOAL_FACTS,
        OPTIMIZE,
        GOAL_FACTS_INTERNALLY,
        OPTIMIZE_INTERNALLY
    };

    class HM2Base : public Heuristic {

        using Tuple = std::vector<FactPair>;
    // parameters
    


    

    // auxiliary methods
    void update_hm_table();
    int update_hm_entry(int t, int val);
    void extend_tuple(const Tuple &t, const OperatorProxy &op);

    bool contains_fact(FactPair fact, Tuple facts);


    int get_operator_pre_value(const OperatorProxy &op, int var) const;
  

    void generate_all_tuples();

    std::set<uint32_t> get_tail(std::vector<FactPair> preconditions);


    void dump_table() const;
    void dump_atom(const FactPair &atom);
    void dump_tuple(std::vector<FactPair> &t);

    

protected:
    const int m;
    CostPartition cp_;
    int n_cp_;
    const Tuple goals;
    size_t num_operators_;
    const bool has_op_with_empty_preconds;
    int dummy_precond_node_id;
    const bool has_cond_effects;
    const bool use_gpu_; //required here because of hyper edge CP init in torch version
    torch::Tensor optimized_edge_costs_;
    std::vector<int> edge_costs_edge_major_;
    std::vector<int> edge_costs_cp_major_;


    void init_hm_table(const Tuple &t);
    void init_hm_table(const std::vector<Tuple> &t_list);

    int getGlobalIndex(FactPair atom_1, FactPair atom_2) const;
    Tuple get_operator_pre(const OperatorProxy &op) const;
    Tuple get_operator_eff(const OperatorProxy &op) const;
    int getTotalIndices(int m, int total_number);
    Tuple iterate_atoms(int idx) const;
    int eval(const Tuple &t) const;
    std::vector<Hyperedge> get_hyperedges(int m);
    virtual std::vector<uint32_t> get_goal_indices() = 0;

    virtual torch::Tensor optimize_cp(const State &ancestor_state) = 0;

    void remove_dominated_hyperedge(std::vector<Hyperedge>& hyperedges);
    void simply_tails(std::vector<Hyperedge>& hyperedges);
    void assign_cost_partition(std::vector<Hyperedge>& hyperedges);
    void build_edge_cost_layouts(const std::vector<Hyperedge> &hyperedges);
    void assign_optimize_cost_partition(std::vector<Hyperedge>& hyperedges);
    void assign_random_cost_partition(std::vector<Hyperedge>& hyperedges);
    void assign_goal_facts_cost_partition(std::vector<Hyperedge>& hyperedges);
    virtual std::unique_ptr<HM2Base> get_base_heuristic(CostPartition &cp, int n_cp) = 0;

    virtual std::vector<std::vector<int>> get_goal_facts_h_values_per_cost(State& state) = 0;


    // h^m table
    std::vector<int> hm_table;
    utils::HashMap<std::tuple<FactPair, FactPair>, int> h2_indices;


    int table_size_;

    std::vector<int> variable_offsets; 
    bool was_updated;



    virtual int compute_heuristic(const State &ancestor_state) override;
    virtual std::vector<int> compute_heuristic(const std::vector<State> &ancestor_states) override;

public:

    

    HM2Base(
        int m, int cp, int n_cp, bool use_gpu, const std::shared_ptr<AbstractTask> &transform,
        bool cache_estimates, const std::string &description,
        utils::Verbosity verbosity);
    virtual bool dead_ends_are_reliable() const override;

};
}


