#include "hm2_base.h"

#include "../plugins/plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cassert>
#include <limits>
#include <numeric>
#include <random>
#include <set>

using namespace std;

namespace hm2_base {
HM2Base::HM2Base(
    int m, int cp, int n_cp, bool use_gpu, const shared_ptr<AbstractTask> &transform,
    bool cache_estimates, const string &description,
    utils::Verbosity verbosity)
    : Heuristic(transform, cache_estimates, description, verbosity),
      m(m),
      cp_(static_cast<CostPartition>(cp)),
      n_cp_(n_cp),
      goals(task_properties::get_fact_pairs(task_proxy.get_goals())),
      num_operators_(task_proxy.get_operators().size()),
      has_op_with_empty_preconds(task_properties::has_operator_with_empty_precondition(task_proxy)),
      dummy_precond_node_id(-1),
      has_cond_effects(task_properties::has_conditional_effects(task_proxy)),
      use_gpu_(use_gpu) {
    assert((m >= 1 and m <= 2) && "m must be 1 or 2");
    generate_all_tuples();
}

bool HM2Base::dead_ends_are_reliable() const {
    return !task_properties::has_axioms(task_proxy) && !has_cond_effects;
}

int HM2Base::getGlobalIndex(FactPair fp1, FactPair fp2) const {
    assert(fp1.var != fp2.var);
    if (fp1.var < fp2.var) {
        return h2_indices.at(make_tuple(fp1, fp2));
    }
    assert(fp1.var > fp2.var);
    return h2_indices.at(make_tuple(fp2, fp1));
}

int HM2Base::getTotalIndices(int m, int total_number) {
    if (m == 1) {
        return total_number;
    }
    if (m == 2) {
        return total_number + total_number * (total_number - 1) / 2;
    }
    throw invalid_argument("not implemented for m = " + to_string(m));
}

bool HM2Base::contains_fact(FactPair fact, Tuple facts) {
    for (FactPair f : facts) {
        if (f.var == fact.var) {
            return true;
        }
    }
    return false;
}

set<uint32_t> HM2Base::get_tail(Tuple preconditions) {
    //for (size_t i = 0; i < preconditions.size(); i++) {
    //    FactPair fact = preconditions[i];
    //    cout << "Fact: Var: " << fact.var << ", Value: " << fact.value << endl;
    //}
    set<uint32_t> tail;
    for (size_t idx1 = 0; idx1 < preconditions.size(); idx1++) {
        FactPair pre1 = preconditions[idx1];
        int pre1_idx = variable_offsets[pre1.var] + pre1.value;
        tail.emplace(pre1_idx);
        if (m >= 2) {
            for (size_t idx2 = idx1 + 1; idx2 < preconditions.size(); idx2++) {
                FactPair pre2 = preconditions[idx2];
                int pair_idx = getGlobalIndex(pre1, pre2);
                tail.emplace(pair_idx);
            }
        }
    }

    // Empty
    if (tail.empty()) {
        assert(has_op_with_empty_preconds);
        tail.emplace(dummy_precond_node_id);
    }

    return tail;
}

HM2Base::Tuple HM2Base::iterate_atoms(int idx) const {
    Tuple res;
    int global_idx = 0;
    for (size_t i = 0; i < task_proxy.get_variables().size(); ++i) {
        size_t domain_size = task_proxy.get_variables()[i].get_domain_size();
        for (size_t j = 0; j < domain_size; ++j) {
            global_idx++;
            if (global_idx <= idx) {
                continue;
            }
            FactPair fact(i, j);
            res.push_back(fact);
        }
    }
    return res;
}



void HM2Base::init_hm_table(const Tuple &t) {
    hm_table = vector<int>(table_size_, numeric_limits<int>::max());
    if (has_op_with_empty_preconds)
        hm_table.back() = 0;

    for (size_t i = 0; i < t.size(); ++i) {
        FactPair atom = t[i];
        int atom_idx1 = variable_offsets[atom.var] + atom.value;
        hm_table[atom_idx1] = 0;
        if (m >= 2) {
            for (size_t j = i + 1; j < t.size(); ++j) {
                FactPair atom2 = t[j];
                // int atom_idx2 = variable_offsets[atom2.var] + atom2.value;
                size_t table_index = getGlobalIndex(atom, atom2);
                assert(table_index < hm_table.size());
                hm_table[table_index] = 0;
            }
        }
    }
}

void HM2Base::init_hm_table(const vector<Tuple> &t_list) {
    int bs = t_list.size();
    if (hm_table.size() != table_size_ * bs) {
        hm_table.resize(table_size_ * bs, numeric_limits<int>::max());
    } else {
        std::fill(hm_table.begin(), hm_table.end(), numeric_limits<int>::max());
    }

    for (size_t tuple_idx = 0; tuple_idx < t_list.size(); ++tuple_idx) {
        if (has_op_with_empty_preconds) {
            // Table is laid out table-major with batch as the inner dimension:
            // entry(table_index, batch_index) lives at [batch_index + table_index * bs].
            hm_table[tuple_idx + dummy_precond_node_id * bs] = 0;
        }

        Tuple t = t_list[tuple_idx];
        for (size_t i = 0; i < t.size(); ++i) {
            FactPair atom = t[i];
            int atom_idx1 = variable_offsets[atom.var] + atom.value;
            hm_table[tuple_idx + atom_idx1 * bs] = 0;
            if (m >= 2) {
                for (size_t j = i + 1; j < t.size(); ++j) {
                    FactPair atom2 = t[j];
                    // int atom_idx2 = variable_offsets[atom2.var] + atom2.value;
                    size_t table_index = getGlobalIndex(atom, atom2);
                    assert(table_index < hm_table.size());
                    hm_table[tuple_idx + table_index * bs] = 0;
                }
            }
        }
    }
}

vector<Hyperedge> HM2Base::get_hyperedges(int m) {
    //if (m < 2 || m > 3) {
    //    cerr << "We only support the construction of hyperedges with m=2 or m=3." << endl;
    //    utils::exit_with(utils::ExitCode::SEARCH_UNSUPPORTED);
    //}

    vector<Hyperedge> hyperedges;
    int operator_id = 0;
    for (OperatorProxy op : task_proxy.get_operators()) {
        Tuple preconditions = get_operator_pre(op);
        Tuple effects = get_operator_eff(op);
        set<uint32_t> tail = get_tail(preconditions);

        int cost = op.get_cost();


        for (size_t idx1 = 0; idx1 < effects.size(); idx1++) {
            FactPair eff1 = effects[idx1];
            int eff1_idx = variable_offsets[eff1.var] + eff1.value;
            hyperedges.emplace_back(tail, eff1_idx, cost, operator_id);
            for (size_t idx2 = idx1 + 1; idx2 < effects.size(); idx2++) {
                FactPair eff2 = effects[idx2];
                int pair_idx = getGlobalIndex(eff1, eff2);
                hyperedges.emplace_back(tail, pair_idx, cost, operator_id);
            }
            Tuple all_atoms = iterate_atoms(0);
            for (size_t idx3 = 0; idx3 < all_atoms.size(); ++idx3) {
                FactPair persist_fact = all_atoms[idx3];

                if (eff1.var == persist_fact.var) {
                    continue;
                }

                bool is_valid = true;
                for (FactPair eff_fact : effects) {
                    if ((persist_fact.var == eff_fact.var) && (persist_fact.value != eff_fact.value)) {
                        is_valid = false;
                        break;
                    }
                }
                if (!is_valid) {
                    continue;
                }

                bool contains_second_fact = false;

                for (FactPair f : preconditions) {
                    if (f.var == persist_fact.var) {
                        if (f.value != persist_fact.value) {
                            is_valid = false;
                            break;
                        }
                        contains_second_fact = true;
                    }
                }

                if (!is_valid) {
                    continue;
                }

                if (!contains_second_fact) {
                    preconditions.push_back(persist_fact);
                }

                set<uint32_t> tail_2 = get_tail(preconditions);
                int pair_idx = getGlobalIndex(eff1, persist_fact);
                hyperedges.emplace_back(tail_2, pair_idx, cost, operator_id);

                if (!contains_second_fact) {
                    preconditions.pop_back();
                }
            }
        }
        operator_id++;
    }

    // IMPORTANT: assign cost partitions before pruning.
    // Pruning uses Hyperedge::cost_partition for equality/dominance checks.
    // If we prune before partitioning, operator-distinct edges can collapse and
    // partition-specific costs get mis-assigned, breaking admissibility.
    assign_cost_partition(hyperedges);

    remove_dominated_hyperedge(hyperedges);
    simply_tails(hyperedges); // Important to do after remove_dominated_hyperedge

    build_edge_cost_layouts(hyperedges);
    return hyperedges;
}

void HM2Base::build_edge_cost_layouts(const std::vector<Hyperedge> &hyperedges) {
    edge_costs_edge_major_.clear();
    edge_costs_cp_major_.clear();

    if (cp_ == CostPartition::GOAL_FACTS_INTERNALLY) {
        // Internal mode used for goal-facts cost partitioning.
        // Do not materialize O(num_edges * num_actions) edge-cost layouts here;
        // the Torch implementation generates the needed cost tensors in chunks.
        return;
    }

    if (hyperedges.empty() || n_cp_ <= 0) {
        return;
    }

    const int num_edges = static_cast<int>(hyperedges.size());
    edge_costs_edge_major_.reserve(num_edges * n_cp_);
    edge_costs_cp_major_.assign(num_edges * n_cp_, 0);

    for (int edge_index = 0; edge_index < num_edges; ++edge_index) {
        const auto &edge = hyperedges[edge_index];
        assert(static_cast<int>(edge.cost_partition.size()) == n_cp_);
        edge_costs_edge_major_.insert(
            edge_costs_edge_major_.end(),
            edge.cost_partition.begin(),
            edge.cost_partition.end());
        for (int cp_index = 0; cp_index < n_cp_; ++cp_index) {
            edge_costs_cp_major_[cp_index * num_edges + edge_index] = edge.cost_partition[cp_index];
        }
    }
}

void HM2Base::generate_all_tuples() {
    int num_variables = task_proxy.get_variables().size();
    variable_offsets = vector<int>(num_variables, 0);
    int total_size = 0;
    for (int i = 0; i < num_variables; ++i) {
        int domain_size = task_proxy.get_variables()[i].get_domain_size();
        variable_offsets[i] = total_size;
        total_size += domain_size;
    }

    int table_index = total_size;
    for (int v1 = 0; v1 < num_variables; ++v1) {
        for (int val1 = 0; val1 < task_proxy.get_variables()[v1].get_domain_size(); ++val1) {
            for (int v2 = v1 + 1; v2 < num_variables; ++v2) {
                for (int val2 = 0; val2 < task_proxy.get_variables()[v2].get_domain_size(); ++val2) {
                    //cout << "((" << v1 << ", " << val1 << "), (" << v2 << ", " << val2 << ")) -> " << table_index << endl;
                    auto pair = make_tuple(FactPair(v1, val1), FactPair(v2, val2));
                    h2_indices[pair] = table_index;
                    table_index++;
                }
            }
        }
    }

    table_size_ = table_index;




    hm_table = vector<int>(table_size_, numeric_limits<int>::max());

    // handle operators with empty preconditon by introducing a dummynode which has cost zero
    // this is because the max aggregation over an empty tail may has an undefined behavior in torch
    if (has_op_with_empty_preconds) {
        log << "Introducing dummy node to the hypergraph for handling empty preconditions of operators." << endl;
        ++table_size_;
        hm_table.push_back(0);
        dummy_precond_node_id = table_size_ - 1;
    }
}

HM2Base::Tuple HM2Base::get_operator_pre(const OperatorProxy &op) const {
    Tuple preconditions = task_properties::get_fact_pairs(op.get_preconditions());
    sort(preconditions.begin(), preconditions.end());
    return preconditions;
}

HM2Base::Tuple HM2Base::get_operator_eff(const OperatorProxy &op) const {
    Tuple effects;
    for (EffectProxy eff : op.get_effects()) {
        effects.push_back(eff.get_fact().get_pair());
    }
    sort(effects.begin(), effects.end());
    return effects;
}

int HM2Base::eval(const Tuple &t) const {
    int max = 0;
    if (m >= 1) {
        for (size_t i = 0; i < t.size(); ++i) {
            FactPair fact1 = t[i];
            int index = variable_offsets[fact1.var] + fact1.value;
            int h = hm_table.at(index);
            if (h > max) {
                max = h;
            }
            if (m >= 2) {
                for (size_t j = i + 1; j < t.size(); ++j) {
                    FactPair fact2 = t[j];
                    int table_index = getGlobalIndex(fact1, fact2);
                    int h = hm_table.at(table_index);
                    if (h > max) {
                        max = h;
                    }
                }
            }
        }
    }
    //cout << "Max: " << max << endl;
    return max;
}

int HM2Base::compute_heuristic(const State &ancestor_state) {
    (void)ancestor_state;
    throw runtime_error("not implemented");
}

vector<int> HM2Base::compute_heuristic(const vector<State> &ancestor_states) {
    vector<int> results = {};
    for (const State &state : ancestor_states) {
        results.push_back(compute_heuristic(state));
    }
    return results;
}

void HM2Base::remove_dominated_hyperedge(vector<Hyperedge> &hyperedges) {
    // In GOAL_FACTS_INTERNALLY mode, operator-specific costs are generated on-the-fly
    // (e.g., "make operator i free"). The Hyperedge::cost_partition vectors are
    // intentionally not materialized in that mode, so pruning based on
    // cost_partition would be unsound (it can merge/prune operator-distinct edges).
    if (!hyperedges.empty() && hyperedges.front().cost_partition.empty()) {
        return;
    }

    log << "Original Hyperedges Count: " << hyperedges.size() << endl;

    // Remove duplicates
    sort(hyperedges.begin(), hyperedges.end());
    hyperedges.erase(unique(hyperedges.begin(), hyperedges.end()), hyperedges.end());
    log << "Deduplicated Hyperedges Count: " << hyperedges.size() << endl;

    // Remove dominated hyperedge
    unordered_map<uint32_t, vector<Hyperedge>> grouped_hyperedges;

    // Grouping the hyperedges by head
    for (const auto &edge : hyperedges) {
        grouped_hyperedges[edge.head].push_back(edge);
    }

    // Create a new vector to store the non-dominated hyperedegs
    vector<Hyperedge> non_dominated_hyperedges;

    // Note: Ensure that duplicate hyperedges have been removed before
    // We can eliminate an edge e1=<T1, h1, c1> if there exists an edge e2=<T2, h2, c2> such that
    // T1 is a superset of T2, h1 equals h2, and c1 is greater than or equal to c2.
    for (const auto & [head, edges] : grouped_hyperedges) {
        for (const auto &edge1 : edges) {
            bool edge1_dominated = false;
            for (const auto &edge2 : edges) {
                // We have already filtered out all duplicates
                if (edge1 == edge2) {
                    continue;
                }

                if (edge2.dominates(edge1)) {
                    edge1_dominated = true;
                    break;
                }
            }

            if (!edge1_dominated) {
                non_dominated_hyperedges.push_back(edge1);
            }
        }
    }

    hyperedges = move(non_dominated_hyperedges);

    log << "Undominated Hyperedges Count: " << hyperedges.size() << endl;
}

// We remove nodes in the tail that are supersumed. For example
// The node representing a is supersumed by the node a,b.
// This only happens with m > 1 and if the preconditon contains
// multiple atoms.
void HM2Base::simply_tails(std::vector<Hyperedge> &hyperedges) {
    if (m == 2) {
        for (auto &edge : hyperedges) {
            if (edge.tail.size() > 1) {
                int op_id = edge.operator_id;
                Tuple preconditions = get_operator_pre(task_proxy.get_operators()[op_id]);
                for (size_t pre_id = 0; pre_id < preconditions.size(); ++pre_id) {
                    FactPair pre = preconditions[pre_id];
                    int pre_node_id = variable_offsets[pre.var] + pre.value;
                    edge.tail.erase(pre_node_id);
                }
            }
        }
    }
}

void HM2Base::assign_cost_partition(vector<Hyperedge> &hyperedges) {
    // Cost Partitions
    //cout << "Cost Partition Mode: " << static_cast<int>(cp_) << endl;
    if (cp_ != CostPartition::NONE) {
        if (cp_ == CostPartition::GOAL_FACTS_INTERNALLY) {
            // Internal mode used for goal-facts cost partitioning.
            // Do not create per-edge cost_partition vectors of size num_operators_ + 1.
            for (Hyperedge &edge : hyperedges) {
                edge.cost_partition.clear();
            }
        } else if (cp_ == CostPartition::GOAL_FACTS) {
            log << endl << "Computing goal facts cost partition..." << endl;
            assign_goal_facts_cost_partition(hyperedges);
            log << "...done!" << endl << endl;
        } else if (cp_ == CostPartition::RANDOM) {
            log << endl << "Computing randomized cost partition..." << endl;
            assign_random_cost_partition(hyperedges);
            log << "...done!" << endl << endl;
        }
    }
    if (!hyperedges.empty() && !hyperedges.at(0).cost_partition.empty()) {
        n_cp_ = hyperedges.at(0).cost_partition.size();
        log << "Cost Function Count: " << n_cp_ << endl;
    }
}

void HM2Base::assign_random_cost_partition(vector<Hyperedge> &hyperedges) {
    n_cp_ = min(n_cp_, (int)num_operators_);
    if (n_cp_ == 0) {
        cp_ = CostPartition::NONE;
        n_cp_ = 1;
        return;
    }
    vector<vector<size_t>> action_to_hyperedges(num_operators_);
    for (size_t index = 0; index < hyperedges.size(); ++index) {
        action_to_hyperedges[hyperedges[index].operator_id].push_back(index);
    }

    mt19937 rng;
    rng.seed(187);
    map<size_t, size_t> operator_cost_id;
    for (size_t op_id = 0; op_id < num_operators_; ++op_id) {
        uniform_int_distribution<size_t> dist(0, n_cp_ - 1);
        operator_cost_id[op_id] = dist(rng);
    }
    log << "len hyperedges: " << hyperedges.size() << endl;
    for (auto &hyperedges_ids : action_to_hyperedges) {
        for (auto &edge_id : hyperedges_ids) {
            Hyperedge &edge = hyperedges[edge_id];
            vector<int64_t> edge_cp(n_cp_, 0);
            int operator_id = edge.operator_id;
            int cp_id = operator_cost_id[operator_id];
            edge_cp[cp_id] = edge.cost;
            edge.cost_partition.clear();     // if we want to use only the cost partition, uncoment this line
            edge.cost_partition.insert(edge.cost_partition.end(), edge_cp.begin(), edge_cp.end());
            edge.cost_partition.push_back(edge.cost);
            assert((int)edge.cost_partition.size() == n_cp_ + 1);
            int64_t partition_sum = 0;
            for (int i = 0; i < n_cp_; ++i) {
                partition_sum += edge.cost_partition[i];
            }
            assert(edge.cost_partition.back() == edge.cost);
            assert(partition_sum == edge.cost);
        }
    }
}

void HM2Base::assign_goal_facts_cost_partition(vector<Hyperedge> &hyperedges) {
    int num_actions = num_operators_;
    mt19937 rng;
    rng.seed(187);

    // Sample atomic goal facts (without replacement) based on n_cp_.
    // We only consider the atomic goals which are the first ones.
    const size_t num_atomic_goals = goals.size();
    const size_t requested = static_cast<size_t>(max(1, n_cp_));
    const size_t num_sampled_goals = min(requested, num_atomic_goals);
    vector<size_t> sampled_goal_indices;
    sampled_goal_indices.reserve(num_sampled_goals);
    {
        vector<size_t> all_goal_indices(num_atomic_goals);
        iota(all_goal_indices.begin(), all_goal_indices.end(), 0);
        shuffle(all_goal_indices.begin(), all_goal_indices.end(), rng);
        all_goal_indices.resize(num_sampled_goals);
        sampled_goal_indices = std::move(all_goal_indices);
    }
    log << "Using " << num_sampled_goals << " of " << num_atomic_goals
        << " goal atoms for cost partitioning (n_cp=" << n_cp_ << ")." << endl;

    State initial_state = task_proxy.get_initial_state();
    CostPartition cost_part = CostPartition::GOAL_FACTS_INTERNALLY;
    unique_ptr<HM2Base> heuristic = get_base_heuristic(cost_part, num_actions + 1);

    vector<vector<int>> goal_h_values_per_cost_function = heuristic->get_goal_facts_h_values_per_cost(initial_state);
    assert((int)goal_h_values_per_cost_function.size() == num_actions + 1);
    assert(goal_h_values_per_cost_function[0].size() == get_goal_indices().size());

    // One cost function per sampled goal (plus an implicit total-cost component later).
    vector<vector<size_t>> cost_function_to_action(num_sampled_goals);
    for (size_t action_index = 0; action_index < task_proxy.get_operators().size(); ++action_index) {
        assert(goal_h_values_per_cost_function[action_index].size() == get_goal_indices().size());
        assert(goals.size() <= goal_h_values_per_cost_function[action_index].size());
        int max_contribution = 0;
        vector<size_t> best_goal_positions;
        best_goal_positions.reserve(num_sampled_goals);
        for (size_t goal_pos = 0; goal_pos < sampled_goal_indices.size(); ++goal_pos) {
            const size_t goal_index = sampled_goal_indices[goal_pos];
            int cur_contribution = goal_h_values_per_cost_function.back()[goal_index] - goal_h_values_per_cost_function[action_index][goal_index];
            assert(cur_contribution >= 0);
            if (cur_contribution > max_contribution) {
                max_contribution = cur_contribution;
                best_goal_positions = vector<size_t>({goal_pos});
            } else if (cur_contribution == max_contribution) {
                best_goal_positions.push_back(goal_pos);
            }
        }
        uniform_int_distribution<int> dist(0, best_goal_positions.size() - 1);
        const size_t chosen_goal_pos = best_goal_positions[dist(rng)];
        cost_function_to_action[chosen_goal_pos].push_back(action_index);
    }

    vector<size_t> action_to_cost_function(num_actions, static_cast<size_t>(-1));
    for (size_t cost_function_id = 0; cost_function_id < cost_function_to_action.size(); ++cost_function_id) {
        for (size_t action_id : cost_function_to_action[cost_function_id]) {
            action_to_cost_function[action_id] = cost_function_id;
        }
    }
    for (auto &edge : hyperedges) {
        assert(action_to_cost_function[edge.operator_id] != static_cast<size_t>(-1));
        size_t cost_function_id = action_to_cost_function[edge.operator_id];
        vector<int> edge_cp(cost_function_to_action.size(), 0);
        edge_cp[cost_function_id] = edge.cost;
        edge.cost_partition.clear();
        edge.cost_partition.insert(edge.cost_partition.end(), edge_cp.begin(), edge_cp.end());
        edge.cost_partition.push_back(edge.cost);
        assert(edge.cost_partition.size() == cost_function_to_action.size() + 1);
        int64_t partition_sum = 0;
        for (size_t i = 0; i + 1 < edge.cost_partition.size(); ++i) {
            partition_sum += edge.cost_partition[i];
        }
        assert(edge.cost_partition.back() == edge.cost);
        assert(partition_sum == edge.cost);
    }
}
}
