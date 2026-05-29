#include "hm2_heuristic.h"

#include "../plugins/plugin.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cassert>
#include <limits>
#include <set>

using namespace std;

#include <stdexcept>
#include <string>


namespace hm2_heuristic {
HM2Heuristic::HM2Heuristic(
    int m, int cp, int n_cp, const shared_ptr<AbstractTask> &transform,
    bool cache_estimates, const string &description,
    utils::Verbosity verbosity)
    : HM2Base(m, cp, n_cp, false, transform, cache_estimates, description, verbosity) {}






int HM2Heuristic::compute_heuristic(const State &ancestor_state) {
    State state = convert_ancestor_state(ancestor_state);
    if (task_properties::is_goal_state(task_proxy, state)) {
        return 0;
    } else {
        Tuple s_tup = task_properties::get_fact_pairs(state);

        init_hm_table(s_tup);
        update_hm_table();

        int h = eval(goals);

        if (h == numeric_limits<int>::max())
            return DEAD_END;
        return h;
    }
}

void HM2Heuristic::dump_atom(const FactPair &atom) {
    cout << "h(" << atom.var << ", " << atom.value << "), ";
}

void HM2Heuristic::dump_tuple(vector<FactPair> &t) {
    for (FactPair &atom : t) {
        dump_atom(atom);
    }
}

vector<vector<int>> HM2Heuristic::get_goal_facts_h_values_per_cost(State &state) {
    vector<uint32_t> goal_indices = get_goal_indices();
    vector<vector<int>> result;
    for (int cost_function_id = 0; cost_function_id < n_cp_; ++cost_function_id) {
        vector<int> h_values_per_goal;
        //compute_heuristic(state, cost_function_id);
        compute_heuristic(state);
        for (const auto goal_index : goal_indices) {
            h_values_per_goal.push_back(hm_table[goal_index]);
        }
        result.push_back(h_values_per_goal);
    }
    assert((int)result.size() == n_cp_);
    return result;
}

void HM2Heuristic::update_hm_table() {
    int round = 0;
    do {
        ++round;
        //dump_table();
        was_updated = false;

        for (OperatorProxy op : task_proxy.get_operators()) {
            Tuple pre = get_operator_pre(op);
            //cout << "pre: ";
            //for (FactPair fact : pre) {
            //    dump_atom(fact);
            //}
            int c1 = eval(pre);
            if (c1 != numeric_limits<int>::max()) {
                Tuple eff = get_operator_eff(op);
                if (m >= 1) {
                    for (size_t i = 0; i < eff.size(); ++i) {
                        FactPair effect_p = eff[i];
                        int effect_p_idx = variable_offsets[effect_p.var] + effect_p.value;
                        update_hm_entry(effect_p_idx, c1 + op.get_cost());
                        if (m >= 2) {
                            for (size_t j = i + 1; j < eff.size(); ++j) {
                                FactPair effect_q = eff[j];
                                if (effect_p.var == effect_q.var) {
                                    continue;
                                }
                                int tuple_index = getGlobalIndex(effect_p, effect_q);
                                update_hm_entry(tuple_index, c1 + op.get_cost());
                            }
                            Tuple all_atoms = iterate_atoms(0);
                            for (size_t idx = 0; idx < all_atoms.size(); ++idx) {
                                FactPair fact = all_atoms[idx];
                                if (effect_p.var == fact.var) {
                                    continue;
                                }

                                bool is_valid = true;
                                for (FactPair eff_fact : eff) { // remove atoms that are in delete list
                                    if ((fact.var == eff_fact.var) && (fact.value != eff_fact.value)) {
                                        is_valid = false;
                                        break;
                                    }
                                }
                                if (!is_valid) {
                                    continue;
                                }


                                bool found = false;
                                for (FactPair precondition_fact : pre) {
                                    if (precondition_fact.var == fact.var) {
                                        if (precondition_fact.value != fact.value) {
                                            is_valid = false;
                                            break;
                                        }
                                        found = true;
                                        break;
                                    }
                                }

                                if (!is_valid) {
                                    continue;
                                }

                                if (!found) {
                                    pre.push_back(fact);
                                }

                                //cout << "new_tuple: ";
                                //for (FactPair fact : pre) {
                                //    dump_atom(fact);
                                //}
                                int c2 = eval(pre);
                                if (c2 != numeric_limits<int>::max()) {
                                    int new_tuple_index = getGlobalIndex(effect_p, fact);
                                    update_hm_entry(new_tuple_index, c2 + op.get_cost());
                                }
                                if (!found) {
                                    pre.pop_back();
                                }
                            }
                        }
                    }
                }
            }
        }
    } while (was_updated);
}


int HM2Heuristic::update_hm_entry(int index, int val) {
    if (hm_table[index] > val) {
        hm_table[index] = val;
        was_updated = true;
    }
    return val;
}

unique_ptr<hm2_base::HM2Base> HM2Heuristic::get_base_heuristic(hm2_base::CostPartition &cp, int num_cost_partitions) {
    return make_unique<HM2Heuristic>(m, static_cast<int>(cp), num_cost_partitions, task, does_cache_estimates(), "test?", utils::Verbosity::SILENT);
}

vector<uint32_t> HM2Heuristic::get_goal_indices() {
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


void HM2Heuristic::dump_table() const {
    Tuple atoms_tmp = iterate_atoms(0);
    for (size_t i = 0; i < atoms_tmp.size(); ++i) {
        FactPair fact1 = atoms_tmp[i];
        int fact_idx = variable_offsets[fact1.var] + fact1.value;
        cout << "h([" << fact1.var << "=" << fact1.value << "]) = " << hm_table[fact_idx] << endl;
    }
    const Tuple atoms = iterate_atoms(0);
    for (size_t i = 0; i < atoms.size(); ++i) {
        FactPair fact1 = atoms[i];
        // int fact_idx = variable_offsets[fact1.var] + fact1.value;
        //cout << "h([" << fact1.var << "=" << fact1.value << "]) = " << hm_table[fact_idx] << ", idx: " << fact_idx << endl;
        Tuple atoms2 = iterate_atoms(i + 1);
        for (size_t j = 0; j < atoms2.size(); ++j) {
            FactPair fact2 = atoms2[j];
            if (fact1.var == fact2.var) {
                continue;
            }
            int table_index = getGlobalIndex(fact1, fact2);
            cout << "h([" << fact1.var << "=" << fact1.value << ", " << fact2.var << "=" << fact2.value << "]) = " << hm_table[table_index] << endl;
        }
    }
}

class HM2HeuristicFeature
    : public plugins::TypedFeature<Evaluator, HM2Heuristic> {
public:
    HM2HeuristicFeature() : TypedFeature("hm2") {
        document_title("h^m heuristic");

        add_option<int>("m", "subset size", "2", plugins::Bounds("1", "infinity"));
        add_option<int>("cp", "cost partitioning mode: [none=0, random=1, goal_facts=2]", "0", plugins::Bounds("0", "2"));
        add_option<int>("n_cp", "cost partitioning mode: [none=0, random=1, goal_facts=2]", "1", plugins::Bounds("1", "infinity"));
        add_heuristic_options_to_feature(*this, "hm2");

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

    virtual shared_ptr<HM2Heuristic> create_component(
        const plugins::Options &opts,
        const utils::Context &) const override {
        return plugins::make_shared_from_arg_tuples<HM2Heuristic>(
            opts.get<int>("m"),
            opts.get<int>("cp"),
            opts.get<int>("n_cp"),
            get_heuristic_arguments_from_options(opts)
            );
    }
};

static plugins::FeaturePlugin<HM2HeuristicFeature> _plugin;
}
