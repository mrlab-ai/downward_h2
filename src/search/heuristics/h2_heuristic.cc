#include "h2_heuristic.h"

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"

#include <set>

namespace h2_heuristic {
H2Heuristic::H2Heuristic(const std::shared_ptr<AbstractTask> &transform,
            bool cache_estimates, const std::string &description,
            utils::Verbosity verbosity)
    : Heuristic(transform, cache_estimates, description, verbosity)
    , index_offsets()
    , h1_table()
    , h2_table()
    , operators()
    , goal()
    , has_cond_effects(task_properties::has_conditional_effects(task_proxy)) {
    int num_values = 0;

    for (int var_id = 0; var_id < task->get_num_variables(); ++var_id) {
        index_offsets.emplace_back(num_values);
        num_values += task->get_variable_domain_size(var_id);
    }

    // Preallocate h1_table and h2_table to avoid dynamic allocations when computing heuristic values

    h1_table.resize(num_values);
    h2_table.resize(num_values);

    for (int i = 0; i < num_values; ++i) {
        h2_table[i].resize(num_values);
    }

    // Convert operators to the internal format

    for (const auto& op : task_proxy.get_operators()) {
        std::vector<int> precondition;

        for (const auto& cond_proxy : op.get_preconditions()) {
            const auto [var_id, val_id] = cond_proxy.get_pair();
            precondition.emplace_back(get_index(var_id, val_id));
        }

        std::vector<int> add_effect;

        for (const auto& eff_proxy : op.get_effects()) {
            const auto [var_id, val_id] = eff_proxy.get_fact().get_pair();
            add_effect.emplace_back(get_index(var_id, val_id));
        }

        std::set<int> delete_effect;

        for (const auto& eff_proxy : op.get_effects()) {
            const auto [var_id, val_id] = eff_proxy.get_fact().get_pair();

            for (int other_val = 0; other_val < task->get_variable_domain_size(var_id); ++other_val) {
                delete_effect.emplace(get_index(var_id, other_val));
            }
        }

        std::vector<int> delete_effect_complement;

        for (int var_id = 0; var_id < task->get_num_variables(); ++var_id) {
            for (int val_id = 0; val_id < task->get_variable_domain_size(var_id); ++val_id) {
                const auto index = get_index(var_id, val_id);

                if (delete_effect.count(index) == 0) {
                    delete_effect_complement.emplace_back(index);
                }
            }
        }

        operators.emplace_back(precondition, add_effect, delete_effect_complement, op.get_cost());
    }

    // Convert the goal to the internal format

    for (const auto& pair_proxy : task_proxy.get_goals()) {
        const auto [var_id, val_id] = pair_proxy.get_pair();
        goal.emplace_back(get_index(var_id, val_id));
    }
}

bool H2Heuristic::dead_ends_are_reliable() const {
    return !task_properties::has_axioms(task_proxy) && !has_cond_effects;
}

int H2Heuristic::get_index(int var_id, int val_id) {
    return index_offsets[var_id] + val_id;
}

int H2Heuristic::evaluate(const std::vector<int>& indices) {
    int v = 0;

    for (std::size_t i = 0; i < indices.size(); i++) {
        v = std::max(v, h1_table[indices[i]]);

        if (v == INTERNAL_DEAD_END) {
            return INTERNAL_DEAD_END;
        }

        for (std::size_t j = i + 1; j < indices.size(); j++) {
            v = std::max(v, h2_table[indices[i]][indices[j]]);

            if (v == INTERNAL_DEAD_END) {
                return INTERNAL_DEAD_END;
            }
        }
    }

    return v;
}

int H2Heuristic::evaluate(const std::vector<int>& indices, int index) {
    int v = 0;

    v = std::max(v, h1_table[index]);

    if (v == INTERNAL_DEAD_END) {
        return INTERNAL_DEAD_END;
    }

    for (std::size_t i = 0; i < indices.size(); i++) {
        if (index == indices[i]) {
            continue;
        }

        v = std::max(v, h2_table[index][indices[i]]);

        if (v == INTERNAL_DEAD_END) {
            return INTERNAL_DEAD_END;
        }
    }

    return v;
}

void H2Heuristic::update(const std::size_t index, const int value, bool& changed) {
    if (h1_table[index] > value) {
        h1_table[index] = value;
        changed = true;
    }
}

void H2Heuristic::update(const std::size_t first_index, const std::size_t second_index, const int value, bool& changed) {
    if (h2_table[first_index][second_index] > value) {
        h2_table[first_index][second_index] = value;
        h2_table[second_index][first_index] = value;
        changed = true;
    }
}

void H2Heuristic::fill_tables(const State& state) {
    std::fill(h1_table.begin(), h1_table.end(), INTERNAL_DEAD_END);

    for (auto& h2_row : h2_table) {
        std::fill(h2_row.begin(), h2_row.end(), INTERNAL_DEAD_END);
    }

    const auto& variable_values = state.get_unpacked_values();
    const auto num_variables = static_cast<int>(variable_values.size());

    for (int first_var_id = 0; first_var_id < num_variables; ++first_var_id) {
        const auto first_val_id = variable_values[first_var_id];
        const auto first_idx = get_index(first_var_id, first_val_id);
        h1_table[first_idx] = 0;

        for (int second_var_id = 0; second_var_id < num_variables; ++second_var_id) {
            const auto second_val_id = variable_values[second_var_id];
            const auto second_idx = get_index(second_var_id, second_val_id);
            h2_table[first_idx][second_idx] = 0;
        }
    }

    bool changed;

    do {
        changed = false;

        for (const auto& [precondition, add_effect, delete_effect_complement, cost] : operators) {
            const auto c1 = evaluate(precondition);

            if (c1 == INTERNAL_DEAD_END) {
                continue;
            }

            for (std::size_t i = 0; i < add_effect.size(); i++) {
                const auto p = add_effect[i];
                update(p, c1 + cost, changed);

                for (std::size_t j = i + 1; j < add_effect.size(); j++) {
                    const auto q = add_effect[j];

                    if (p != q) {
                        update(p, q, c1 + cost, changed);
                    }
                }

                for (const auto r : delete_effect_complement) {
                    const auto c2 = std::max(c1, evaluate(precondition, r));

                    if (c2 != INTERNAL_DEAD_END) {
                        update(p, r, c2 + cost, changed);
                    }
                }
            }
        }
    } while (changed);
}

std::vector<int> H2Heuristic::compute_heuristic(const std::vector<State> &ancestor_states) {
    std::vector<int> h_values;
    for (const State &ancestor_state : ancestor_states) {
        h_values.push_back(compute_heuristic2(ancestor_state));
    }
    return h_values;
}

int H2Heuristic::compute_heuristic(const State& ancestor_state) {
    return compute_heuristic2(ancestor_state);
}

int H2Heuristic::compute_heuristic2(const State& ancestor_state) {
    const auto state = task_proxy.convert_ancestor_state(ancestor_state);
    fill_tables(state);
    const auto value = evaluate(goal);
    return (value == INTERNAL_DEAD_END) ? DEAD_END : value;
}

class H2HeuristicFeature : public plugins::TypedFeature<Evaluator, H2Heuristic> {
public:
    H2HeuristicFeature()
        : TypedFeature("h2") {
        document_title("h^2 heuristic");

        add_heuristic_options_to_feature(*this, "h2");

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

    virtual std::shared_ptr<H2Heuristic> create_component(
        const plugins::Options &opts,
        const utils::Context &) const override {
        return plugins::make_shared_from_arg_tuples<H2Heuristic>(
            get_heuristic_arguments_from_options(opts)
            );
    }
};

static plugins::FeaturePlugin<H2HeuristicFeature> _plugin;
}
