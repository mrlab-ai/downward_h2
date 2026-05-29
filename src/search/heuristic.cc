#include "heuristic.h"

#include "evaluation_context.h"
#include "evaluation_context_batched.h"
#include "evaluation_result.h"
#include "evaluation_result_batched.h"

#include "plugins/plugin.h"
#include "task_utils/task_properties.h"
#include "tasks/cost_adapted_task.h"
#include "tasks/root_task.h"

#include <cassert>
#include <cstdlib>
#include <limits>

using namespace std;
Heuristic::Heuristic(
    const shared_ptr<AbstractTask> &transform,
    bool cache_estimates, const string &description,
    utils::Verbosity verbosity)
    : Evaluator(true, true, true, description, verbosity),
      heuristic_cache(HEntry(NO_VALUE, true)), //TODO: is true really a good idea here?
      cache_evaluator_values(cache_estimates),
      task(transform),
      task_proxy(*task) {
}

Heuristic::~Heuristic() {
}

void Heuristic::set_preferred(const OperatorProxy &op) {
    preferred_operators.insert(op.get_ancestor_operator_id(tasks::g_root_task.get()));
}

State Heuristic::convert_ancestor_state(const State &ancestor_state) const {
    return task_proxy.convert_ancestor_state(ancestor_state);
}

void add_heuristic_options_to_feature(
    plugins::Feature &feature, const string &description) {
    feature.add_option<shared_ptr<AbstractTask>>(
        "transform",
        "Optional task transformation for the heuristic."
        " Currently, adapt_costs() and no_transform() are available.",
        "no_transform()");
    feature.add_option<bool>("cache_estimates", "cache heuristic estimates", "true");
    add_evaluator_options_to_feature(feature, description);
}

tuple<shared_ptr<AbstractTask>, bool, string, utils::Verbosity>
get_heuristic_arguments_from_options(const plugins::Options &opts) {
    return tuple_cat(
        make_tuple(
            opts.get<shared_ptr<AbstractTask>>("transform"),
            opts.get<bool>("cache_estimates")
            ),
        get_evaluator_arguments_from_options(opts));
}

EvaluationResult Heuristic::compute_result(EvaluationContext &eval_context) {
    EvaluationResult result;

    assert(preferred_operators.empty());

    const State &state = eval_context.get_state();
    bool calculate_preferred = eval_context.get_calculate_preferred();

    int heuristic = NO_VALUE;

    if (!calculate_preferred && cache_evaluator_values &&
        heuristic_cache[state].h != NO_VALUE && !heuristic_cache[state].dirty) {
        heuristic = heuristic_cache[state].h;
        result.set_count_evaluation(false);
    } else {
        heuristic = compute_heuristic(state);
        if (cache_evaluator_values) {
            heuristic_cache[state] = HEntry(heuristic, false);
        }
        result.set_count_evaluation(true);
    }

    assert(heuristic == DEAD_END || heuristic >= 0);

    if (heuristic == DEAD_END) {
        /*
          It is permissible to mark preferred operators for dead-end
          states (thus allowing a heuristic to mark them on-the-fly
          before knowing the final result), but if it turns out we
          have a dead end, we don't want to actually report any
          preferred operators.
        */
        preferred_operators.clear();
        heuristic = EvaluationResult::INFTY;
    }

#ifndef NDEBUG
    TaskProxy global_task_proxy = state.get_task();
    OperatorsProxy global_operators = global_task_proxy.get_operators();
    if (heuristic != EvaluationResult::INFTY) {
        for (OperatorID op_id : preferred_operators)
            assert(task_properties::is_applicable(global_operators[op_id], state));
    }
#endif

    result.set_evaluator_value(heuristic);
    result.set_preferred_operators(preferred_operators.pop_as_vector());
    assert(preferred_operators.empty());

    return result;
}

EvaluationResultBatched Heuristic::compute_result(EvaluationContextBatched &eval_context) {
    int bs = eval_context.get_batch_size();
    std::vector<State> states = eval_context.get_states();
    std::vector<EvaluationResult> results;
    std::vector<State> non_evaluated_States;
    std::vector<bool> was_cached;

    std::vector<int> h_values;
    std::vector<bool> set_count_evaluations;
    EvaluationResultBatched batched_result;

    bool calculate_preferred =false;

    std::vector<State> non_cached_states;

    for (int i = 0; i < bs; i++) {
        const State &state = states[i];
        if (!calculate_preferred && cache_evaluator_values &&
            heuristic_cache[state].h != NO_VALUE && !heuristic_cache[state].dirty) {
            was_cached.push_back(true);
        } else {
            was_cached.push_back(false);
            non_cached_states.push_back(state);
        }
    }
    std::vector<int> non_cached_h_values;

    if (non_cached_states.size() > 0) {
        non_cached_h_values = compute_heuristic(non_cached_states);
    }
    int current_non_cached = 0;

    for (int i = 0; i < bs; i++) {
        bool cached = was_cached[i];
        if (cached) {
            int h = heuristic_cache[states[i]].h;
            assert(h == DEAD_END || h >= 0);
            if (h == DEAD_END) {
                h = EvaluationResult::INFTY;
            }
            set_count_evaluations.push_back(false);
            h_values.push_back(h);
        } else {
            int h = non_cached_h_values[current_non_cached];
            current_non_cached++;
            assert(h == DEAD_END || h >= 0);
            if (h == DEAD_END) {
                h = EvaluationResult::INFTY;
            }
            set_count_evaluations.push_back(true);
            h_values.push_back(h);
            State state = states[i];
            if (cache_evaluator_values) {
                heuristic_cache[state] = HEntry(h, false);
            }
        }      
    }

    batched_result.set_evaluator_values(h_values);
    batched_result.set_count_evaluations(set_count_evaluations);


    return batched_result;
}

std::vector<int> Heuristic::compute_heuristic(const std::vector<State> &ancestor_states) {
    std::vector<int> h_values;
    for (const State &ancestor_state : ancestor_states) {
        h_values.push_back(compute_heuristic(ancestor_state));
    }
    return h_values;
}

bool Heuristic::does_cache_estimates() const {
    return cache_evaluator_values;
}

bool Heuristic::is_estimate_cached(const State &state) const {
    return heuristic_cache[state].h != NO_VALUE;
}

int Heuristic::get_cached_estimate(const State &state) const {
    assert(is_estimate_cached(state));
    return heuristic_cache[state].h;
}
