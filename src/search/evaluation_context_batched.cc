#include "evaluation_context_batched.h"
#include "evaluation_context.h"

#include "evaluation_result.h"
#include "evaluator.h"
#include "search_statistics.h"

#include <cassert>

using namespace std;

EvaluationContextBatched::EvaluationContextBatched(
    const EvaluatorCacheBatched &cache, const std::vector<State> &states, std::vector<int> &g_values,
    bool is_preferred, SearchStatistics *statistics,
    bool calculate_preferred)
    : cache(cache),
        states(states),
        g_values(g_values),
        preferred(is_preferred),
        statistics(statistics),
        calculate_preferred(calculate_preferred) {
}


EvaluationContextBatched::EvaluationContextBatched(
    const std::vector<State> &state, std::vector<int> &g_values, bool is_preferred,
        SearchStatistics *statistics, bool calculate_preferred)
    : EvaluationContextBatched(EvaluatorCacheBatched(), state, g_values, is_preferred,
                        statistics, calculate_preferred) {
}



EvaluationResultBatched &EvaluationContextBatched::get_result(Evaluator *evaluator) {
    EvaluationResultBatched &result = cache[evaluator];
    if (result.is_uninitialized()) {
        result = evaluator->compute_result(*this);
        std::vector<bool> count = result.get_count_evaluations();
        size_t bs = count.size();
        for (size_t i = 0; i < bs; i++) { //TODO: OK, this is definitely not the way to do it
            if (statistics &&
                evaluator->is_used_for_counting_evaluations() &&
                count[i]) {
                statistics->inc_evaluations();
            }
        }
    }
    return result;
}

const EvaluatorCacheBatched &EvaluationContextBatched::get_cache() const {
    return cache;
}

const State &EvaluationContextBatched::get_state() const {
    return states[0];
}

const std::vector<State> &EvaluationContextBatched::get_states() const {
    return states;
}

int EvaluationContextBatched::get_batch_size() const {
    return states.size();
}

int EvaluationContextBatched::get_g_value() const {
    return g_values[0];
}
std::vector<int> EvaluationContextBatched::get_g_values() const {
    return g_values;
}

bool EvaluationContextBatched::is_preferred() const {
    return preferred;
}

std::vector<bool> EvaluationContextBatched::is_evaluator_value_infinite(Evaluator *eval) {
    (void)eval;
    throw::runtime_error("Not implemented, use the indexed version instead");
}

bool EvaluationContextBatched::is_evaluator_value_infinite(Evaluator *eval, int idx) {
    EvaluationResultBatched evaluation_results = get_result(eval);
    return evaluation_results.is_infinite()[idx];
}


std::vector<int> EvaluationContextBatched::get_evaluator_value(Evaluator *eval) {
    EvaluationResultBatched evaluation_results = get_result(eval);
    std::vector<int> h = evaluation_results.get_evaluator_values();
    return h;
}

std::vector<int> EvaluationContextBatched::get_evaluator_value_or_infinity(Evaluator *eval) {
    EvaluationResultBatched evaluation_results = get_result(eval);
    std::vector<int> results = evaluation_results.get_evaluator_values();
    return results;
}

const vector<OperatorID> &
EvaluationContextBatched::get_preferred_operators(Evaluator *eval) {
    (void)eval;
    throw::runtime_error("Not implemented");
}


bool EvaluationContextBatched::get_calculate_preferred() const {
    return calculate_preferred;
}
