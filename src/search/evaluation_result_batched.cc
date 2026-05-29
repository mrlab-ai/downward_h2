#include "evaluation_result_batched.h"

using namespace std;

const int EvaluationResultBatched::INFTY = numeric_limits<int>::max();

EvaluationResultBatched::EvaluationResultBatched() : 
evaluator_values({}), //for some reason, linker cannot find UNINITIALIZED
count_evaluations({}) {
}

bool EvaluationResultBatched::is_uninitialized() const {
    return uninitialized;
}

std::vector<bool> EvaluationResultBatched::is_infinite() const {
    std::vector<bool> results;
    for (size_t i = 0; i < evaluator_values.size(); i++) {
        results.push_back(evaluator_values[i] == INFTY);
    }
    return results;
}

bool EvaluationResultBatched::is_infinite(int i) const {
    return evaluator_values[i] == INFTY;
}


std::vector<int> EvaluationResultBatched::get_evaluator_values() const {
    return evaluator_values;
}

int EvaluationResultBatched::get_evaluator_value(int i) const {
    return evaluator_values[i]; 
}


std::vector<bool>  EvaluationResultBatched::get_count_evaluations() const {
    return count_evaluations;
}

bool  EvaluationResultBatched::get_count_evaluation(int i) const {
    return count_evaluations[i];
}

void EvaluationResultBatched::set_evaluator_values(std::vector<int> values) {
    uninitialized = false;
    evaluator_values = values;
}

void EvaluationResultBatched::set_count_evaluations(std::vector<bool> count_eval) {
    uninitialized = false;
    count_evaluations = count_eval;
}
