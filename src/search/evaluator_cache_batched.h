#ifndef EVALUATOR_CACHE_BATCHED_H
#define EVALUATOR_CACHE_BATCHED_H

#include "evaluation_result.h"
#include "evaluation_result_batched.h"

#include <unordered_map>

class Evaluator;

using EvaluationResultsBatched = std::unordered_map<Evaluator *, EvaluationResultBatched>;

/*
  Store evaluation results for evaluators.
*/
class EvaluatorCacheBatched {
    EvaluationResultsBatched eval_results;

public:
    EvaluationResultBatched &operator[](Evaluator *eval);

    template<class Callback>
    void for_each_evaluator_result(int i, const Callback &callback) const {
        for (const auto &element : eval_results) {
            const Evaluator *eval = element.first;
            const EvaluationResultBatched &result = element.second;
            callback(eval, result, i);
        }
    }
};

#endif
