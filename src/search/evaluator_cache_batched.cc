#include "evaluator_cache_batched.h"

using namespace std;


EvaluationResultBatched &EvaluatorCacheBatched::operator[](Evaluator *eval) {
    return eval_results[eval];
}
