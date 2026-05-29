#ifndef EVALUATION_RESULT_BATCHED_H
#define EVALUATION_RESULT_BATCHED_H

#include "operator_id.h"

#include <limits>
#include <vector>

class EvaluationResultBatched {
    static const int UNINITIALIZED = -2;
    bool uninitialized = true;

    std::vector<int> evaluator_values;
    std::vector<bool> count_evaluations;
public:
    // "INFINITY" is an ISO C99 macro and "INFINITE" is a macro in windows.h.
    static const int INFTY;

    EvaluationResultBatched();

    /* TODO: Can we do without this "uninitialized" business?

       One reason why it currently exists is to simplify the
       implementation of the EvaluationContext class, where we don't
       want to perform two separate map operations in the case where
       we determine that an entry doesn't yet exist (lookup) and hence
       needs to be created (insertion). This can be avoided most
       easily if we have a default constructor for EvaluationResult
       and if it's easy to test if a given object has just been
       default-constructed.
    */

    /*
      TODO: Can we get rid of count_evaluation?
      The EvaluationContext needs to know (for statistics) if the
      heuristic actually computed the heuristic value or just looked it
      up in a cache. Currently this information is passed over the
      count_evaluation flag, which is somewhat awkward.
    */
   

    bool is_uninitialized() const;

    std::vector<bool> is_infinite() const;
    bool is_infinite(int i) const;

    std::vector<int> get_evaluator_values() const;
    int get_evaluator_value(int i) const;

    std::vector<bool> get_count_evaluations() const;
    bool get_count_evaluation(int i) const;

    const std::vector<OperatorID> &get_preferred_operators() const; // <- we don't need that

    void set_evaluator_values(std::vector<int> value);
    void set_count_evaluations(std::vector<bool> count_eval);
};

#endif
