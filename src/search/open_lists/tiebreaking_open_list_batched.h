#ifndef OPEN_LISTS_TIEBREAKING_OPEN_LIST_BATCHED_H
#define OPEN_LISTS_TIEBREAKING_OPEN_LIST_BATCHED_H

#include "../open_list_factory.h"

namespace tiebreaking_open_list_batched {
class TieBreakingOpenListBatchedFactory : public OpenListFactory {
    std::vector<std::shared_ptr<Evaluator>> evals;
    bool unsafe_pruning;
    bool pref_only;
public:
    TieBreakingOpenListBatchedFactory(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        bool unsafe_pruning, bool pref_only);

    virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;

    
};
}

#endif
