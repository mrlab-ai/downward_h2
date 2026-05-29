#include "eager_search_batched.h"

#include "../evaluation_context.h"
#include "../evaluation_context_batched.h"
#include "../evaluator.h"
#include "../heuristics/hm2_base.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../plugins/options.h"
#include "../task_utils/successor_generator.h"
#include "../utils/logging.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>

using namespace std;

namespace eager_search_batched {
EagerSearchBatched::EagerSearchBatched(
    const shared_ptr<OpenListFactory> &open, bool reopen_closed,
    const shared_ptr<Evaluator> &f_eval,
    const vector<shared_ptr<Evaluator>> &preferred,
    const shared_ptr<PruningMethod> &pruning,
    const shared_ptr<Evaluator> &lazy_evaluator, OperatorCost cost_type,
    int bound, double max_time, const string &description,
    utils::Verbosity verbosity)
    : SearchAlgorithm(
          cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      open_list(open->create_state_open_list()),
      f_evaluator(f_eval),     // default nullptr
      preferred_operator_evaluators(preferred),
      lazy_evaluator(lazy_evaluator),     // default nullptr
      pruning_method(pruning) {
    if (lazy_evaluator && !lazy_evaluator->does_cache_estimates()) {
        cerr << "lazy_evaluator must cache its estimates" << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void EagerSearchBatched::initialize() {
    log << "Conducting best first search"
        << (reopen_closed_nodes ? " with" : " without")
        << " reopening closed nodes, (real) bound = " << bound
        << endl;
    assert(open_list);

    set<Evaluator *> evals;
    open_list->get_path_dependent_evaluators(evals);

    /*
      Collect path-dependent evaluators that are used for preferred operators
      (in case they are not also used in the open list).
    */
    for (const shared_ptr<Evaluator> &evaluator : preferred_operator_evaluators) {
        evaluator->get_path_dependent_evaluators(evals);
    }

    /*
      Collect path-dependent evaluators that are used in the f_evaluator.
      They are usually also used in the open list and will hence already be
      included, but we want to be sure.
    */
    if (f_evaluator) {
        f_evaluator->get_path_dependent_evaluators(evals);
    }

    /*
      Collect path-dependent evaluators that are used in the lazy_evaluator
      (in case they are not already included).
    */
    if (lazy_evaluator) {
        lazy_evaluator->get_path_dependent_evaluators(evals);
    }

    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    /*
      Note: we consider the initial state as reached by a preferred
      operator.
    */
    EvaluationContext eval_context(initial_state, 0, true, &statistics);

    statistics.inc_evaluated_states();

    if (open_list->is_dead_end(eval_context)) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context))
            statistics.print_checkpoint_line(0);
        start_f_value_statistics(eval_context);
        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        open_list->insert(eval_context, initial_state.get_id());
    }

    print_initial_evaluator_values(eval_context);

    pruning_method->initialize(task);
}

void EagerSearchBatched::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

SearchStatus EagerSearchBatched::step() {
    optional<SearchNode> node;
    while (true) {
        if (open_list->empty()) {
            log << "Completely explored state space -- no solution!" << endl;
            return FAILED;
        }
        StateID id = open_list->remove_min();
        State s = state_registry.lookup_state(id);
        node.emplace(search_space.get_node(s));


        if (node->is_closed())
            continue;

        /*
          We can pass calculate_preferred=false here since preferred
          operators are computed when the state is expanded.
        */
        std::vector<State> states = {s};
        std::vector<int> g_values = {node->get_g()};
        EvaluationContextBatched eval_context(states, g_values, false, &statistics);

        node->close();
        assert(!node->is_dead_end());
        update_f_value_statistics(eval_context);
        statistics.inc_expanded();
        break;
    }

    const State &s = node->get_state();
    if (check_goal_and_set_plan(s))
        return SOLVED;

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(s, applicable_ops);

    /*
      TODO: When preferred operators are in use, a preferred operator will be
      considered by the preferred operator queues even when it is pruned.
    */
    pruning_method->prune_operators(s, applicable_ops);
    
    std::vector<State> succ_states_batch;
    std::vector<int> succ_g_values;
    std::vector<SearchNode> succ_nodes_batch;
    std::vector<int> succ_nodes_indices;
    std::vector<OperatorID> applicable_ops_batch;

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        if ((node->get_real_g() + op.get_cost()) >= bound)
            continue;

        State succ_state = state_registry.get_successor_state(s, op);
        statistics.inc_generated();
        bool is_preferred = false;

        SearchNode succ_node = search_space.get_node(succ_state);

        for (Evaluator *evaluator : path_dependent_evaluators) {
            evaluator->notify_state_transition(s, op_id, succ_state);
        }

        // Previously encountered dead end. Don't re-evaluate.
        if (succ_node.is_dead_end())
            continue;
        
        if (succ_node.is_new()) {
            int succ_node_idx = succ_nodes_batch.size();
            succ_nodes_batch.push_back(succ_node);
            // We have not seen this st'ate before.
            // Evaluate and create a new node.

            // Careful: succ_node.get_g() is not available here yet,
            // hence the stupid computation of succ_g.
            // TODO: Make this less fragile.
            int succ_g = node->get_g() + get_adjusted_cost(op);

            auto it = std::find(succ_states_batch.begin(), succ_states_batch.end(), succ_state);
            if (it != succ_states_batch.end()) {
                int batch_index = std::distance(succ_states_batch.begin(), it);
                if (succ_g < succ_g_values[batch_index]) {
                    succ_g_values[batch_index] = succ_g;
                    applicable_ops_batch[batch_index] = op_id;
                    succ_nodes_indices[batch_index] = succ_node_idx;
                }
            } else {
                statistics.inc_evaluated_states();

                succ_states_batch.push_back(succ_state);
                succ_g_values.push_back(succ_g);
                applicable_ops_batch.push_back(op_id);
                succ_nodes_indices.push_back(succ_node_idx);
            }
            


            //EvaluationContext succ_eval_context(
            //    succ_state, succ_g, is_preferred, &statistics);
            //
            //if (open_list->is_dead_end(succ_eval_context)) {
            //    succ_node.mark_as_dead_end();
            //    statistics.inc_dead_ends();
            //    continue;
            //}
            //succ_node.open(*node, op, get_adjusted_cost(op));
            //open_list->insert(succ_eval_context, succ_state.get_id());
            //if (search_progress.check_progress(succ_eval_context)) {
            //    statistics.print_checkpoint_line(succ_node.get_g());
            //    reward_progress();
            //}
        } else if (succ_node.get_g() > node->get_g() + get_adjusted_cost(op)) {
            // We found a new cheapest path to an open or closed state.
            if (reopen_closed_nodes) {
                if (succ_node.is_closed()) {
                    /*
                      TODO: It would be nice if we had a way to test
                      that reopening is expected behaviour, i.e., exit
                      with an error when this is something where
                      reopening should not occur (e.g. A* with a
                      consistent heuristic).
                    */
                    statistics.inc_reopened();
                }
                succ_node.reopen(*node, op, get_adjusted_cost(op));

                EvaluationContext succ_eval_context(
                    succ_state, succ_node.get_g(), is_preferred, &statistics);

                /*
                  Note: our old code used to retrieve the h value from
                  the search node here. Our new code recomputes it as
                  necessary, thus avoiding the incredible ugliness of
                  the old "set_evaluator_value" approach, which also
                  did not generalize properly to settings with more
                  than one evaluator.

                  Reopening should not happen all that frequently, so
                  the performance impact of this is hopefully not that
                  large. In the medium term, we want the evaluators to
                  remember evaluator values for states themselves if
                  desired by the user, so that such recomputations
                  will just involve a look-up by the Evaluator object
                  rather than a recomputation of the evaluator value
                  from scratch.
                */
                open_list->insert(succ_eval_context, succ_state.get_id());
            } else {
                // If we do not reopen closed nodes, we just update the parent pointers.
                // Note that this could cause an incompatibility between
                // the g-value and the actual path that is traced back.
                succ_node.update_parent(*node, op, get_adjusted_cost(op));
            }
        }
    }
    bool is_preferred = false;
    EvaluationContextBatched succ_eval_context_batch(
        succ_states_batch, succ_g_values, is_preferred, &statistics);

    size_t batch_size = succ_states_batch.size();
    for (size_t i = 0; i < batch_size; i++) {
        OperatorID op_id = applicable_ops_batch[i];
        OperatorProxy op = task_proxy.get_operators()[op_id];
        int succ_node_id = succ_nodes_indices[i];
        SearchNode &succ_node = succ_nodes_batch[succ_node_id];
        State &succ_state = succ_states_batch[i];
        
        if (open_list->is_dead_end(succ_eval_context_batch, i)) {
            succ_node.mark_as_dead_end();
            statistics.inc_dead_ends();
            continue; 
        }
        succ_node.open(*node, op, get_adjusted_cost(op));

        open_list->insert(succ_eval_context_batch, succ_state.get_id(), i);
        if (search_progress.check_progress(succ_eval_context_batch)) {
            statistics.print_checkpoint_line(succ_node.get_g());
            reward_progress();
        }


    }

    return IN_PROGRESS;
}

void EagerSearchBatched::reward_progress() {
    // Boost the "preferred operator" open lists somewhat whenever
    // one of the heuristics finds a state with a new best h value.
    open_list->boost_preferred();
}

void EagerSearchBatched::dump_search_space() const {
    search_space.dump(task_proxy);
}

void EagerSearchBatched::start_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}

/* TODO: HACK! This is very inefficient for simply looking up an h value.
   Also, if h values are not saved it would recompute h for each and every state. */
void EagerSearchBatched::update_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        int f_value = eval_context.get_evaluator_value(f_evaluator.get());
        statistics.report_f_value_progress(f_value);
    }
}

void EagerSearchBatched::update_f_value_statistics(EvaluationContextBatched &eval_context) {
    if (f_evaluator) { 
        std::vector<int> f_values = eval_context.get_evaluator_value(f_evaluator.get());
        for (int f_value : f_values) {
            statistics.report_f_value_progress(f_value);    
        }
    }
}

void add_eager_search_options_to_feature(
    plugins::Feature &feature, const string &description) {
    add_search_pruning_options_to_feature(feature);
    // We do not add a lazy_evaluator options here
    // because it is only used for astar but not the other plugins.
    add_search_algorithm_options_to_feature(feature, description);
}

tuple<shared_ptr<PruningMethod>, shared_ptr<Evaluator>, OperatorCost,
      int, double, string, utils::Verbosity>
get_eager_search_arguments_from_options(const plugins::Options &opts) {
    return tuple_cat(
        get_search_pruning_arguments_from_options(opts),
        make_tuple(opts.get<shared_ptr<Evaluator>>(
                       "lazy_evaluator", nullptr)),
        get_search_algorithm_arguments_from_options(opts)
        );
}
}
