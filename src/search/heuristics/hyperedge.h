#ifndef HM2_BASE_HYPEREDGE_H
#define HM2_BASE_HYPEREDGE_H

#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

struct Hyperedge
{
    std::set<uint32_t> tail;
    uint32_t head;
    int64_t cost;
    uint32_t operator_id;
    // Last entry is always the original cost function
    // and preceding entries are admissible cost partitions.
    // => we want to max(original, sum(partitions))
    std::vector<int> cost_partition;

    Hyperedge(const std::set<uint32_t>& _tail, uint32_t _head, int _cost, uint32_t operator_id);

    // This edge dominates (is prefered during h-computation) other
    bool dominates(const Hyperedge& other) const;

    // Less than operator for sorting
    bool operator<(const Hyperedge& other) const;

    // Equality operator
    bool operator==(const Hyperedge& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Hyperedge& edge);
};

#endif