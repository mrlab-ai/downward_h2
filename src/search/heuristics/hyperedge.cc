#include "hyperedge.h"

#include <algorithm>
#include <cassert>

Hyperedge::Hyperedge(const std::set<uint32_t>& _tail, uint32_t _head, int _cost, uint32_t operator_id) : tail(_tail), head(_head), cost(_cost), operator_id(operator_id) {
    cost_partition.push_back(cost);
    assert(!tail.empty());
}

// This edge dominates (is prefered during h-computation) other if
// 1) its tail is a subset of the other tail
// 2) the cost of this edge is always smaller or equal
bool Hyperedge::dominates(const Hyperedge& other) const
{
    if (head != other.head)
    {
        return false;
    }

    // Check if tail is a subset of the other
    if (!std::includes(other.tail.begin(), other.tail.end(), tail.begin(), tail.end()))
    {
        return false;
    }

    assert(cost_partition.size() == other.cost_partition.size());

    // Check for cost
    for (size_t i = 0; i < cost_partition.size(); ++i)
    {
        if (other.cost_partition.at(i) < cost_partition.at(i))
        {
            return false;
        }
    }
    return true;
}

bool Hyperedge::operator<(const Hyperedge& other) const
{
    // Compare head first
    if (head < other.head)
        return true;
    if (head > other.head)
        return false;

    // If head is equal, compare tail
    if (tail < other.tail)
        return true;
    if (tail > other.tail)
        return false;

    // If tail is equal, compare cost
    if (cost < other.cost)
        return true;
    if (cost > other.cost)
        return false;

    // If cost is also equal, compare cost partition
    return cost_partition < other.cost_partition;
}

bool Hyperedge::operator==(const Hyperedge& other) const
{
    return (tail == other.tail) && (head == other.head) && (cost == other.cost) && (cost_partition == other.cost_partition);
}

std::ostream& operator<<(std::ostream& os, const Hyperedge& edge) {
    os << "Hyperedge(tail={";
    for (auto it = edge.tail.begin(); it != edge.tail.end(); ++it) {
        if (it != edge.tail.begin()) os << ", ";
        os << *it;
    }
    os << "}, head=" << edge.head
       << ", cost=" << edge.cost
       << ", operator_id=" << edge.operator_id
       << ", cost_partition=[";
    for (size_t i = 0; i < edge.cost_partition.size(); ++i) {
        if (i > 0) os << ", ";
        os << edge.cost_partition[i];
    }
    os << "])";
    return os;
}
