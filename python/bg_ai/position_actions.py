"""Count and construct physical board drags, using original slot identities.

A drag removes one minion and inserts it into a new slot; it is not a swap.
Slots in each emitted action refer to the board after all preceding actions.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def _validate_order(order: Sequence[int]) -> tuple[int, ...]:
    result = tuple(order)
    if len(result) > 7 or any(type(i) is not int for i in result) or sorted(result) != list(range(len(result))):
        raise ValueError("Order must be a permutation of zero to six original board indices")
    return result


def _stationary_indices(order: tuple[int, ...]) -> tuple[int, ...]:
    """The longest increasing subsequence can stay put while other cards move."""
    paths: list[tuple[int, ...]] = []
    for i, value in enumerate(order):
        prefix = max((paths[j] for j in range(i) if order[j] < value), key=len, default=())
        paths.append((*prefix, value))
    return max(paths, key=len, default=())


def minimum_drag_count(order: Sequence[int]) -> int:
    """Exact minimum number of remove-and-insert actions from the current order."""
    target = _validate_order(order)
    return len(target) - len(_stationary_indices(target))


def drag_plan(order: Sequence[int]) -> list[dict[str, int | str]]:
    """Return a shortest executable sequence; repeated card IDs remain distinct."""
    target = _validate_order(order)
    stationary = set(_stationary_indices(target))
    current = list(range(len(target)))
    actions: list[dict[str, int | str]] = []
    for target_index in range(len(target) - 1, -1, -1):
        original_index = target[target_index]
        if original_index in stationary:
            continue
        source_index = current.index(original_index)
        current.pop(source_index)
        # Place this card immediately before its already positioned successor.
        destination_index = (current.index(target[target_index + 1])
                             if target_index + 1 < len(target) else len(current))
        current.insert(destination_index, original_index)
        actions.append({"kind": "move", "original_index": original_index,
                        "from_index": source_index, "to_index": destination_index})
    assert tuple(current) == target
    assert len(actions) == len(target) - len(stationary)
    return actions


def affordable_orders(orders: Iterable[Sequence[int]], max_drags: int) -> list[tuple[int, ...]]:
    """Filter before neural scoring or search; preserve input order and identity."""
    if type(max_drags) is not int or max_drags < 0:
        raise ValueError("Maximum drags must be a nonnegative integer")
    result = []
    for order in orders:
        target = _validate_order(order)
        if minimum_drag_count(target) <= max_drags:
            result.append(target)
    return result
