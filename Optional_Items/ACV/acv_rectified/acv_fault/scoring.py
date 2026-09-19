"""The competition's linear rank-decay scoring metric (see the info kit,
Section 4): for a file with n cars where the true faulty car lands at rank r
(1 = most likely), score = (n - (r - 1)) / n.
"""
from __future__ import annotations


def rank_decay_score(ranked_car_ids: list[str], true_faulty_car: str) -> float:
    n = len(ranked_car_ids)
    if true_faulty_car not in ranked_car_ids:
        return 0.0
    r = ranked_car_ids.index(true_faulty_car) + 1
    return (n - (r - 1)) / n
