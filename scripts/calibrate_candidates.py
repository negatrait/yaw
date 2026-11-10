"""Fit fixed candidate cutoffs to frozen reference counts; never impose batch quotas."""

import argparse
import json
import math
from contextlib import closing
from pathlib import Path

import numpy as np

from yaw.core import load_config
from yaw.daemon.graph.storage import find_entity_candidates, open_readonly_yaw_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="JSON cases: entity_ids, candidate_count")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--volume-ratio", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.volume_ratio is None:
        args.volume_ratio = config.candidate_volume_ratio
    if not math.isfinite(args.volume_ratio) or args.volume_ratio <= 0:
        parser.error("volume ratio must be finite and positive")
    cases = json.loads(args.reference.read_text(encoding="utf-8"))["cases"]
    if not cases:
        parser.error("reference must contain at least one case")
    bounds = [
        (
            math.ceil(0.9 * args.volume_ratio * case["candidate_count"]),
            math.floor(1.1 * args.volume_ratio * case["candidate_count"]),
        )
        for case in cases
    ]
    with closing(open_readonly_yaw_db(args.db)) as conn:
        conn.execute("BEGIN")
        dense_cutoffs = np.arange(0.65, 1.0001, 0.0025)
        lexical_cutoffs = np.arange(0.25, 1.0001, 0.0025)
        counts = []
        for case in cases:
            pairs = find_entity_candidates(conn, case["entity_ids"], dense_threshold=0.65, lexical_threshold=0.25)
            scores = np.array([(dense, lexical) for _, _, dense, lexical in pairs], dtype=np.float64).reshape(-1, 2)
            counts.append(
                np.array(
                    [
                        len(scores)
                        - np.searchsorted(np.sort(scores[scores[:, 0] < d, 1]), lexical_cutoffs, side="left")
                        for d in dense_cutoffs
                    ]
                )
            )
        feasible = np.ones(counts[0].shape, dtype=bool)
        error = np.zeros(counts[0].shape)
        for count, case, (lo, hi) in zip(counts, cases, bounds, strict=True):
            feasible &= (count >= lo) & (count <= hi)
            error += np.abs(count - case["candidate_count"] * args.volume_ratio)
        if not feasible.any():
            raise ValueError("No fixed cutoff pair meets every volume band; inspect the reference cases and score ties")
        error += np.abs(dense_cutoffs[:, None] - 0.8) * 0.01
        error[~feasible] = np.inf
        di, li = np.unravel_index(error.argmin(), error.shape)
        dense, lexical = round(float(dense_cutoffs[di]), 4), round(float(lexical_cutoffs[li]), 4)
        actual = [
            len(find_entity_candidates(conn, case["entity_ids"], dense_threshold=dense, lexical_threshold=lexical))
            for case in cases
        ]
        if not all(lo <= count <= hi for count, (lo, hi) in zip(actual, bounds, strict=True)):
            raise ValueError("Selected cutoff counts changed during verification")
        print(
            json.dumps(
                {
                    "candidate_dense_threshold": dense,
                    "candidate_lexical_threshold": lexical,
                    "candidate_volume_ratio": args.volume_ratio,
                    "reference_counts": [case["candidate_count"] for case in cases],
                    "candidate_counts": actual,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
