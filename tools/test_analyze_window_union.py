"""CPU-only integrity/oracle checks; no model, CUDA, or inference dependency."""

from copy import deepcopy
from itertools import combinations
import random
import unittest

from analyze_window_union import analyze_rows


def trace(ids, cpu, residents):
    rows, owners = [], {}
    for token, route in enumerate(ids):
        for rank, expert in enumerate(route):
            host = int((token, expert) in cpu)
            pointer = 0 if host else 500000 + expert * 100
            source = 0
            if not host and expert not in residents and expert not in owners:
                source = 100000 + expert * 100
                owners[expert] = 8 * token + rank
            rows.append(dict(device=0, p0=40, layer=3, T=len(ids), gu_type=10, down_type=10,
                             blob=1000, seq=10 + token, token=token, rank=rank, id=expert,
                             initial_gpu=500000 + expert * 100 if expert in residents else 0,
                             initial_ram=100000 + expert * 100, ram_state=1, ram_owned=1,
                             ram_pending=0, ram_ready=1, cpu=host, miss=0, skip=0, promo=0,
                             fetch_mask=int(bool(source)), gpu_ptr=pointer, fetch_src=source,
                             cpu_src=100000 + expert * 100 if host else 0, copy_owner=-1,
                             permanent=int(expert in residents)))
    for row in rows:
        if row["gpu_ptr"]:
            row["copy_owner"] = owners.get(row["id"], -1)
    return rows


class WindowUnionTests(unittest.TestCase):
    def setUp(self):
        self.rows = trace([list(range(8)), [0, 1, 2, 8, 9, 10, 11, 12]],
                          {(0, 1), (0, 2), (1, 2), (0, 3)}, {0})

    def total(self, rows):
        return analyze_rows(rows)["window_totals"]["T2"]

    def test_actual_backends_residency_and_bound(self):
        result = self.total(self.rows)
        self.assertEqual((result["cpu_rows"], result["cpu_only_repeated_groups"],
                          result["cpu_only_repeated_rows"], result["mixed_groups"]), (4, 1, 2, 1))
        self.assertEqual(result["fetched_groups"], 10)
        self.assertEqual(result["fixed_budget_removable_bytes_upper"], 2000)
        self.assertEqual(result["multiplicity"]["resident"][2], 1)
        self.assertEqual(result["multiplicity"]["all_fetched"][2], 1)

    def test_pending_pointer_is_not_ready_backing(self):
        rows = deepcopy(self.rows)
        for row in rows:
            if row["id"] == 4:
                # An unpublished pending-D2H source goes through the host
                # response path even when no storage read is necessary.
                row.update(initial_ram=0, ram_ready=0, ram_owned=0, ram_pending=1,
                           ram_state=3, miss=1, fetch_mask=0)
        result = self.total(rows)
        self.assertEqual(result["eligible_layers"], 0)
        self.assertEqual(result["veto_layers"]["initial_RAM_pending_demotion"], 1)
        self.assertEqual(result["veto_layers"]["host_response_required"], 1)
        self.assertNotIn("disk_miss", result["veto_layers"])
        self.assertEqual(result["fixed_budget_removable_bytes_upper"], 0)

    def test_bad_owner_and_missing_rank_fail_validation(self):
        rows = deepcopy(self.rows)
        rows[-1]["copy_owner"] = -1
        with self.assertRaisesRegex(ValueError, "copy owner mismatch"):
            self.total(rows)
        with self.assertRaisesRegex(ValueError, "incomplete layer"):
            self.total(self.rows[:-1])

    def test_cardinality_oracle_against_exhaustive_subsets(self):
        rng = random.Random(1701)
        for _ in range(25):
            ids = [rng.sample(range(12), 8), rng.sample(range(12), 8)]
            residents = {expert for expert in range(12) if rng.random() < 0.2}
            cpu, on_gpu = set(), set(residents)
            for token, route in enumerate(ids):
                for expert in route:
                    if expert not in on_gpu and rng.random() < 0.6:
                        cpu.add((token, expert))
                    else:
                        on_gpu.add(expert)
            result = analyze_rows(trace(ids, cpu, residents))
            costs = [sum(expert in route for route in ids)
                     for expert in set(ids[0] + ids[1]) - residents]
            best = max((len(subset) for count in range(len(costs) + 1)
                        for subset in combinations(costs, count) if sum(subset) <= len(cpu)), default=0)
            self.assertEqual(result["layers"][0]["oracle_CPU_groups"], best)


if __name__ == "__main__":
    unittest.main()
