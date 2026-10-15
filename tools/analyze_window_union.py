"""Validate opt-in corrected window metadata and bound fixed-CPU-budget copies.

This consumes MAYA_WIN_UNION_DIAG=<prefix> output (<prefix>.cudaN.csv).
It never runs inference and predicts no time savings. Its oracle holds initial
residents on GPU and uses no more CPU rows than the observed layer, ignoring
per-token CPU quotas, CPU cost, overlap and future cache admission.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def analyze_rows(rows):
    layers = defaultdict(list)
    for row in rows:
        row = {key: int(value) for key, value in row.items()}
        layers[(row["device"], row["seq"] - row["token"], row["layer"])].append(row)
    totals = {}
    details = []
    for key, records in sorted(layers.items()):
        first = records[0]
        width = first["T"]
        identity = f"device={key[0]} first_seq={key[1]} layer={key[2]}"

        def require(ok, message):
            if not ok:
                raise ValueError(f"{identity}: {message}")

        require(2 <= width <= 8 and len(records) == width * 8, "incomplete layer")
        require({(r["token"], r["rank"]) for r in records} ==
                {(t, rank) for t in range(width) for rank in range(8)}, "duplicate/missing token ranks")
        for field in ("T", "p0", "gu_type", "down_type", "blob"):
            require(len({r[field] for r in records}) == 1, f"inconsistent {field}")
        records.sort(key=lambda r: (r["token"], r["rank"]))
        experts = defaultdict(list)
        for row in records:
            experts[row["id"]].append(row)
            require(row["seq"] == key[1] + row["token"], "request sequence mismatch")
            require(row["id"] >= 0 and row["blob"] > 0, "invalid ID/blob")
            for field in ("cpu", "miss", "skip", "promo", "fetch_mask", "ram_owned", "ram_pending", "ram_ready"):
                require(row[field] in (0, 1), f"invalid {field}")
            require(row["cpu"] + bool(row["gpu_ptr"]) + row["skip"] == 1,
                    "entry is omitted or assigned to multiple backends")
            require(not row["cpu"] or row["cpu_src"] != 0, "CPU row has no backing")
            require(not row["fetch_src"] or row["gpu_ptr"] != 0, "copy source has no destination")
            require(not row["fetch_mask"] or row["fetch_src"] != 0, "RAM-fetch mask has no source")
            require(not row["promo"] or (row["gpu_ptr"] and row["permanent"]), "promotion has no permanent GPU plan")
            if row["gpu_ptr"]:
                owners = [i for i, item in enumerate(records)
                          if item["gpu_ptr"] == row["gpu_ptr"] and item["fetch_src"]]
                require(len(owners) <= 1, "duplicate GPU copy owners")
                require(row["copy_owner"] == (owners[0] if owners else -1), "copy owner mismatch")
                if owners:
                    require(records[owners[0]]["id"] == row["id"], "different IDs share a GPU copy")
            else:
                require(row["copy_owner"] == -1, "non-GPU entry has a copy owner")
        for uses in experts.values():
            require(len({r["token"] for r in uses}) == len(uses), "expert repeated within one token")
            for field in ("initial_gpu", "initial_ram", "ram_state", "ram_owned", "ram_pending", "ram_ready"):
                require(len({r[field] for r in uses}) == 1, f"inconsistent initial {field}")
            gpu_uses = [r for r in uses if r["gpu_ptr"]]
            require(len({r["gpu_ptr"] for r in gpu_uses}) <= 1, "one expert has multiple GPU destinations")
            if uses[0]["initial_gpu"]:
                require(all(r["gpu_ptr"] == uses[0]["initial_gpu"] for r in uses),
                        "initial resident changed backend/destination")

        label = f"T{width}/GU{first['gu_type']}/D{first['down_type']}"
        if label not in totals:
            totals[label] = dict(layers=0, rows=0, logical_groups=0, cpu_rows=0,
                                 fetched_groups=0, fetched_bytes=0, eligible_layers=0,
                                 eligible_fetched_bytes=0, fixed_budget_removable_bytes_upper=0,
                                 fixed_budget_rows_bound_bytes=0, cpu_only_repeated_groups=0,
                                 cpu_only_repeated_rows=0, mixed_groups=0,
                                 veto_layers=Counter(), multiplicity=defaultdict(Counter))
        total = totals[label]
        total["layers"] += 1
        total["rows"] += len(records)
        total["logical_groups"] += len(experts)
        cpu_rows = sum(r["cpu"] for r in records)
        copies = sum(bool(r["fetch_src"]) for r in records)
        total["cpu_rows"] += cpu_rows
        total["fetched_groups"] += copies
        total["fetched_bytes"] += copies * first["blob"]
        veto = set()
        nonresident_multiplicities = []
        cpu_only_groups = 0
        for uses in experts.values():
            representative = uses[0]
            multiplicity = len(uses)
            cpu = sum(r["cpu"] for r in uses)
            fetched = sum(bool(r["fetch_src"]) for r in uses)
            require(fetched <= 1, "one expert copied more than once")
            resident = bool(representative["initial_gpu"])
            if resident:
                require(not fetched, "initial resident copied again")
                category = "resident"
            else:
                nonresident_multiplicities.append(multiplicity)
                if not representative["ram_ready"]:
                    if representative["ram_pending"]:
                        veto.add("initial_RAM_pending_demotion")
                    elif representative["initial_ram"] and not representative["ram_owned"]:
                        veto.add("initial_RAM_owner_mismatch")
                    else:
                        veto.add("initial_RAM_not_ready_or_unpublished")
                if any(r["miss"] for r in uses):
                    # The device could not obtain a published source. The host
                    # can answer from a pending demotion/lookahead RAM slot;
                    # this bit does not establish an actual disk read.
                    veto.add("host_response_required")
                if representative["ram_ready"]:
                    require(representative["initial_ram"] and representative["ram_owned"] and
                            not representative["ram_pending"] and representative["ram_state"] in (1, 2, 6),
                            "contradictory RAM-ready metadata")
                    require(all(not r["cpu"] or r["cpu_src"] == representative["initial_ram"] for r in uses),
                            "CPU backing differs from initial ready RAM")
                    require(all(not r["fetch_src"] or r["fetch_src"] == representative["initial_ram"] for r in uses),
                            "fetch backing differs from initial ready RAM")
                if cpu == multiplicity:
                    category = "CPU_only"
                    cpu_only_groups += 1
                    if multiplicity > 1:
                        total["cpu_only_repeated_groups"] += 1
                        total["cpu_only_repeated_rows"] += multiplicity
                elif cpu and not any(r["skip"] for r in uses):
                    category = "mixed"
                    total["mixed_groups"] += 1
                    require(fetched == 1, "mixed nonresident group has no copy")
                else:
                    category = "GPU_only_fetched"
                    if not any(r["skip"] for r in uses):
                        require(fetched == 1, "nonresident GPU group has no copy")
            if any(r["skip"] for r in uses):
                veto.add("skipped_expert")
                category = "skipped"
            total["multiplicity"][category][multiplicity] += 1
            if fetched:
                total["multiplicity"]["all_fetched"][multiplicity] += 1
                total["multiplicity"]["permanent_fetch" if any(r["promo"] for r in uses)
                                      else "transient_fetch"][multiplicity] += 1

        # Every group has the same packed bytes within this layer. Sorting its
        # CPU row cost (multiplicity) yields the maximum possible number of
        # all-CPU groups under this aggregate row budget. This is an exact
        # cardinality oracle for that relaxed budget, not a time optimizer.
        best_cpu_groups, used_rows = 0, 0
        for multiplicity in sorted(nonresident_multiplicities):
            if used_rows + multiplicity > cpu_rows:
                break
            used_rows += multiplicity
            best_cpu_groups += 1
        require(best_cpu_groups >= cpu_only_groups, "fixed-budget oracle lost baseline CPU-only groups")
        removable = best_cpu_groups - cpu_only_groups
        relaxed = cpu_rows - cpu_only_groups
        if not veto:
            require(0 <= removable <= copies and removable <= relaxed, "invalid copy ceiling")
        if veto:
            for reason in veto:
                total["veto_layers"][reason] += 1
        else:
            total["eligible_layers"] += 1
            total["eligible_fetched_bytes"] += copies * first["blob"]
            total["fixed_budget_removable_bytes_upper"] += removable * first["blob"]
            total["fixed_budget_rows_bound_bytes"] += relaxed * first["blob"]
        details.append(dict(device=key[0], first_seq=key[1], p0=first["p0"], layer=key[2],
                            T=width, gu_type=first["gu_type"], blob=first["blob"], cpu_rows=cpu_rows,
                            logical_groups=len(experts), fetched_groups=copies,
                            cpu_only_groups=cpu_only_groups, oracle_CPU_groups=best_cpu_groups,
                            oracle_CPU_rows=used_rows, removable_blobs_upper=removable if not veto else None,
                            veto=sorted(veto)))
    if not layers:
        raise ValueError("empty diagnostic trace")
    for total in totals.values():
        denominator = total["eligible_fetched_bytes"]
        total["eligible_copy_reduction_fraction_upper"] = (
            total["fixed_budget_removable_bytes_upper"] / denominator if denominator else None)
        total["all_copy_reduction_fraction_upper"] = (
            total["fixed_budget_removable_bytes_upper"] / total["fetched_bytes"] if total["fetched_bytes"] else None)
    by_width = {}
    for label, total in totals.items():
        width = label.split("/")[0]
        combined = by_width.setdefault(width, dict(veto_layers=Counter(), multiplicity=defaultdict(Counter)))
        for name, value in total.items():
            if isinstance(value, int):
                combined[name] = combined.get(name, 0) + value
        combined["veto_layers"].update(total["veto_layers"])
        for category, histogram in total["multiplicity"].items():
            combined["multiplicity"][category].update(histogram)
    for total in by_width.values():
        denominator = total["eligible_fetched_bytes"]
        total["eligible_copy_reduction_fraction_upper"] = (
            total["fixed_budget_removable_bytes_upper"] / denominator if denominator else None)
        total["all_copy_reduction_fraction_upper"] = (
            total["fixed_budget_removable_bytes_upper"] / total["fetched_bytes"] if total["fetched_bytes"] else None)
    return dict(schema="maya-window-union-analysis-v1", diagnostic=True,
                time_savings_predicted=False,
                mask_semantics={"miss": "host response required; not proof of disk I/O",
                                "fetch_mask": "source present in the published RAM table at route time",
                                "fetch_src": "actual GPU copy source, including host-response sources",
                                "ram_ready": "published, independently owned RAM backing in a ready state with no unretired demotion",
                                "actual_disk_io_recorded": False},
                assumptions=["initial residents stay GPU", "all selected experts evaluated",
                             "eligible layers require all nonresident groups initially ready in RAM",
                             "CPU row budget may move freely among tokens within one layer",
                             "no increase in aggregate CPU rows per layer", "fixed selected routes/cache state",
                             "ignores CPU/GPU cost, overlap, admission changes and planner overhead"],
                window_totals=by_width, totals=totals, layers=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.traces:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    result = analyze_rows(rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for label, total in result["window_totals"].items():
        fraction = total["eligible_copy_reduction_fraction_upper"]
        percent = f"{100 * fraction:.3f}%" if fraction is not None else "n/a"
        print(f"{label}: {total['eligible_layers']}/{total['layers']} RAM-ready layers, "
              f"copy ceiling {percent} eligible bytes; {total['veto_layers']}")


if __name__ == "__main__":
    main()
