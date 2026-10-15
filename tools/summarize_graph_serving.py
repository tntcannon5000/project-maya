"""Validate engagement and summarize the bounded graph serving comparison."""
import argparse
import json
from pathlib import Path

from summarize_window_graphs import COUNTS


def summarize(off, on):
    result = {"scope": "One short coding prompt; repeated requests, not independent trials. No profiler. Raw per-round drafter timing is enabled equally."}
    manifests = []
    requests = []
    for label, directory in (("off", off), ("on", on)):
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = json.loads((directory / "requests.json").read_text())
        log = (directory / "engine.log").read_text(errors="replace")
        manifests.append(manifest)
        requests.append(rows)
        counts = [list(map(int, row)) for row in COUNTS.findall(log)]
        if label == "on":
            if not counts or sum(row[2] for row in counts) == 0:
                raise ValueError("Graph replay did not engage; this is not a graph comparison")
            if any(sum(row[6:9]) for row in counts):
                raise ValueError("Graph capture/fallback/launch failures occurred")
        result[label] = {
            "rows": [{k: row[k] for k in ("kind", "width", "tokens", "tps", "prompt_ms", "acceptance", "draft_ms", "verify_ms")} for row in rows],
            "captures": sum(row[1] for row in counts),
            "replays": sum(row[2] for row in counts),
            "memory_and_graph_lines": [line for line in log.splitlines() if any(s in line for s in ("expert pool ", "RAM tier ", "MoE graph CUDA0 host totals", "MoE graph CUDA0 bypass reasons"))],
        }
    for key in ("binary_sha256", "drafter_sha256", "context_capacity", "prompt_ids", "output_cap", "width_order"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError("Unmatched configuration: " + key)
    ignored = {"MAYA_WIN_MOE_GRAPH", "STRATA_GLM_DFLASH_STATS"}
    envs = [{k: v for k, v in m["environment"].items() if k not in ignored} for m in manifests]
    if envs[0] != envs[1]:
        raise ValueError("Environment differs beyond graph flag and output path")
    paired = []
    if len(requests[1]) < len(requests[0]):
        raise ValueError("Candidate request schedule is incomplete")
    for a, b in zip(requests[0], requests[1]):
        if (a["kind"], a["width"]) != (b["kind"], b["width"]):
            raise ValueError("Unmatched request schedule")
        paired.append(dict(kind=a["kind"], width=a["width"], off_tps=a["tps"], on_tps=b["tps"],
                           change_percent=100 * (b["tps"] / a["tps"] - 1),
                           output_ids_exact=a["output_ids"] == b["output_ids"],
                           offered_equal=a["offered"] == b["offered"],
                           accepted_equal=a["accepted"] == b["accepted"]))
    result["paired"] = paired
    result["occupied_prompt_tokens"] = manifests[0]["occupied_prompt_tokens"]
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("off", type=Path)
    ap.add_argument("on", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.off, args.on)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["paired"], indent=2))
