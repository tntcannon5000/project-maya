"""Offline causal residency research; never launches inference or a build.

Uses the existing churn replay's capacities, CPU plan and spare semantics.
Static models train only on writing/web. Mixture rules are fixed before
evaluation; CSV is a held-out domain, while web is a training-domain check.
Counterfactual traffic is approximate and is not an engine/TPS result.
"""
import argparse
import collections
import hashlib
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

import audit_residency_churn as churn

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ("train-writing", "train-web")
EVAL = ("current-csv", "current-web")
MAX_MIX = 0.25
PRIOR_ANCHORS = 16.0
ONLINE_CONFIDENCE_ANCHORS = 128.0
DRIFT_WINDOW = 32


def load_run(folder):
    rows = churn.load(folder / "routes.0")
    manifest = json.loads((folder / "manifest.json").read_text())
    nt = len((folder / "tokens.txt").read_text().split())
    prompt, repeats = int(manifest["command"][-3]), int(manifest["command"][-1])
    phases = {t: f"r{t // nt}-" + ("prompt" if t % nt < prompt else "decode")
              for t in range(nt * repeats)}
    assert len(rows) == nt * repeats * 42
    return rows, phases, nt


def train_window(horizon):
    counts = np.zeros((46, 288, 288), dtype=np.float64)
    seen = np.zeros((46, 288), dtype=np.float64)
    marginal = np.zeros((46, 288), dtype=np.float64)
    anchors = collections.Counter()
    for name in TRAIN:
        rows, _, nt = load_run(ROOT / "docs/evidence/maya-demand" / name)
        sequences = collections.defaultdict(list)
        for row in rows:
            sequences[row["t"] // nt, row["l"]].append(row["ids"])
        for (_, layer), seq in sequences.items():
            for ids in seq:
                marginal[layer, ids] += 1
            # Do not create labels across request boundaries or right censoring.
            for j in range(len(seq) - horizon):
                future = sorted({e for ids in seq[j + 1:j + horizon + 1] for e in ids})
                for expert in seq[j]:
                    seen[layer, expert] += 1
                    counts[layer, expert, future] += 1
                anchors[layer] += 1
    rates = marginal / marginal.sum(axis=1, keepdims=True).clip(min=1)
    prior = 1 - (1 - rates) ** (8 * horizon)
    probabilities = (counts + PRIOR_ANCHORS * prior[:, None, :]) / (seen[:, :, None] + PRIOR_ANCHORS)
    return probabilities, rates, dict(anchors)


def normalize(score):
    return score / max(float(score.sum()), 1e-30)


def replay(rows, phases, sizes, caps, prior, mode, model=None, train_rates=None):
    cache = {layer: set(prior[layer][:caps[layer] - 3]) for layer in caps}
    frequency = np.zeros((46, 288), dtype=np.float64)
    transitions = np.zeros((46, 288, 288), dtype=np.float64)
    seen = np.zeros((46, 288), dtype=np.float64)
    window_counts = np.zeros_like(transitions)
    window_seen = np.zeros_like(seen)
    recent = collections.defaultdict(lambda: collections.deque(maxlen=DRIFT_WINDOW))
    labels = collections.defaultdict(lambda: collections.deque(maxlen=17))
    previous, last = {}, {}
    stats = collections.defaultdict(collections.Counter)
    events = 0
    for row in rows:
        layer, ids, token = row["l"], row["ids"], row["t"]
        stat = stats[phases[token]]
        stat["routes"] += 1
        missing = [(i, e) for i, e in enumerate(ids) if e not in cache[layer]]
        for expert in ids:
            frequency[layer, expert] += 1
            last[layer, expert] = token
        cpu = {e for i, e in sorted(missing, key=lambda ie: (frequency[layer, ie[1]], -ie[0]))
               [:churn.PLAN[len(missing)]]}
        fetched = [e for _, e in missing if e not in cpu]
        stat["fetch_bytes"] += len(fetched) * sizes[layer]
        stat["cpu_weight_bytes"] += len(cpu) * sizes[layer]
        stat["hits"] += 8 - len(missing)
        stat["scratch_uses"] += max(0, len(fetched) - 3)
        for i, expert in enumerate(ids):
            stat["mask_disagreements"] += ((expert not in cache[layer]) != bool(row["off"] >> i & 1))
        events += 8
        if events >= 32768:
            frequency = np.floor(frequency / 2)
            events = 0
        if layer in previous:
            for expert in previous[layer]:
                if seen[layer, expert] >= 2048:
                    seen[layer, expert] *= 0.5
                    transitions[layer, expert] *= 0.5
                seen[layer, expert] += 1
                transitions[layer, expert, ids] += 1
        previous[layer] = ids
        base = ((transitions[layer, ids] + 16 * frequency[layer] *
                 (8 / max(frequency[layer].sum(), 1))) / (seen[layer, ids, None] + 16)).mean(axis=0)
        recent[layer].append(ids)
        labels[layer].append(ids)
        if len(labels[layer]) == 17:
            stat["matured_anchors"] += 1
            anchor = labels[layer][0]
            future = sorted({e for xs in list(labels[layer])[1:] for e in xs})
            for expert in anchor:
                if window_seen[layer, expert] >= 2048:
                    window_seen[layer, expert] *= 0.5
                    window_counts[layer, expert] *= 0.5
                window_seen[layer, expert] += 1
                window_counts[layer, expert, future] += 1
        score = base
        mix = 0.0
        if mode.startswith("static"):
            score = model[layer, ids].mean(axis=0)
        elif mode == "adaptive-static16":
            # Prefix-only empirical shift gate; no evaluation future is read.
            observed = np.bincount([e for xs in recent[layer] for e in xs], minlength=288).astype(float)
            observed /= max(observed.sum(), 1)
            total_variation = float(np.abs(observed - train_rates[layer]).sum() / 2)
            mix = MAX_MIX * (1 - total_variation) ** 4 if len(recent[layer]) >= DRIFT_WINDOW else 0.0
            score = (1 - mix) * normalize(base) + mix * normalize(model[layer, ids].mean(axis=0))
        elif mode in ("online16", "adaptive-online16"):
            rates = frequency[layer] / max(frequency[layer].sum(), 1)
            prior_probability = 1 - (1 - rates) ** (8 * 16)
            online = ((window_counts[layer, ids] + PRIOR_ANCHORS * prior_probability) /
                      (window_seen[layer, ids, None] + PRIOR_ANCHORS)).mean(axis=0)
            if mode == "online16":
                score = online
            else:
                support = float(window_seen[layer, ids].mean())
                mix = MAX_MIX * support / (support + ONLINE_CONFIDENCE_ANCHORS)
                score = (1 - mix) * normalize(base) + mix * normalize(online)
        stat["mixture_sum"] += mix
        for expert in fetched[:3]:
            cache[layer].add(expert)
        while len(cache[layer]) > caps[layer] - 3:
            cache[layer].remove(min(cache[layer], key=lambda e: (score[e], frequency[layer, e],
                                                                last.get((layer, e), -1), e)))
    result = {}
    for phase, stat in stats.items():
        tokens = stat["routes"] / 42
        result[phase] = dict(stat)
        result[phase].update(tokens=tokens,
                             fetch_MB_per_token=stat["fetch_bytes"] / tokens / 1e6,
                             cpu_weight_MB_per_token=stat["cpu_weight_bytes"] / tokens / 1e6,
                             host_logical_MB_per_token=(stat["fetch_bytes"] + stat["cpu_weight_bytes"]) / tokens / 1e6,
                             mean_mixture_weight=stat["mixture_sum"] / stat["routes"],
                             mask_disagreement_percent=100 * stat["mask_disagreements"] / (stat["routes"] * 8))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path("G:/Models/GLM-5.3-Flash-3.0bit-Q4Kattn/pack-maya/native_experts.txt"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/evidence/maya-missing-residency/residency-research-replay.json")
    args = parser.parse_args()
    sizes = {int(f[0]): int(f[4]) for line in args.metadata.read_text().splitlines()
             if not line.startswith("#") and (f := line.split())}
    cap_file = ROOT / "docs/evidence/maya-demand/route-capacity-study.json"
    caps = {int(k): v for k, v in json.loads(cap_file.read_text())["baseline_slots"].items()}
    prior = collections.defaultdict(list)
    prior_file = ROOT / "data/expert-profile-glm.bin"
    for layer, expert in struct.iter_unpack("<HH", prior_file.read_bytes()[24:]):
        prior[layer].append(expert)
    static4, _, anchors4 = train_window(4)
    static16, train_rates, anchors16 = train_window(16)
    result = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "scope": "offline approximate fixed-route research; no inference/build or achieved TPS",
              "training": list(TRAIN), "held_out_domain": "current-csv",
              "training_domain_check": "current-web; not an independent holdout",
              "fixed_parameters": {"maximum_mixture": MAX_MIX, "prior_anchors": PRIOR_ANCHORS,
                                   "online_confidence_anchors": ONLINE_CONFIDENCE_ANCHORS,
                                   "drift_window": DRIFT_WINDOW, "drift_gate_power": 4},
              "training_anchors_per_layer": {"4": anchors4, "16": anchors16},
              "spare_payload_bytes_per_layer_sum": sum(sizes[layer] for layer in caps),
              "spare_payload_bytes_three_per_layer": sum(3 * sizes[layer] for layer in caps),
              "runs": {}, "source_sha256": {}}
    for name in EVAL:
        rows, phases, _ = load_run(ROOT / "docs/evidence/maya-churn" / name)
        modes = {"baseline-transition": (None, None), "static4": (static4, None),
                 "static16": (static16, None), "adaptive-static16": (static16, train_rates),
                 "online16": (None, None), "adaptive-online16": (None, None)}
        values = {mode: replay(rows, phases, sizes, caps, prior, mode, model, rates)
                  for mode, (model, rates) in modes.items()}
        # Reuse the original replay, rather than giving capacity scenarios a different policy.
        values["capacity-only"] = {str(extra): churn.replay(rows, sizes,
                                  {layer: value + extra for layer, value in caps.items()},
                                  prior, "base", phases) for extra in range(4)}
        result["runs"][name] = values
        # Confirm our retained-policy clone against the existing research replay.
        old = values["capacity-only"]["0"]
        for phase in old:
            assert abs(old[phase]["fetch_MB_per_token"] - values["baseline-transition"][phase]["fetch_MB_per_token"]) < 1e-9
    files = [Path(__file__), ROOT / "tools/audit_residency_churn.py",
             ROOT / "src/core/glm_reuse_policy.hpp", cap_file, prior_file, args.metadata]
    files += [ROOT / "docs/evidence/maya-demand" / name / "routes.0" for name in TRAIN]
    files += [ROOT / "docs/evidence/maya-churn" / name / "routes.0" for name in EVAL]
    files += [ROOT / "docs/evidence/maya-demand" / name / "tokens.txt" for name in TRAIN]
    files += [ROOT / "docs/evidence/maya-churn" / name / "tokens.txt" for name in EVAL]
    for path in files:
        result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
    for name, modes in result["runs"].items():
        for mode, phases in modes.items():
            if mode == "capacity-only":
                continue
            repeat = phases["r1-decode"]
            print(f"{name:12} {mode:20} fetch_MB={repeat['fetch_MB_per_token']:.3f} "
                  f"CPU_logical_MB={repeat['cpu_weight_MB_per_token']:.3f} "
                  f"mixture={repeat['mean_mixture_weight']:.5f}")


if __name__ == "__main__":
    main()
