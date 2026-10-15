"""Summarize direct-vs-staged native expert replay, not serving TPS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
source = a.directory / "fixed.transport.csv"
rows = list(csv.DictReader(source.open()))
modes = ("stage_gpu", "direct_gpu", "stage_cpu_gpu", "direct_cpu_gpu")
result = {"csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "scope": "Actual-address expert replay; CPU runs as contention, GPU contribution checked exactly; not serving TPS.",
          "modes": {}, "comparisons": {}}
reference = None
for mode in modes:
    rs = [r for r in rows if r["mode"] == mode]
    if not rs:
        raise ValueError(f"Missing mode {mode}")
    signature = [(r["pass"], r["seq"], r["layer"], r["bytes"], r["cpu_experts"]) for r in rs]
    if reference is None:
        reference = signature
    if signature != reference or not all(int(r["verified"]) == 1 for r in rs):
        raise ValueError(f"Unmatched or unverified mode {mode}")
    per_pass = {}
    for pass_id in sorted({int(r["pass"]) for r in rs}):
        selected = [r for r in rs if int(r["pass"]) == pass_id]
        if len(selected) != 336:
            raise ValueError(f"Expected 336 layer records, got {len(selected)}")
        per_pass[str(pass_id)] = {k: sum(float(r[k]) for r in selected) / 8
                                 for k in ("bytes", "gpu_window_ms", "cpu_ms", "wall_ms")}
    result["modes"][mode] = {
        "layer_records": len(rs), "tokens_equivalent": len(rs) / 42,
        "per_pass": per_pass,
        "mean_per_token_equivalent": {k: sum(float(r[k]) for r in rs) / (len(rs) / 42)
                                      for k in ("bytes", "gpu_window_ms", "cpu_ms", "wall_ms")}}
result["quant_classes"] = {}
for key in sorted({(r["gu_type"], r["down_type"]) for r in rows}):
    name = "/".join(key)
    result["quant_classes"][name] = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode and (r["gu_type"], r["down_type"]) == key]
        result["quant_classes"][name][mode] = {
            "records": len(selected),
            "nominal_bytes": sum(int(r["bytes"]) for r in selected),
            "gpu_ms": sum(float(r["gpu_window_ms"]) for r in selected),
            "wall_ms": sum(float(r["wall_ms"]) for r in selected)}
for staged, direct in (("stage_gpu", "direct_gpu"), ("stage_cpu_gpu", "direct_cpu_gpu")):
    b = result["modes"][staged]["mean_per_token_equivalent"]
    c = result["modes"][direct]["mean_per_token_equivalent"]
    result["comparisons"][direct] = {k: {"difference_ms": c[k] - b[k], "ratio": c[k] / b[k]}
                                      for k in ("gpu_window_ms", "wall_ms")}
(a.directory / "summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
