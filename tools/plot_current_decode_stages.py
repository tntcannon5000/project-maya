"""Render saved first-request profiles; does not run inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1] / 'docs/evidence'
baseline = json.loads((root / 'maya-demand/trace-graphs/trace-repeats.json').read_text())['0']
candidate = json.loads((root / 'maya-residency-overlap/trace-final/trace-summary.json').read_text())
keys = ['wall_ms', 'fetch_ms', 'arithmetic_ms', 'cpu_wait_ms', 'd2h_ms', 'no_gpu_ms', 'fetch_arithmetic_overlap_ms']
labels = ['Whole token loop', 'Expert fetch: RAM → VRAM', 'GPU arithmetic', 'CPU-answer wait kernels', 'Expert demotion: VRAM → RAM', 'GPU inactive gaps', 'Fetch / arithmetic overlap*']
fig, ax = plt.subplots(figsize=(12, 6.8))
fig.patch.set_facecolor('#f8fafc')
ax.set_facecolor('#f8fafc')
y = np.arange(len(keys))
for offset, data, color, label in [(-.18, baseline, '#a6b4c5', 'Previous graph baseline'), (.18, candidate, '#187e9d', 'Latest retained candidate')]:
    values = [data[k] for k in keys]
    ax.barh(y + offset, values, height=.32, color=color, label=label)
    for yy, value in zip(y + offset, values):
        ax.text(value + .18, yy, f'{value:.2f}', va='center', fontsize=10)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 30)
ax.set_xlabel('Mean milliseconds per token — overlapping intervals, NOT an additive breakdown')
ax.set_title('Maya decode: where token time goes', loc='left', fontsize=19, weight='bold', pad=37)
ax.text(0, 1.045, 'First-request diagnostic averages • 128 scored tokens each • c=1 • GSQ-RCO 3.0bit / Q4 attention', transform=ax.transAxes, fontsize=10)
ax.legend(loc='lower right', frameon=False)
ax.xaxis.grid(True, alpha=.18)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(.03, .035, '*Overlap is time shared by fetch and arithmetic, not an extra cost. CPU-answer wait is partly overlapped too.\nLatest warm-repeat stage trace is not available. Warm unprofiled throughput: 43.30 → 45.20 TPS; separate measurements.', fontsize=10, color='#334155')
fig.tight_layout(rect=(0, .10, 1, 1))
out = root / 'maya-residency-overlap/report/current-stage-breakdown.png'
fig.savefig(out, dpi=160)
print(out)
