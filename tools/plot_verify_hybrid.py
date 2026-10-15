"""Plot event-stage aggregates; deliberately not presented as a stream timeline."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/evidence/dflash-batched-verification'
data = json.loads((OUT / 'hybrid-phase-summary.json').read_text())
off, on = [r['phases'][0] for r in data]
o, n = off['stage_ms'], on['stage_ms']
parts = [
    [('Fetch', o['moe_fetch']), ('Expert compute', o['moe_experts'])],
    [('Fetch', n['moe_component_prefix']), ('Gate/up', n['moe_component_gu']),
     ('Down wait', n['moe_component_down_join']), ('Expert compute', n['moe_experts'])],
]
colors = {'Fetch': '#4477aa', 'Gate/up': '#eeaa33', 'Down wait': '#bb5566',
          'Expert compute': '#228833', 'Other stages': '#aab2bd'}
fig, ax = plt.subplots(figsize=(11, 4.8))
fig.subplots_adjust(left=.16, right=.98, top=.87, bottom=.30)
seen = set()
for y, (stages, phase) in enumerate(zip(parts, [off, on])):
    stages.append(('Other stages', phase['total_ms'] - sum(v for _, v in stages)))
    left = 0
    for label, value in stages:
        ax.barh(y, value, left=left, height=.48, color=colors[label],
                label=label if label not in seen else None)
        if value >= 3:
            ax.text(left + value/2, y, f'{value:.2f}', ha='center', va='center',
                    fontsize=10, color='white' if label in ('Fetch', 'Expert compute', 'Down wait') else '#111')
        seen.add(label)
        left += value
    ax.text(left + .4, y, f'{left:.2f} ms', va='center')
ax.set_yticks([0, 1], ['Component off', 'Component on'])
ax.invert_yaxis()
ax.set_xlim(0, 65)
ax.set_xlabel('Average GPU event-stage time per verification round (ms)')
ax.set_title('Hybrid verification: a small reduction from component streaming', loc='left', fontsize=13)
ax.spines[['top', 'right']].set_visible(False)
fig.legend(loc='lower left', bbox_to_anchor=(.16, .115), ncol=5, frameon=False, fontsize=9)
fig.text(.03, .025,
         '45 rounds each; two drafts; 128 output tokens. Aggregated costs, not a chronological trace.\n'
         'Down transfers overlap gate/up; down-wait is the exposed tail. Profiled results are not clean TPS qualification.',
         fontsize=9, color='#444')
for extension in ('png', 'svg'):
    fig.savefig(OUT / f'hybrid-phase-breakdown.{extension}', dpi=170)
