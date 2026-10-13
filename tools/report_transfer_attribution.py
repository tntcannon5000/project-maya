"""Render the live CPU-contention ablation from saved, matched diagnostic traces."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'docs/evidence'
out=root/'maya-transfer-attribution'
a=json.loads((root/'maya-deep-profile/warm-detail/trace-repeats.json').read_text())['1']
b=json.loads((out/'cpu-first/trace-repeats.json').read_text())['1']
keys=['wall_ms','fetch_ms','cpu_wait_ms','arithmetic_ms','fetch_arithmetic_overlap_ms']
labels=['Whole token loop','Expert fetch','CPU-answer waits','GPU arithmetic','Fetch / arithmetic overlap']
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
for delta,vals,color,label in [(-.18,a,'#237e9e','Normal concurrent execution'),(.18,b,'#ce8840','CPU first — diagnostic only')]:
    ax.barh([i+delta for i in range(len(keys))],[vals[k] for k in keys],height=.33,color=color,label=label)
    for i,k in enumerate(keys):ax.text(vals[k]+.15,i+delta,f'{vals[k]:.2f}',va='center')
ax.set_yticks(range(len(keys)),labels);ax.invert_yaxis();ax.set_xlim(0,32)
ax.set_xlabel('ms/token — overlapping intervals, NOT additive')
ax.set_title('Removing CPU contention speeds copies but slows tokens\nSame 383 MB/token; warm 128-token diagnostic means')
ax.legend(loc='lower right');fig.savefig(out/'cpu-contention.png',dpi=160)
print(out/'cpu-contention.png')
