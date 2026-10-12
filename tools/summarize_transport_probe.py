"""Summarize one bounded actual-address transport replay, never infer live TPS."""
import argparse,csv,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();root=a.directory
rows=list(csv.DictReader((root/'fixed.transport.csv').open()))
assert len(rows)==5*2*336,len(rows)
summary={};reference=None
for mode in ['pull','pull_cpu','pull_gpu','pull_cpu_gpu','dma']:
    rs=[r for r in rows if r['mode']==mode]
    signature=[(r['pass'],r['seq'],r['layer'],r['bytes'],r['cpu_experts']) for r in rs]
    if reference is None:reference=signature
    assert reference==signature
    tokens=len(rs)/42
    result={k:sum(float(r[k]) for r in rs)/tokens for k in ['bytes','fetch_ms','gpu_window_ms','cpu_ms','wall_ms','cpu_experts']}
    result['effective_GB_s']=result['bytes']/result['fetch_ms']/1e6
    result['tokens_equivalent']=tokens
    result['pass_fetch_ms']=[sum(float(r['fetch_ms']) for r in rs if int(r['pass'])==i)/8 for i in (0,1)]
    summary[mode]=result
fixed=list(csv.DictReader((root/'fixed.csv').open()))[-8:]
assert abs(statistics.mean(float(r['fetch_bytes']) for r in fixed)-summary['pull']['bytes'])<1
summary['notes']='GPU timestamps include tiny stamp/dispatch gaps. Replay is eight last-token address sets, two passes; not live token latency. CPU and GPU work match selected expert shapes; shared/attention excluded.'
(root/'replay-summary.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
names=['pull','pull_cpu','pull_gpu','pull_cpu_gpu','dma']
vals=[summary[n]['fetch_ms'] for n in names]
ax.barh(range(5),vals,color=['#377d9e','#c78932','#589b70','#ac615c','#8266a9'])
ax.set_yticks(range(5),['Mapped copies only','Mapped + CPU experts','Mapped + GPU experts','Mapped + CPU and GPU','DMA copies only']);ax.invert_yaxis()
for i,v in enumerate(vals):ax.text(v+.08,i,f'{v:.2f} ms  |  {summary[names[i]]["effective_GB_s"]:.1f} GB/s',va='center')
ax.set_xlim(0,max(vals)*1.35);ax.set_xlabel('Copy time summed per token-equivalent, same actual source addresses')
ax.set_title(f'Actual-address replay: {summary["pull"]["bytes"]/1e6:.1f} MB/token-equivalent\n8 route histories × 2 passes; not end-to-end throughput')
fig.savefig(root/'replay-comparison.png',dpi=150)
print(json.dumps(summary,indent=2))
