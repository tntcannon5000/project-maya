"""Verify route accounting and render the bounded churn audit."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/maya-churn'
d=json.loads((OUT/'audit.json').read_text())
checks=[]
for name, res in d.items():
    rows=list(csv.DictReader((OUT/name/'fixed.csv').open()))
    for repeat in (0,1):
        selected=[r for r in rows if int(r['repeat'])==repeat]
        s=res['observed'][f'r{repeat}-decode']
        for counter in ('fetch_bytes','hits','promotions'):
            measured=sum(int(r[counter]) for r in selected)
            key='admissions' if counter=='promotions' else counter
            assert measured==s[key],(name,repeat,counter,measured,s[key])
        checks.append(f'{name}, repeat {repeat}: route fetch bytes, hits and admissions match engine counters exactly.')
    for phase,s in res['observed'].items():
        assert s['admissions']==s.get('admissions_with_next_use',0)+s.get('admissions_right_censored',0)
        assert s['admissions_with_next_use']==s['admissions_surviving_next_use']+s['admissions_lost_before_next_use']

refs={'current-csv':'maya-transfer-attribution/replay-stamps',
      'current-web':'maya-residency-overlap/web-candidate'}
for name, ref in refs.items():
    path=ROOT/'docs/evidence'/ref/'fixed.logits.f32'
    if path.exists():
        a=np.fromfile(OUT/name/'fixed.logits.f32',dtype=np.float32)
        b=np.fromfile(path,dtype=np.float32)
        assert np.array_equal(a,b),(name,'logits differ')
        checks.append(f'{name}: all {a.size:,} logits exactly equal retained reference {ref}; maximum difference 0.')

fig,ax=plt.subplots(figsize=(10,4.5))
left=np.zeros(len(d)); names=list(d)
labels=['Residency loss; return in 1 token','Return in 2–4 tokens','Return in 5–16 tokens','Other fetch traffic']
colors=['#c44e52','#dd8452','#e5c66a','#4c72b0']
for i,(label,color) in enumerate(zip(labels,colors)):
    vals=[]
    for name in names:
        s=d[name]['observed']['r1-decode']
        cumulative=[0]+[s[f'loss_fetch_MB_per_token_within_{n}'] for n in (1,4,16)]+[s['fetch_MB_per_token']]
        vals.append(cumulative[i+1]-cumulative[i])
    ax.barh(names,vals,left=left,label=label,color=color)
    left+=vals
ax.set_xlabel('RAM → VRAM MB/token (decimal); repeated fixed-input decode')
ax.set_title('Very short-return residency losses explain little of current traffic')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False)
fig.tight_layout();fig.savefig(OUT/'churn.png',dpi=160);plt.close(fig)

lines=['# Residency churn audit — 2026-10-13','',
'## Verdict','',
'The current cache does not exhibit large immediate eviction/refetch thrashing on these two short fixed-input tasks. It serves about 70% of expert selections from VRAM. Only about 2.1% of repeated GPU fetch bytes concern an expert last observed resident within four tokens. A simple stricter admission rule made replay worse on both tasks. Retain the current policy.','',
'## Direct measurements','',
'Each process uses the retained transition/overlap configuration, 22 GiB GPU pool, 101 GiB RAM budget, 24 GiB shadow allowance, no MTP, and two sequential identical requests. CSV has 128 decode tokens/request; web uses its frozen continuation. The short prompts also run through route logging. Rows below exclude prompts. These are diagnostic runs, not new unprofiled TPS claims.','',
'| Task / request | Fetch MB/token | Resident hit % | Loss → return ≤1 token, MB/t | ≤4 tokens, MB/t | ≤16 tokens, MB/t |',
'|---|---:|---:|---:|---:|---:|']
for name,res in d.items():
    for repeat in (0,1):
        s=res['observed'][f'r{repeat}-decode']
        lines.append(f"| {name} / {repeat+1} | {s['fetch_MB_per_token']:.2f} | {s['hit_percent']:.2f} | "+' | '.join(f"{s[f'loss_fetch_MB_per_token_within_{n}']:.2f}" for n in (1,4,16))+' |')
lines+=['','The return columns are cumulative. A loss means the expert was a logged hit or confirmed admission at its preceding use, and is now off-GPU. The interval is between demands, not eviction-to-demand latency. CPU-served returns are counted as losses in raw JSON but do not contribute to these GPU-byte columns. Losing residency does not prove a bad eviction: retaining that expert may displace a more useful one.','',
'![Churn breakdown](churn.png)','',
'## Causal admission screen','',
'Before looking at results, fixed a single rule: only admit a fetched expert if its aged frequency is at least 1.5× that of the current transition-selected resident victim. Rejected fetches still compute through scratch; no expert computation is skipped. No future demand is consulted; no rule fitting between CSV and the held-out web task. This is an offline screen, not a production implementation.','',
'Replay models actual GPU-only admission (at most three spare slots per layer), CPU selection, count aging and transition scoring. It remains approximate: boundary ordering, asynchronous slot states, exact device counters and tie ordering are not reproduced. In particular, admission decisions are sequential here, whereas production fills spares then evicts at the boundary.','',
'| Repeated task | Actual MB/t | Replay current MB/t | Replay strict admission MB/t | Change | Current replay residency-mask disagreement |',
'|---|---:|---:|---:|---:|---:|']
for name,res in d.items():
    s=res['observed']['r1-decode'];b=res['replay_transition']['r1-decode'];m=res['replay_admission_margin']['r1-decode']
    lines.append(f"| {name} | {s['fetch_MB_per_token']:.2f} | {b['fetch_MB_per_token']:.2f} | {m['fetch_MB_per_token']:.2f} | {100*(m['fetch_MB_per_token']/b['fetch_MB_per_token']-1):+.2f}% | {b['mask_disagreement_percent']:.3f}% |")
lines+=['','Baseline replay fidelity is strong here; it does not validate every counterfactual policy. Reject this simple admission rule as a live implementation candidate for now. It is not evidence against every admission policy.','',
'## Admission survival','',
'| Repeated task | Admissions | With observed later demand | Still resident at that demand | Lost before demand | No later demand in trace |',
'|---|---:|---:|---:|---:|---:|']
for name,res in d.items():
    s=res['observed']['r1-decode']
    lines.append('| '+name+' | '+' | '.join(str(s[k]) for k in ('admissions','admissions_with_next_use','admissions_surviving_next_use','admissions_lost_before_next_use','admissions_right_censored'))+' |')
lines+=['','Counts are attributed to the admission phase. Absence of later demand is right-censored, not proof that an admission was useless. More than half of observed returning admissions lose residency, but very short-return losses are rare. This points to predicting medium/longer reuse, not an obvious failure to protect the next token.','',
'## What this does and does not establish','',
'- GPU pool has 2,693 slots across 42 × 288 experts. Reserving three spares/layer leaves roughly 21.2% of expert identities resident; mixed quant sizes mean this is not a weight-byte fraction. About 70% selection hits are meaningful evidence that residency concentrates useful experts.','- The historical 411→228 MB/token oracle used different, simplified all-miss admission. It is neither a guaranteed achievable saving nor a rigorous current-policy capacity lower bound.','- We cannot identify all unavoidable misses from this audit. Four-token return traffic is a diagnostic category, not a bound on all possible savings. Even the 54–60 MB/token returning within 16 tokens cannot all be eliminated for free.','- This covers two short fixed-input tasks, first and repeated requests; it does not establish behavior for long contexts, arbitrary task switches or free generation. No settings were changed.','- Next residency work should require an evidence-backed predictor of reuse over tens of tokens, evaluated on a separate task. A blanket frequency admission margin is not justified. Transfer/CPU memory contention remains a separate proven bottleneck.','',
'## Validation and artifacts','']+['- '+x for x in checks]+['',
'Opt-in route logs now append a fourth mask (actual promotions) before `|`; the first three masks and near-ID parsing remain compatible. Engine compute/scheduling is unchanged. Raw routes, manifests, token fixtures, counters and audit.json are retained beside this report. Large logits remain local and untracked. Build emitted existing CUDA/C++ warnings; no new compiler failure.','',
'Reproduce: `.venv/Scripts/python.exe tools/audit_residency_churn.py docs/evidence/maya-churn/current-csv docs/evidence/maya-churn/current-web`, then `tools/report_residency_churn.py`.']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
print('\n'.join(checks))
