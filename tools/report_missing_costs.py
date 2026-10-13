"""Report bounded experiments; preserve the distinction between screens and qualification."""
import json, re, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; P=R/'docs/evidence/maya-missing-costs'
def read(n,file='summary.json'):return json.loads((P/n/file).read_text())
def speed(n):
    s=read(n);return f"{s['0']['TPS']:.2f} / {s['1']['TPS']:.2f}"
def prefill(n):return re.findall(r'prefill repeat=\d+ tokens=8192 ms=([\d.]+)',(P/n/'engine.log').read_text())
lines=['# Maya missing-expert execution campaign','',
 '2026-10-13. Branch `research/maya-missing-expert-cost`, preserved baseline `86c6ea0`.', '',
 '## Decision', '',
 '**Keep the previously retained configuration. No new generally reliable speed gain was established.** '
 'Six complete mechanisms were screened, including targeted combinations. The best small CPU screen did not '
 'generalize to repeated 8K occupied context. All new switches remain off by default. Prototypes, tests and '
 'negative results are checkpointed; daily settings were not changed and nothing was published. 60 TPS was not reached.', '',
 '## Fresh controlled comparisons', '',
 'Same GSQ-RCO 3.0bit Q4-attention pack, c=1, fixed input histories, CPU plan011223344, '
 '22GiB expert pool. No speculative tokens, MTP, K4V, skipping or new quantization. Each scored continuation '
 'has128 tokens and two requests after tier warming. TPS is tokens divided by summed forward time. These are '
 'short local comparisons, not confidence intervals.', '',
 '| Workload | Preserved baseline first / repeat TPS | Coarse CPU candidate first / repeat TPS | Repeat change |',
 '|---|---:|---:|---:|']
for label,b,c in [('Short coding','baseline-confirm','coarse-confirm'),('8192 occupied code tokens','baseline-8k','coarse-8k')]:
    lines.append(f"| {label} | {speed(b)} | {speed(c)} | {(read(c)['1']['TPS']/read(b)['1']['TPS']-1)*100:+.2f}% |")
lines+=['', 'Baseline repeated short TPS varied45.71→46.29 between the initial and confirmation runs. '
 'The candidate confirmation gain is only0.84%, and the longer-context repeat regresses0.86%. '
 'Do not promote the46.88 initial screen to a sustained performance claim.', '',
 '| GU streaming alone | Baseline repeat TPS | GU repeat TPS | Change |','|---|---:|---:|---:|',
 f'| Short | {read("baseline-confirm")["1"]["TPS"]:.2f} | {read("gu-confirm")["1"]["TPS"]:.2f} | +0.95% |',
 f'| 8K occupied | {read("baseline-8k")["1"]["TPS"]:.2f} | {read("gu-8k")["1"]["TPS"]:.2f} | −0.90% |', '',
 'The isolated streaming candidate was also qualified rather than inferred from its failed combination. '
 'It likewise fails to establish a general gain; both full-logit comparisons are exact.', '',
 '## Hypothesis and results matrix', '',
 '| Mechanism | Plausible saving before test | First / repeat TPS screen | Verified engagement / result | Decision |',
 '|---|---|---:|---|---|']
experiments=[
 ('Coarse single-expert CPU executor','At most1.265ms exposed single-expert wait','cpu-single','5605 real coarse batches; full logits exact; short confirmation+0.84%,8K−0.86%','Keep opt-in only'),
 ('Paired gate/up row streaming','Mostly relocates existing overlap; demanded down tail only~0.26ms','gu-bands','84 paired prefix copies and84 GU consumers/token; exact; combined profile flat','Off'),
 ('Four independent mapped-copy loads','Conditional~1.5ms if concurrent replay bandwidth transfers to inference','fetch-pipe4','Template4 launch proof in engagement capture; exact bytes/logits; no useful standalone gain','Off'),
 ('Completed16-route causal reuse','Replay warm3–5% fewer uploads; first-request regression predicted','reuse16','12600 matured anchors; repeat383→368MB, first464→484MB; hybrid arithmetic placement changes','Reject for retention'),
 ('Exact paired Q2 AVX512 rows','Bounded by CPU wait, does not remove weight bytes','q2-paired','Standalone rows/hidden bytes exact; full-model39.65M logits exact; diagnostic row counters','Off'),
 ('Delay full CPU admission until down copy completes','Recover contention, bounded by added CPU tail','release-down','Same generation published after all down bytes on copy stream; phase callbacks engage; exact','Off'),
]
for title,bound,n,proof,decision in experiments:lines.append(f'| {title} | {bound} | {speed(n)} | {proof} | {decision} |')
lines+=['', '**The residency run had route logging enabled**, so its TPS is diagnostic. Its byte/correctness results, '
 'not a sub-percent timing difference, determine rejection. Other headline screens are unprofiled.', '',
 '## Dependencies and combinations', '',
 f'- GU bands + coarse CPU: {speed("combined")} TPS, exact. Slower than either isolated screen.',
 f'- Coarse CPU + paired Q2 + pipelined copy: {speed("cpu-copy-combination")} TPS, exact. No additive gain.',
 '- The existing down transfers already hide almost all gate/up computation on routes with actual demand '
 '(only0.000152ms/token exposed in the source trace). Exposed all-resident work has no same-layer demand to hide. '
 'Moving overlap earlier is not automatically additional saved time.',
 '- The online16 cache policy learns only from fully observed past windows, not an oracle. It reduces warm '
 'CPU and upload traffic but adapts poorly initially. Static/mixed predictors and spare reclamation were '
 'also bounded by offline replay; these are feasibility evidence, not live performance claims.', '',
 '## Numerical and prefill results', '',
 'All scheduling/copy/Q2 experiments compare exactly across39,649,280 logits in each full short test. '
 'Short and8K coarse-candidate qualification also match all logits and non-timing tier counters exactly. '
 'A combined32-token diagnostic verifies1344 late phase publications after all demand copies, '
 '2688 launches each of paired-prefix4/down-component4/GU/down, and2,478,080 exact first-request prefix logits. '
 'A separate counter run records9,263,104 Q2 GU rows and18,495,360 paired down rows (includes setup). '
 'Its second request has a different cache history from the128-token reference; no second-request parity claim is made. '
 'Q2 standalone tests include96 dot pairs, actual expert dimensions, hidden quantized bytes, '
 'clamped/unclamped SwiGLU, misalignment, odd tails and unsupported-format guards.', '',
 'Online16 hybrid placement differs: max absolute logit difference3.4219, mean0.08974, greedy agreement99.21875%. '
 'It is not described as hybrid parity. Controlled GPU-only isolation matches all9,912,320 logits exactly. '
 'This separates backend movement from lost/corrupted ownership; no numerical threshold was weakened.', '',
 '| 8192-token prefill | Preserved baseline ms | Coarse CPU candidate ms |', '|---|---:|---:|']
for i,(b,c) in enumerate(zip(prefill('baseline-8k'),prefill('coarse-8k'))):lines.append(f'| Request{i+1} | {float(b):.3f} | {float(c):.3f} |')
lines+=['', 'Prefill continues through the original batched path. No new candidate is promoted, so there is '
 'no fresh lifecycle certification claim for a retained optimization. Normal teardown/repeated requests '
 'passed in the experiments. The previous retained candidate keeps its existing STOP/recovery/cold-tier qualification.', '',
 '## Diagnostic timeline: why more overlap did not win', '',
 'The following bars partition each token into disjoint observed intervals. They sum to the whole diagnostic '
 'loop. GPU compute includes weight reads/dequantization; CPU runs concurrently. These are separate Nsight '
 'runs and must not be substituted for unprofiled TPS.', '',
 '| Stage | Previously retained ms/token | GU + coarse CPU prototype ms/token |','|---|---:|---:|']
baseline=json.loads((R/'docs/evidence/maya-three-costs/retained-trace/partition-means.json').read_text())['1']
trial=read('combined-trace','partition-means.json')['1']
labels=[k for k in baseline if k!='wall_ms']; colors=['#247da4','#6baf76','#e59639','#cc5a65','#9985b4','#cbd0d7']
for k in labels:lines.append(f'| {k} | {baseline[k]:.3f} | {trial[k]:.3f} |')
lines.append(f'| **Total** | **{baseline["wall_ms"]:.3f}** | **{trial["wall_ms"]:.3f}** |')
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
for y,(name,row) in enumerate([('Previously retained',baseline),('Streaming + coarse CPU prototype',trial)]):
    left=0
    for k,color in zip(labels,colors):
        ax.barh(y,row[k],left=left,color=color,label=k if y==0 else None)
        if row[k]>.65:ax.text(left+row[k]/2,y,f'{row[k]:.2f}',ha='center',va='center',color='white' if color=='#247da4' else 'black')
        left+=row[k]
    ax.text(left+.18,y,f'{left:.2f} ms',va='center')
ax.set_yticks([0,1],['Previously retained','Streaming + coarse CPU prototype']);ax.invert_yaxis()
ax.set_xlim(0,26);ax.set_xlabel('Mean ms/token — disjoint diagnostic intervals')
ax.set_title('More overlap, effectively unchanged token loop\n128 repeated fixed coding tokens per trace; this prototype is not promoted')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False)
fig.savefig(P/'comparison.png',dpi=160);plt.close(fig)
lines+=['','![Diagnostic comparison](comparison.png)', '',
 'Exposed fetch drops~0.71ms, but GPU arithmetic and CPU waits grow. The prototype is not a demonstrated '
 'profiled speedup. Individual timelines and layer stages are in `combined-trace/`.', '',
 '## Traffic and memory', '',
 '| Repeated workload | H2D MB/token, baseline / coarse | D2H MB/token, baseline / coarse |','|---|---:|---:|']
for label,b,c in [('Short','baseline-confirm','coarse-confirm'),('8K','baseline-8k','coarse-8k')]:
    B,C=read(b)['1'],read(c)['1']
    lines.append(f'| {label} | {B["fetch_bytes"]/1e6:.3f} / {C["fetch_bytes"]/1e6:.3f} | {B["demotion_bytes"]/1e6:.3f} / {C["demotion_bytes"]/1e6:.3f} |')
lines+=['', 'All main runs retain2693 GPU slots (21.999GiB) and11819 RAM slots (97.822GiB). '
 'Short CPU logical expert payload remains564.156MB/token. This is payload accounting, not a DRAM-controller '
 'measurement. No extra weight cache or model quantization was introduced.', '',
 '| Confirmation run | Minimum available system RAM GiB | Whole-device peak VRAM MiB |','|---|---:|---:|']
for n in ['baseline-confirm','coarse-confirm','baseline-8k','coarse-8k']:
    mem=read(n,'memory.json');gpu=list(csv.DictReader((P/n/'gpu.csv').open()))
    peak=max(float(next(v for k,v in r.items() if k.strip().startswith('memory.used')).split()[0]) for r in gpu)
    lines.append(f'| {n} | {min(x["available_physical"] for x in mem)/2**30:.2f} | {peak:.0f} |')
lines+=['', '## Remaining bottleneck and next substantive direction', '',
 'The retained profile still has~9.1ms exposed upload,3.16ms CPU-answer wait and9.4ms GPU work '
 '(partly overlapping). Same-layer scheduling has little untouched compute to hide. A substantial gain '
 'requires useful earlier knowledge/residency or a demonstrably faster joint CPU/upload path; these tests '
 'do not establish one. Activation-conditioned value prediction needs causal feature collection and '
 'task-disjoint feasibility evidence before implementation as a production policy. Generic frequency '
 'admission or adding more overlap launches is not supported by the results.', '',
 '## Reproduction and checkpoints', '',
 '`tools/run_missing_costs.py NAME --baseline` uses the preserved executable. Without `--baseline`, '
 'the new binary runs the previous retained flags; each new experiment requires an explicit switch. '
 '`tools/summarize_three_costs.py NAME --root docs/evidence/maya-missing-costs --reference REFERENCE` '
 'produces counters and full-logit comparisons. `tools/build-missing-tests.bat` builds only the two relevant CPU tests.', '',
 'See `docs/MAYA-MISSING-COSTS-CAMPAIGN.md` and the three missing-expert research notes for mechanisms, '
 'safety reviews and negative bounds. Small checkpoints preserve rollback; compact evidence is committed. '
 'Large logits, Nsight and SQLite artifacts remain local/ignored. The unrelated `maya-bench.txt` is untouched.']
body='\n'.join(lines)+'\n'
body=re.sub(r'\b(plan|has|varied|only|the|and|warm|first|repeat|most|combined|Completed|across|verifies|records|include|all|difference|mean|agreement|retain|remains|Template|Request|model)(?=\d)',r'\1 ',body)
body=re.sub(r'(?<=\d)(?=(?:ms|MB|GiB|TPS)\b)',' ',body)
(P/'RESULTS.md').write_text(body,encoding='utf-8')
print(P/'RESULTS.md')
