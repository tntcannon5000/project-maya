"""Render the bounded drafter screen, including its workload/capacity caveats."""
import json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/dflash-quant-width'

def main():
    quants=['Q4_K_M','Q5_K_M','Q8_0']
    data={q:json.loads((OUT/q/'requests.json').read_text()) for q in quants}
    cells={q:{r['width']:r for r in data[q] if r['kind']=='matrix'} for q in quants}
    assert all(len(cells[q])==7 for q in quants)
    lines=['# DFlash2 quantization and draft-width screen — 2026-10-14','',
      '## Scope and method','',
      '- Maya v1.1.7 retained fork, branch `research/dflash-quant-width`; source checkpoint `2940ce7`.',
      '- Same GSQ-RCO 3.0bit Q4-attention target pack, one RTX 5090, c=1, 131072-token allocated context. The prompt is short; this is not 128K occupied-context performance.',
      '- One FastAPI dashboard coding prompt, greedy generation, 192 output tokens per request. One warmup and one no-draft control per drafter load, then seven widths in order 4,1,7,2,6,3,5.',
      '- `spec=N+1` includes the anchor; confidence cutoff and cumulative cutoff disabled, minimum drafts N. Per-round records confirm each requested width engaged.',
      '- Main-model MTP, suffix drafting and KV streaming are off. These are DFlash2 tests, not separate MTP or K4V tests.',
      '- Lightweight upstream per-round statistics and timing enabled; no Nsight profiler. TPS is DONE output tokens / decode wall time, excluding prompt processing. No independent profiler-free confirmation was added to this small screen.',
      '- Same retained settings and requested 22 GiB expert pool. Automatic VRAM sizing means larger drafters reduce actual expert slots: a realistic deployment comparison, not an isolated precision-only ablation.',
      '- Single sample per cell, sequential repeated requests, no confidence intervals. Cache history/order, foreground activity, and hybrid CPU/GPU numerical differences can affect results. Output histories were recorded and do differ; acceptance is not measured against identical forced routes.',
      '', '## Results','', 'Cells: decode TPS / accepted draft percentage. Acceptance is accepted prefix tokens divided by verified draft tokens.','',
      '| Draft tokens | Q4_K_M | Q5_K_M | Q8_0 |','|---:|---:|---:|---:|']
    for w in range(1,8):
        lines.append('| '+str(w)+' | '+' | '.join(f"{cells[q][w]['tps']:.2f} / {cells[q][w]['acceptance']*100:.1f}%" for q in quants)+' |')
    lines += ['', '## Loaded controls and memory','',
      '| Drafter | File GB | Drafter CUDA MB | Expert pool GiB / slots | No-draft TPS | Best draft width / TPS |','|---|---:|---:|---|---:|---|']
    meta=json.loads((OUT/'drafter-metadata.json').read_text())
    for q in quants:
        log=(OUT/q/'engine.log').read_text()
        pool=re.search(r'expert pool ([\d.]+) GB, .*?\((\d+) total\)',log)
        mb=re.search(r'glm dflash: 5 layers.*?; (\d+) MB on CUDA0',log)
        control=next(r for r in data[q] if r['kind']=='control')
        best=max(cells[q].values(),key=lambda r:r['tps'])
        size=next(m['bytes'] for m in meta if f'-{q}.gguf' in m['file'])/1e9
        lines.append(f"| {q} | {size:.3f} | {mb[1]} | {pool[1]} / {pool[2]} | {control['tps']:.2f} | {best['width']} / {best['tps']:.2f} |")
    lines += ['', '## Verification costs','',
      'Mean per-round timing from upstream counters. Draft and verification costs below are diagnostic wall times, not separate CUDA kernel timings.','',
      '| Quant | Width | Rounds | Draft ms/round | Verify ms/round | Mean accepted/round | First difference from loaded no-draft control (0-based) |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for q in quants:
        control=next(r for r in data[q] if r['kind']=='control')['output_ids']
        for w,r in sorted(cells[q].items()):
            n=len(r['rounds'])
            diff=next((i for i,(a,b) in enumerate(zip(control,r['output_ids'])) if a!=b),None)
            lines.append(f"| {q} | {w} | {n} | {r['draft_ms']/n:.2f} | {r['verify_ms']/n:.2f} | {r['accepted']/n:.2f} | {diff if diff is not None else 'none'} |")
    lines += ['', '## Implementation and validation','',
      '- Downloaded all three drafters from the pinned HF revision in `download-manifest.json`; SHA256 checked against LFS hashes. Tensor headers confirm Q4_K/Q6_K, Q5_K/Q6_K and Q8_0, respectively.',
      '- Added Q5_K selector-table reads using the existing GGML block layout and affine dequantization formula. Main matrix-vector kernels already supported Q5_K. Q4/Q8 arithmetic paths are unchanged.',
      '- Q5 existing GPU-versus-CPU drafter reference: PASS, minimum hidden cosine 0.992372, 15/112 candidate differences, 2/7 draft-pick differences under the existing thresholds. This is approximate agreement, not bit-exact parity; no thresholds were weakened.',
      '- All 21 matrix cells completed; actual draft widths checked; all engines exited normally. No persistent daily launcher settings changed. Pre-change executable retained as `build/strata-before-q5-selector.exe`.',
      '', '## Decision','',
      'Use the measured table rather than assuming acceptance increases monotonically with precision. Wider batches are dominated by target verification cost on this hybrid single-GPU setup. This screen does not establish a reliable drafting speedup over its loaded no-draft controls. Leave the daily preset unchanged. These free-generation samples are not a new numerical parity qualification.',
      '',
      'All three engine logs report a verification scratch-slot overflow during the first width-4 test. In `src/core/glm_window.cu`, `window()` detects the sticky overflow flag after the batched pass, restores recurrent state, and reruns that window token by token. The warning prints only once per process, so these logs establish engagement but do not count all retries or attribute their total cost. The per-token flag resets for each subsequent window. This is a concrete engine bottleneck, not evidence that larger drafters or DFlash2 are intrinsically unhelpful. Fixing and measuring scratch capacity is a separate follow-up; this matrix deliberately retains the current scheduling implementation.',
      '', 'Raw manifests, token outputs, round records, engine logs and GPU telemetry are alongside this report. RAM-tier sizing and actual expert slots are recorded in each engine log.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[18:40]))

if __name__=='__main__':main()
