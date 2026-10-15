"""Observed-round timing budgets, not simulated decoding or predicted TPS.

Counterfactuals retain recorded histories, acceptance and non-target time. They
assume savings subtract from wall time and therefore are optimistic ceilings.
"""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(path, target):
    requests = json.loads(path.read_text())
    request = next(r for r in requests if r['kind'] == 'confidence')
    rounds = request['rounds']
    wall = request['decode_ms']
    draft = sum(r[2] for r in rounds)
    verify = sum(r[3] for r in rounds)
    zero_draft = sum(r[2] for r in rounds if r[0] == 0)
    nonzero_verify = sum(r[3] for r in rounds if r[0] > 0)
    need = max(0, wall - request['tokens'] * 1000 / target)
    if min(wall, verify, nonzero_verify) <= 0 or wall - draft - verify < -1:
        raise ValueError('invalid or inconsistent timing totals')
    groups = []
    for width in sorted({int(r[0]) for r in rounds}):
        selected = [r for r in rounds if r[0] == width]
        groups.append(dict(drafts=width, rounds=len(selected),
            accepted=sum(r[1] for r in selected), draft_ms=sum(r[2] for r in selected),
            verify_ms=sum(r[3] for r in selected),
            mean_verify_ms=sum(r[3] for r in selected)/len(selected)))
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        measured_tps=request['tps'], tokens=request['tokens'], wall_ms=wall,
        target_tps=target, required_saving_ms=need,
        draft_ms=draft, verify_ms=verify, other_ms=wall-draft-verify,
        zero_draft_overhead_ms=zero_draft,
        ceiling_if_zero_draft_attempts_free=request['tokens']*1000/(wall-zero_draft),
        ceiling_if_all_drafting_free=request['tokens']*1000/(wall-draft),
        required_fraction_all_verification_removed=need/verify,
        required_fraction_nonzero_verification_removed=need/nonzero_verify,
        widths=groups)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('requests',type=Path,nargs='+')
    p.add_argument('--target',type=float,default=60)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.target<=0: p.error('target must be positive')
    result={'caveat':'Optimistic subtraction budgets on observed histories, not measured counterfactual runs; no routing, acceptance, cache or graph-warmup invariance claim.',
            'tasks':[analyze(path,a.target) for path in a.requests]}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
