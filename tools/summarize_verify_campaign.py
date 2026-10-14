"""Summarize saved window experiments; never launch inference or infer speedups."""
import argparse
import json
import re
from pathlib import Path


def summarize(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    requests = json.loads((directory / 'requests.json').read_text())
    text = (directory / 'engine.log').read_text(errors='replace')
    phases = []
    for match in re.finditer(r'glm window prof CUDA(\d+) \((\d+) windows, layers \[(\d+), (\d+)\)\): ([\d.]+) ms a window -([^\r\n]+)', text):
        phases.append(dict(device=int(match[1]), windows=int(match[2]),
                           layers=[int(match[3]), int(match[4])], total_ms=float(match[5]),
                           stage_ms={k: float(v) for k, v in re.findall(r'(\w+) ([\d.]+)', match[6])}))
    env = manifest['environment']
    result = dict(directory=str(directory), binary_sha256=manifest['binary_sha256'],
                  diagnostic=env.get('MAYA_WIN_GROUP_DIAG', '0') != '0',
                  phase_profile=env.get('STRATA_GLM_WIN_PROF') is not None,
                  occupied_prompt_tokens=len(manifest['prompt_ids']),
                  context_capacity=manifest['context_capacity'], environment=env,
                  requests=[], phases=phases,
                  group_lines=[s for s in text.splitlines() if s.startswith('maya verify groups')])
    for row in requests:
        rounds = len(row['rounds'])
        result['requests'].append({k: row[k] for k in ('kind', 'width', 'tokens', 'tps', 'prompt_ms', 'decode_ms', 'offered', 'accepted', 'acceptance')})
        result['requests'][-1].update(rounds=rounds,
                                      verify_ms_per_round=row['verify_ms'] / rounds if rounds else None,
                                      draft_ms_per_round=row['draft_ms'] / rounds if rounds else None)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(p) for p in args.directories]
    args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    for result in results:
        label = 'PERTURBING GROUP DIAGNOSTIC' if result['diagnostic'] else ('phase-profiled' if result['phase_profile'] else 'round-counters only')
        print(result['directory'], label)
        for row in result['requests']:
            print(f"  {row['kind']} drafts={row['width']} {row['tps']:.2f} TPS; verify/round={row['verify_ms_per_round']}")
