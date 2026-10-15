"""Compact route/group evidence for the bounded first-MoE diagnostic.

Arithmetic replay belongs to maya_moe_combine_replay (production CUDA kernel),
not this Python analysis. Raw captures remain local.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    data = path.read_bytes()
    offset = 0

    def take(dtype, n):
        nonlocal offset
        size = np.dtype(dtype).itemsize * n
        if offset + size > len(data):
            raise ValueError('truncated capture')
        result = np.frombuffer(data, dtype=dtype, count=n, offset=offset).copy()
        offset += size
        return result

    magic, t, k, e = take('<i4', 4)
    if magic != 0x4d4f4531 or t < 1 or not 0 < k <= 8 or not 0 < e <= 65536:
        raise ValueError('invalid capture header')
    ids, pos, weights = take('<i4', k), take('<i4', k), take('<f4', k)
    initial = take('<f4', e)
    outputs = np.zeros((k, e), dtype=np.float32)
    groups, seen = [], set()
    while True:
        n = int(take('<i4', 1)[0])
        if n == -1:
            break
        if not 0 < n <= k:
            raise ValueError('invalid group')
        group = []
        for _ in range(n):
            j = int(take('<i4', 1)[0])
            if not 0 <= j < k or j in seen:
                raise ValueError('invalid/duplicate route')
            seen.add(j)
            group.append(j)
            outputs[j] = take('<f4', e)
        groups.append(group)
    final = take('<f4', e)
    if offset != len(data) or len(seen) != k:
        raise ValueError('trailing/incomplete capture')
    for value in (weights, initial, outputs, final):
        if not np.isfinite(value).all():
            raise ValueError('nonfinite capture')
    return dict(t=int(t), k=int(k), e=int(e), ids=ids, positions=pos,
                weights=weights, initial=initial, outputs=outputs, final=final,
                groups=groups, sha256=hashlib.sha256(data).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cold', type=Path)
    parser.add_argument('warm', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    a, b = read(args.cold), read(args.warm)
    if (a['t'], a['k'], a['e']) != (b['t'], b['k'], b['e']) or not np.array_equal(a['ids'], b['ids']):
        raise ValueError('incompatible routes/shapes')
    result = {'tokens': a['t'], 'experts': a['k'], 'embedding': a['e'],
              'cold_sha256': a['sha256'], 'warm_sha256': b['sha256'],
              'cold_groups_route_indices': a['groups'], 'warm_groups_route_indices': b['groups'],
              'routes': []}
    for j in range(a['k']):
        x, y = a['outputs'][j], b['outputs'][j]
        result['routes'].append(dict(route=j, expert=int(a['ids'][j]),
            cold_sorted_row=int(a['positions'][j]), warm_sorted_row=int(b['positions'][j]),
            weight=float(a['weights'][j]), expert_output_max_abs=float(np.abs(x-y).max()),
            expert_output_different_bits=int(np.count_nonzero(x.view('u4') != y.view('u4')))))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
