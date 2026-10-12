"""Offline final numerical qualification; run after inference has stopped."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1] / 'docs/evidence/maya-demand'
results = []
for baseline, candidate in [('baseline', 'confirm-graphs'),
                            ('long-baseline', 'long-graphs')]:
    manifests = [json.loads((root / n / 'manifest.json').read_text())
                 for n in (baseline, candidate)]
    assert manifests[0]['tokens_sha256'] == manifests[1]['tokens_sha256']
    for manifest in manifests:
        manifest['weights'].setdefault('experiment_files', {})
    assert manifests[0]['weights'] == manifests[1]['weights']
    arrays = [np.memmap(root / n / 'fixed.logits.f32', dtype=np.float32, mode='r')
              for n in (baseline, candidate)]
    assert arrays[0].shape == arrays[1].shape
    exact, finite, maximum = True, True, 0.0
    for start in range(0, len(arrays[0]), 154880):
        a, b = [x[start:start + 154880] for x in arrays]
        finite &= bool(np.isfinite(a).all() and np.isfinite(b).all())
        exact &= bool(np.array_equal(a, b))
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64) - b))))
    results.append(dict(name=f'{baseline} vs {candidate}', logits=len(arrays[0]),
                        exact=exact, finite=finite, max_abs=maximum))
    assert finite and exact, results[-1]
    del arrays
(root / 'final-parity.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
