"""Causal scratch-retention screens. No model execution or future-route access."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_layers(rows, width):
    grouped = defaultdict(list)
    for row in rows:
        if row['T'] == width:
            grouped[(row['seq'] - row['token'], row['layer'])].append(row)
    layers = []
    window = -1
    previous = None
    for (_, layer), records in sorted(grouped.items()):
        assert len(records) == width * 8
        assert len({(r['token'], r['rank']) for r in records}) == len(records)
        if previous is None or layer <= previous:
            window += 1
        previous = layer
        owners = sorted((r for r in records if r['fetch_src']), key=lambda r: (r['token'], r['rank']))
        assert len({r['id'] for r in owners}) == len(owners)
        layers.append(dict(layer=layer, records=records, owners=owners, window=window))
    return layers


def simulate(layers, policy, capacity=64, protect_current=True):
    """Every decision uses only prior history and the currently routed layer.

    Slots are max-blob-sized. Current-layer cached sources and transient
    destinations stay live through completion; promotions still require D2D.
    No inference between recorded layers, no lookup/copy cost modeled.
    """
    assert policy in ('lru', 'frequency', 'gdsf')
    cache, frequency = {}, defaultdict(int)
    clock = 0
    age = 0
    events = []
    for entry in layers:
        layer, records, owners = entry['layer'], entry['records'], entry['owners']
        for r in records:
            if r['initial_gpu']:
                cache.pop((layer, r['id']), None)
        protected = {(layer, r['id']) for r in owners if r['ram_ready'] and (layer, r['id']) in cache}
        pinned = set(protected) if protect_current else set()
        drop = set()
        workspaces = 0
        event = dict(window=entry['window'], layer=layer, fetched_bytes=sum(r['blob'] for r in owners),
                     saved_bytes=0, promotion_D2D_bytes=0, hits=0, evictions=0, peak_slots=len(cache))
        for r in owners:
            key = layer, r['id']
            clock += 1
            eligible = bool(r['ram_ready'])
            if eligible:
                frequency[key] += 1
            score = frequency[key] * r['blob']
            if policy == 'lru':
                score = clock
            elif policy == 'gdsf':
                score += age
            if eligible and key in cache:
                event['saved_bytes'] += r['blob']
                event['hits'] += 1
                cache[key] = (score, clock)
                pinned.add(key)
                if r['permanent']:
                    event['promotion_D2D_bytes'] += r['blob']
                    drop.add(key)
            elif not r['permanent']:
                cache.pop(key, None)
                while len(cache) + workspaces >= capacity:
                    available = set(cache) - pinned
                    assert available, 'active workspace exceeds capacity'
                    victim = min(available, key=lambda k: (*cache[k], k))
                    if policy == 'gdsf':
                        age = max(age, cache[victim][0])
                        score = age + frequency[key] * r['blob']
                    del cache[victim]
                    event['evictions'] += 1
                if eligible:
                    cache[key] = score, clock
                    pinned.add(key)
                else:
                    workspaces += 1
            if not eligible:
                drop.add(key)
            event['peak_slots'] = max(event['peak_slots'], len(cache) + workspaces)
            assert event['peak_slots'] <= capacity
        for key in drop:
            cache.pop(key, None)
        event['resident_keys'] = len(cache)
        events.append(event)
    return events


def summarize(events):
    total = sum(e['fetched_bytes'] for e in events)
    saved = sum(e['saved_bytes'] for e in events)
    windows = len({e['window'] for e in events})
    return dict(fetched_bytes=total, saved_bytes=saved, fraction=saved / total if total else 0,
                windows=windows, saved_MB_per_window=saved / windows / 1e6 if windows else 0,
                promotion_D2D_bytes=sum(e['promotion_D2D_bytes'] for e in events),
                hits=sum(e['hits'] for e in events), peak_slots=max((e['peak_slots'] for e in events), default=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(args.trace.open())]
    result = dict(trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  caveat='Causal trace simulation, not measured TPS; shared64slot capacity; all policies start empty; no parameter fitting; current-layer route lookahead requires full-layer scheduling.')
    for width in sorted({r['T'] for r in rows}):
        layers = load_layers(rows, width)
        split = (max(e['window'] for e in layers) + 1) // 2
        result[f'T{width}'] = {}
        for policy, protect in [('lru', False), ('lru', True), ('frequency', True), ('gdsf', True)]:
            name = policy + ('_protect_current' if protect else '_legacy')
            events = simulate(layers, policy, protect_current=protect)
            result[f'T{width}'][name] = dict(all=summarize(events),
                first_half=summarize([e for e in events if e['window'] < split]),
                second_half=summarize([e for e in events if e['window'] >= split]))
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
