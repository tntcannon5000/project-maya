"""Bounded, CPU-only replay of the saved window-union trace; no serving changes.

The simulations assume no unrecorded inference overwrites scratch between
captured windows. Initial resident tiers and CPU/promotion routing stay fixed.
Costs, lookup, admission and D2D time are excluded. This predicts no latency.
"""
import argparse
import bisect
from collections import defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path


def screen(rows, width):
    groups = defaultdict(list)
    for row in rows:
        if row['T'] == width:
            groups[(row['seq'] - row['token'], row['layer'])].append(row)
    layers, future = [], defaultdict(list)
    for index, ((sequence, layer), records) in enumerate(sorted(groups.items())):
        assert len(records) == width * 8
        owners = sorted([r for r in records if r['fetch_src']], key=lambda r: (r['token'], r['rank']))
        assert len({r['id'] for r in owners}) == len(owners)
        for row in owners:
            if row['ram_ready']:
                future[(layer, row['id'])].append(index)
        layers.append((layer, records, owners))
    baseline = sum(r['blob'] for _, _, owners in layers for r in owners)

    def simulate(kind, policy, capacity):
        cache = {}
        saved = transient_saved = promotion_saved = hits = admission = peak = max_pinned = clock = 0

        def victim(candidates, index):
            def next_use(key):
                queue = future[key]
                at = bisect.bisect_right(queue, index)
                return queue[at] if at < len(queue) else float('inf')
            if policy == 'farthest_next_use':
                return max(candidates, key=lambda k: (next_use(k), k))
            return min(candidates, key=lambda k: (cache[k][0], k))

        for index, (layer, records, owners) in enumerate(layers):
            for row in records:
                if row['initial_gpu']:
                    cache.pop((layer, row['id']), None)
            pinned, drop, new = set(), set(), {}
            workspaces = 0
            protected = {(layer, r['id']) for r in owners
                         if r['ram_ready'] and (layer, r['id']) in cache}
            for row in owners:
                key = (layer, row['id'])
                clock += 1
                eligible = bool(row['ram_ready'])
                if eligible and key in cache:
                    saved += row['blob']
                    hits += 1
                    cache[key] = (clock, row['blob'])
                    pinned.add(key)
                    if row['permanent']:
                        promotion_saved += row['blob']
                        drop.add(key)
                    else:
                        transient_saved += row['blob']
                elif not row['permanent']:
                    if kind == 'unified':
                        cache.pop(key, None)
                        while len(cache) + workspaces >= capacity:
                            available = set(cache) - pinned
                            if policy == 'farthest_next_use':
                                available -= protected
                            assert available, 'insufficient live workspace'
                            del cache[victim(available, index)]
                        if eligible:
                            cache[key] = (clock, row['blob'])
                            pinned.add(key)
                        else:
                            workspaces += 1
                        peak = max(peak, len(cache) + workspaces)
                    elif eligible:
                        new[key] = (clock, row['blob'])
                if not eligible:
                    drop.add(key)
            max_pinned = max(max_pinned, len(pinned) + workspaces)
            for key in drop:
                cache.pop(key, None)
            if kind == 'tail':
                cache.update(new)
                while len(cache) > capacity:
                    del cache[victim(set(cache), index)]
                admission += sum(value[1] for key, value in new.items() if key in cache)
                peak = max(peak, len(cache) + 8 * width)
            else:
                peak = max(peak, len(cache))
        return dict(kind=kind, policy=policy, capacity=capacity, baseline_bytes=baseline,
                    saved_H2D_bytes=saved, fraction=saved / baseline,
                    transient_hit_bytes=transient_saved, promotion_hit_bytes=promotion_saved,
                    hit_groups=hits, tail_D2D_admission_bytes=admission,
                    peak_shared_slots=peak, max_live_cache_or_temp_slots=max_pinned)

    output = [simulate('tail', 'lru', 64 - 8 * width),
              simulate('tail', 'farthest_next_use', 64 - 8 * width),
              simulate('unified', 'lru', 64), simulate('unified', 'farthest_next_use', 64)]
    unlimited = simulate('unified', 'lru', len(rows))
    unlimited.update(kind='unlimited_repeated_key_ceiling', capacity='unbounded; infeasible within64slots')
    output.append(unlimited)

    # Exact weighted interval solver used in the original CPU screen. A
    # cacheable upload becomes available after its layer; reuse must begin
    # before the future layer. Initial residency or promotion retires a key.
    last, intervals, live = {}, [], []
    for index, (layer, records, owners) in enumerate(layers):
        for row in records:
            if row['initial_gpu']:
                last.pop((layer, row['id']), None)
        live.append(sum(not r['permanent'] for r in owners))
        for row in owners:
            key = (layer, row['id'])
            if row['ram_ready'] and key in last:
                intervals.append((2 * last[key] + 1, 2 * index, row['blob'], bool(row['permanent'])))
            if row['ram_ready'] and not row['permanent']:
                last[key] = index
            else:
                last.pop(key, None)
    for name, capacity, workspace in [('tail_weighted_oracle', 64 - 8 * width, False),
                                       ('unified_weighted_oracle', 64, True)]:
        count = 2 * len(layers) + 1
        graph = [[] for _ in range(count)]

        def edge(source, target, cap, cost):
            forward = [target, len(graph[target]), cap, cost, cap]
            reverse = [source, len(graph[source]), 0, -cost, 0]
            graph[source].append(forward)
            graph[target].append(reverse)
            return forward

        for index in range(count - 1):
            edge(index, index + 1, capacity, 0)
        big = sum(item[2] for item in intervals) + 1
        active = [edge(2 * index, 2 * index + 1, number, -big)
                  for index, number in enumerate(live) if workspace and number]
        savings = [(edge(start, end, 1, -weight), weight, promoted, start, end)
                   for start, end, weight, promoted in intervals]
        flow = 0
        while flow < capacity:
            distance, previous, queued = [None] * count, [None] * count, [False] * count
            distance[0] = 0
            queue = deque([0])
            queued[0] = True
            while queue:
                source = queue.popleft()
                queued[source] = False
                for at, arc in enumerate(graph[source]):
                    if arc[2] and (distance[arc[0]] is None or distance[arc[0]] > distance[source] + arc[3]):
                        distance[arc[0]] = distance[source] + arc[3]
                        previous[arc[0]] = (source, at)
                        if not queued[arc[0]]:
                            queue.append(arc[0])
                            queued[arc[0]] = True
            assert previous[-1] is not None
            add, target = capacity - flow, count - 1
            while target:
                source, at = previous[target]
                add = min(add, graph[source][at][2])
                target = source
            target = count - 1
            while target:
                source, at = previous[target]
                arc = graph[source][at]
                arc[2] -= add
                graph[target][arc[1]][2] += add
                target = source
            flow += add
        assert all(arc[2] == 0 for arc in active), 'active workspace reservation failed'
        chosen = [(weight, promoted, start, end) for arc, weight, promoted, start, end in savings if arc[2] == 0]
        occupancy = [0] * (count - 1)
        for weight, promoted, start, end in chosen:
            for index in range(start, end):
                occupancy[index] += 1
        if workspace:
            for index, number in enumerate(live):
                occupancy[2 * index] += number
        assert max(occupancy, default=0) <= capacity
        saved = sum(item[0] for item in chosen)
        output.append(dict(kind=name, capacity=capacity, saved_H2D_bytes=saved,
                           fraction=saved / baseline, hit_groups=len(chosen), peak_slots=max(occupancy),
                           max_active_transient_slots=max(live), active_workspace_reserved=workspace,
                           promotion_hit_bytes=sum(w for w, p, s, t in chosen if p),
                           transient_hit_bytes=sum(w for w, p, s, t in chosen if not p),
                           caveat='Future knowledge and free lookup/admission/D2D; promotion source lease ends ideally at layer entry; interval-class bound, not universal optimality or latency.'))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.trace.open(newline='') as stream:
        rows = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    result = {f'T{width}': screen(rows, width) for width in sorted({row['T'] for row in rows})}
    result['trace_sha256'] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for width, simulations in result.items():
        if isinstance(simulations, list):
            print(width, [(row['kind'], row.get('policy'), round(100 * row['fraction'], 3)) for row in simulations])


if __name__ == '__main__':
    main()
