"""Locate differences before/after verification; never infer cause from a rerun."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def float_compare(a, b, vocab):
    if a.size == 0 or a.shape != b.shape or a.size % vocab:
        raise ValueError("Invalid or mismatched logit rows")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    different = a.view(np.uint32) != b.view(np.uint32)
    rows = different.reshape(-1, vocab).any(axis=1)
    return dict(values=int(a.size), finite=finite, bitwise_exact=not bool(different.any()),
                first_different_row=int(np.flatnonzero(rows)[0]) if rows.any() else None,
                max_abs=float(np.abs(a-b).max()) if finite else None)


def slot_layout(path):
    with path.open("rb") as f:
        raw = f.read(44)
        if len(raw) != 44:
            raise ValueError("Truncated slot header: " + str(path))
        magic, halves, position = struct.unpack_from("<QIq", raw)
        first, last, recurrent, runs, run = struct.unpack_from("<iiiiq", raw, 20)
        if magic != 0x3254534C4159414D or halves != 1 or position <= 0 or not (0 <= first < last):
            raise ValueError("Expected one-part MAYALST2 snapshot")
        if recurrent <= 0 or run <= 0 or runs < 0 or runs > 256:
            raise ValueError("Invalid state dimensions")
        sizes = f.read(runs * 8)
        if len(sizes) != runs * 8:
            raise ValueError("Truncated run sizes")
        lengths = struct.unpack("<" + "q" * runs, sizes)
        offset = 44 + len(sizes)
        kda_bytes = recurrent * run * 4
        if any(n < 0 for n in lengths) or path.stat().st_size < offset + kda_bytes + sum(lengths)*4:
            raise ValueError("Truncated state payload")
        return dict(header_hex=(raw+sizes).hex(), position=position, payload_offset=offset,
                    recurrent_layers=recurrent, recurrent_bytes=kda_bytes, bytes=path.stat().st_size)


def byte_region(a, b, offset, length):
    hashes = [hashlib.sha256(), hashlib.sha256()]
    first = None
    count = 0
    with a.open("rb") as fa, b.open("rb") as fb:
        fa.seek(offset); fb.seek(offset)
        while count < length:
            size = min(1024*1024, length-count)
            x, y = fa.read(size), fb.read(size)
            if len(x) != size or len(y) != size:
                raise ValueError("Short snapshot read")
            hashes[0].update(x); hashes[1].update(y)
            if first is None and x != y:
                first = offset + count + next(i for i, (u, v) in enumerate(zip(x, y)) if u != v)
            count += size
    return dict(bytes=length, bitwise_exact=first is None, first_different_file_byte=first,
                sha256=[h.hexdigest() for h in hashes])


def compare(directory, vocab=154880, repeats=2):
    stems = [directory / name for name in ("graph-off.f32", "graph-on.f32")]
    windows = [np.fromfile(p, dtype=np.float32) for p in stems]
    prefix = [np.fromfile(str(p)+".prefix.f32", dtype=np.float32) for p in stems]
    if any(x.size != repeats*vocab for x in prefix):
        raise ValueError("Missing prefix rows")
    result = {"prefix_logits": float_compare(*prefix, vocab),
              "verification_logits": float_compare(*windows, vocab), "snapshots": []}
    for rep in range(repeats):
        paths = [Path(str(p)+f".prefix-{rep}.slot") for p in stems]
        layouts = [slot_layout(p) for p in paths]
        if layouts[0] != layouts[1]:
            raise ValueError("Snapshot layouts differ; compare metadata before interpreting bytes")
        layout = layouts[0]; offset = layout["payload_offset"]; kda = layout["recurrent_bytes"]
        result["snapshots"].append(dict(repeat=rep, layout=layout,
            recurrent_state_and_conv=byte_region(*paths, offset, kda),
            indexer_pool_and_dsa=byte_region(*paths, offset+kda, layout["bytes"]-offset-kda)))
    result["limitations"] = ("Snapshots allocate/copy state and may perturb timing. Identical serialized state proves "
        "equality of captured state, not expert residency/backend placement. Open indexer-pool storage includes "
        "rows not yet occupied; a difference confined there is not automatically causal. Historical failure "
        "without saved state cannot be attributed from a passing rerun.")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    result = compare(args.directory)
    (args.directory/"state-comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
