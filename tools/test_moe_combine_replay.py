"""Bounded production-kernel replay checks; requires built CUDA executable."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def capture(groups, final):
    # Combined sum cancels to zero before adding initial=1. Split sums lose 1.
    data = struct.pack('<4i', 0x4d4f4531, 1, 2, 1)
    data += struct.pack('<4i', 10, 20, 100, 200)
    data += struct.pack('<3f', 1, 1, 1)
    for group in groups:
        data += struct.pack('<i', len(group))
        for j in group:
            data += struct.pack('<if', j, 1e20 if j == 0 else -1e20)
    return data + struct.pack('<if', -1, final)


class ReplayTests(unittest.TestCase):
    def run_pair(self, a, b):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp)/'a.bin', Path(tmp)/'b.bin'
            left.write_bytes(a)
            right.write_bytes(b)
            env = os.environ.copy()
            env['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;' + env['PATH']
            return subprocess.run([str(ROOT/'build/maya_moe_combine_replay.exe'), str(left), str(right)],
                                  capture_output=True, text=True, env=env, timeout=30)

    def test_actual_kernel_partition_effect(self):
        r = self.run_pair(capture([[0, 1]], 1), capture([[0], [1]], 0))
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertEqual(result['cold_replay']['different_bits'], 0)
        self.assertEqual(result['warm_replay']['different_bits'], 0)
        self.assertEqual(result['cold_values_warm_partition_vs_cold']['max_abs'], 1)
        self.assertEqual(result['expert_outputs']['different_bits'], 0)

    def test_reject_more_than_eight_routes(self):
        r = self.run_pair(struct.pack('<4i', 0x4d4f4531, 1, 9, 1), capture([[0, 1]], 1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('bad header', r.stderr)

    def test_reject_truncation(self):
        r = self.run_pair(capture([[0, 1]], 1)[:-1], capture([[0, 1]], 1))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('truncated', r.stderr)


if __name__ == '__main__':
    unittest.main()
