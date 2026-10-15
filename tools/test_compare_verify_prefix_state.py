import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from compare_verify_prefix_state import compare


class PrefixComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = struct.pack("<QI q iiii q qq", 0x3254534C4159414D, 1, 7, 0, 2, 1, 2, 4, 2, 2)
        for arm in ("graph-off", "graph-on"):
            stem = self.root / (arm+".f32")
            np.arange(6, dtype=np.float32).tofile(stem)
            np.arange(6, dtype=np.float32).tofile(str(stem)+".prefix.f32")
            for rep in range(2):
                Path(str(stem)+f".prefix-{rep}.slot").write_bytes(self.header+bytes(range(40)))

    def test_identical_and_localized_state_difference(self):
        r = compare(self.root, vocab=3)
        self.assertTrue(r["verification_logits"]["bitwise_exact"])
        self.assertTrue(r["snapshots"][0]["recurrent_state_and_conv"]["bitwise_exact"])
        path = self.root/"graph-on.f32.prefix-0.slot"
        data = bytearray(path.read_bytes()); data[len(self.header)+3] ^= 1; path.write_bytes(data)
        r = compare(self.root, vocab=3)
        self.assertEqual(r["snapshots"][0]["recurrent_state_and_conv"]["first_different_file_byte"], len(self.header)+3)
        self.assertTrue(r["snapshots"][0]["indexer_pool_and_dsa"]["bitwise_exact"])
        self.assertTrue(r["prefix_logits"]["bitwise_exact"])

    def test_prefix_difference_and_nonfinite_are_reported(self):
        path = self.root/"graph-on.f32.prefix.f32"
        data = np.arange(6, dtype=np.float32); data[4] = float("nan"); data.tofile(path)
        r = compare(self.root, vocab=3)
        self.assertFalse(r["prefix_logits"]["finite"])
        self.assertEqual(r["prefix_logits"]["first_different_row"], 1)

    def test_truncated_snapshot_rejected(self):
        (self.root/"graph-on.f32.prefix-0.slot").write_bytes(self.header+bytes(4))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            compare(self.root, vocab=3)

    def test_different_metadata_rejected(self):
        path = self.root/"graph-on.f32.prefix-0.slot"
        data = bytearray(path.read_bytes()); struct.pack_into("<q", data, 12, 8); path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "layouts differ"):
            compare(self.root, vocab=3)


if __name__ == "__main__":
    unittest.main()
