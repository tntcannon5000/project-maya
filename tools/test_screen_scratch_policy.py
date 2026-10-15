import unittest
from screen_scratch_policy import simulate


def entry(layer, keys, *, permanent=(), ready=True, resident=(), window=0):
    owners = [dict(id=k, blob=100*(k+1), ram_ready=int(ready), permanent=int(k in permanent), initial_gpu=0)
              for k in keys]
    records = owners + [dict(id=k, initial_gpu=1) for k in resident]
    return dict(layer=layer, records=records, owners=owners, window=window)


class ScratchPolicyTests(unittest.TestCase):
    def test_prefix_independent_of_future(self):
        prefix = [entry(0,[0]), entry(1,[1]), entry(0,[0])]
        for policy in ('lru','frequency','gdsf','phase_frequency'):
            self.assertEqual(simulate(prefix,policy,2), simulate(prefix+[entry(2,[2])],policy,2)[:3])

    def test_promotion_needs_copy_and_retires_cache(self):
        events=simulate([entry(0,[1]),entry(0,[1],permanent=(1,)),entry(0,[1],permanent=(1,))],'lru',2)
        self.assertEqual([e['saved_bytes'] for e in events],[0,200,0])
        self.assertEqual(events[1]['promotion_D2D_bytes'],200)
        self.assertEqual(events[1]['resident_keys'],0)

    def test_current_layer_hits_cannot_be_evicted_for_earlier_miss(self):
        seq=[entry(0,[0]),entry(1,[1]),entry(0,[2,0])]
        self.assertEqual(simulate(seq,'lru',2,False)[-1]['saved_bytes'],0)
        self.assertEqual(simulate(seq,'lru',2,True)[-1]['saved_bytes'],100)

    def test_pending_sources_are_not_admitted(self):
        seq=[entry(0,[0],ready=False),entry(0,[0]),entry(0,[0],ready=False),entry(0,[0])]
        self.assertEqual(sum(e['saved_bytes'] for e in simulate(seq,'frequency',1)),0)

    def test_capacity_includes_live_consumers(self):
        with self.assertRaisesRegex(AssertionError,'workspace'):
            simulate([entry(0,[0,1,2])],'gdsf',2)
        seq=[entry(0,[0,1]),entry(1,[2,3]),entry(0,[0,1])]
        self.assertEqual(max(e['peak_slots'] for e in simulate(seq,'gdsf',2)),2)

    def test_initial_residency_retires_duplicate(self):
        seq=[entry(0,[0]),entry(0,[],resident=(0,)),entry(0,[0])]
        self.assertEqual(sum(e['saved_bytes'] for e in simulate(seq,'lru',1)),0)

if __name__=='__main__':unittest.main()
