"""Synthetic trace checks: missing API data must not imply early submission."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from analyze_verify_submission import analyze

class SubmissionTest(unittest.TestCase):
    def fixture(self, path, mode):
        c=sqlite3.connect(path)
        c.executescript("""CREATE TABLE StringIds(id INTEGER,value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,streamId INTEGER,demangledName INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,nameId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,correlationId INTEGER,streamId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,correlationId INTEGER,streamId INTEGER);""")
        names=['embed_streams_kernel','moe_route_kernel','moe_cpu_wait_kernel','argmax_kernel','cudaLaunchKernel']
        c.executemany('INSERT INTO StringIds VALUES (?,?)',enumerate(names))
        tick=1000; correlation=0
        for width in (3,5,3,5):
            sequence=[0]*width+([1]*width+[2]*width)*42+[3]*width
            for name in sequence:
                correlation+=1
                c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',(tick,tick+10,correlation,13,name))
                if not (mode=='missing' and correlation==2):
                    start=tick-5 if mode=='late' and correlation==2 else tick-100
                    c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)',(start,start+2,correlation,4))
                tick+=20
        c.commit();c.close()

    def check(self,mode):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.sqlite';self.fixture(path,mode)
            return analyze(path)

    def test_complete(self):
        data=self.check('normal')
        self.assertEqual(data['activity_count'],data['activity_correlation_matches'])
        for window in data['windows']:
            self.assertEqual(window['gap_count'],window['matched_gap_count'])
            self.assertEqual(window['late_api_start_gap_ms'],0)
            self.assertEqual(window['api_not_returned_gap_ms'],0)
            self.assertAlmostEqual(window['span_ms'],window['busy_union_ms']+window['gpu_gap_ms'])

    def test_missing_is_visible(self):
        window=self.check('missing')['windows'][0]
        self.assertEqual(window['gap_count']-window['matched_gap_count'],1)
        self.assertAlmostEqual(window['unmatched_gap_ms'],0.00001)

    def test_late_is_counted(self):
        window=self.check('late')['windows'][0]
        self.assertAlmostEqual(window['late_api_start_gap_ms'],0.000005)
        self.assertAlmostEqual(window['api_not_returned_gap_ms'],0.000007)

if __name__=='__main__':unittest.main()
