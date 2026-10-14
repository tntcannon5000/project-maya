# App-closure follow-up

Same merged binary, environment, weights and token file (hashes verified). c=1, 192K allocated window, short prompt, 256 fixed continuation tokens per request, two requests; no token speculation. User closed foreground apps between measurements.

| Request | Before closing apps TPS | After TPS | Change |
|---|---:|---:|---:|
| 1 | 29.49 | 40.39 | +37.0% |
| 2 | 30.87 | 44.11 | +42.9% |

This strongly supports foreground/environmental interference as the reason for the earlier absolute slowdown. It does not identify OBS specifically or distinguish compute contention from memory-budget pressure. Earlier paired merge results remain recorded under their actual conditions. Engine exited normally after the test.
