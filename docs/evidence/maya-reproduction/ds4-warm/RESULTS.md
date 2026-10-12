# One warmed DS4 pass

Same retained DS4 core objects, antirez Q2, 131072 context capacity, 700 MiB
reserve, greedy, no speculation. A test-only driver includes the unmodified CLI
generation path, loads the model once, generates an unscored 64-token hello,
then runs each of the three 256-token tasks once. A 2048-token prefill warm-up
precedes the scored 2048/8192-token prompts. No tuning or engine edits.

| TPS | Previous cold DS4 | Warm-process DS4 | Stock Maya-M |
|---|---:|---:|---:|
| CSV parser | 30.22 | 30.60 | 25.0 |
| Photosynthesis | 30.21 | 30.58 | 27.7 |
| HTML/JS animation | 30.33 | 30.84 | 25.5 |
| Mean decode | 30.25 | 30.67 | 26.0 |
| Prefill 2048 | 829.76 | 955.90 | 1141 |
| Prefill 8192 | 911.03 | 819.38 | 2314 |

Mean decode increased about 1.4%; one pass does not establish statistical
significance. The warm 8K sample now starts at token offset 997 to match Maya's
recipe, unlike the previous DS4 opening slice. Do not attribute its difference
solely to warming. Tokenizers/model recipes differ across engines.

Warm-process is precise here: the normal DS4 generation entry creates fresh
conversation graphs, and its prefill path rebuilds the 2560-slot decode cache
between requests (visible in log.txt). This run warms the process, model RAM and
CUDA paths, but does not bypass normal cache management or measure repeated
decode from an artificially fully resident expert set. No claim that cache state
is identical to Maya's is made.

The verdict remains: DS4 Q2 leads observed decode, Maya-M leads observed prefill;
different weights still prevent an engine-only conclusion. The model process
exited successfully. The full run, including unscored warm-ups, is in log.txt.
