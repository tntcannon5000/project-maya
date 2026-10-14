"""Two back-to-back c4 cohorts per binary, no experimental concurrent graphs."""
import run_c4_ab as bench
for label,exe in [('before','strata-c4-retained-baseline.exe'),('merged','strata.exe')]:
 bench.OUT=bench.R/'docs/evidence/maya-release-integration'/('c4-'+label)
 bench.run('retained',bench.R/'build'/exe)
