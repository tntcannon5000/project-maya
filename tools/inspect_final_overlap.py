"""Prove captured-only component engagement from kernel records."""
import json,sqlite3
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'docs/evidence/maya-residency-overlap/trace-final'
d=sqlite3.connect(p/'timeline.sqlite')
rows=d.execute("""select k.gridX, coalesce(k.graphId,0)!=0, count(*)
from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
where s.value like '%moe_component_kernel%' group by 1,2""").fetchall()
assert rows and all(captured for grid,captured,count in rows)
assert any(grid==32 for grid,captured,count in rows)
result=[dict(blocks=grid,captured=bool(captured),launches=count) for grid,captured,count in rows]
(p/'component-engagement.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
