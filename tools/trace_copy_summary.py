import sqlite3,sys,json
c=sqlite3.connect(sys.argv[1])
print(c.execute('select * from ENUM_CUDA_MEMCPY_OPER').fetchall())
print('Copies: kind,count,MB,total-ms,min-bytes,max-bytes')
print(c.execute('select copyKind,count(*),sum(bytes)/1e6,sum(end-start)/1e6,min(bytes),max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind').fetchall())
print('largest copy groups')
print(c.execute('select copyKind,bytes,count(*),sum(end-start)/1e6 from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind,bytes order by sum(end-start) desc limit 12').fetchall())
print('API totals excluding event polling')
print(c.execute("select s.value,count(*),sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value!='cudaEventQuery' group by r.nameId order by sum(r.end-r.start) desc limit 12").fetchall())
