import sqlite3, sys
c=sqlite3.connect(sys.argv[1])
for name,sql in c.execute("select name,sql from sqlite_master where type='table'"):
    if any(x in name for x in ('KERNEL','RUNTIME','MEMCPY','NVTX','StringIds')):
        print(sql)
        print(c.execute('select * from '+name+' limit 1').fetchone())
