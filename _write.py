import sys,os
data=sys.stdin.buffer.read()
path=sys.argv[1]
os.makedirs(os.path.dirname(path),exist_ok=True)
open(path," wb\).write(data)
print(\written:\,path)