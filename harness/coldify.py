#!/usr/bin/env python3
"""把一个权重目录变成「冷」的：逐个文件用 F_NOCACHE 重写成新文件再改名覆盖。
旧文件的页缓存随旧 inode 释放，新文件写入时不进页缓存（本机无 sudo，不能 purge）。
一次只多占一个文件的盘。用法: python3 coldify.py <dir>
"""
import fcntl
import os
import sys
from pathlib import Path

F_NOCACHE = 48  # 来源：macOS <sys/fcntl.h> 中 F_NOCACHE 的值
CHUNK = 64 << 20  # 每次搬 64MB（只影响拷贝速度）

for p in sorted(Path(sys.argv[1]).rglob("*")):
    if not p.is_file():
        continue
    tmp = p.with_name(p.name + ".cold")
    src = os.open(p, os.O_RDONLY)
    dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    fcntl.fcntl(src, F_NOCACHE, 1)
    fcntl.fcntl(dst, F_NOCACHE, 1)
    n = 0
    while True:
        b = os.read(src, CHUNK)
        if not b:
            break
        os.write(dst, b)
        n += len(b)
    os.fsync(dst)
    os.close(src)
    os.close(dst)
    assert n == p.stat().st_size, p
    os.replace(tmp, p)
    print("cold", p.name, n, flush=True)
