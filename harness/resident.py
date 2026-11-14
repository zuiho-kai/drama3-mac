#!/usr/bin/env python3
"""用 mincore 查一个目录里文件有多少页在页缓存里。用法: python3 resident.py <dir>"""
import ctypes, mmap, os, sys
from pathlib import Path
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PG = os.sysconf("SC_PAGE_SIZE")
tot = res = 0
for p in sorted(Path(sys.argv[1]).rglob("*.safetensors")):
    n = p.stat().st_size
    fd = os.open(p, os.O_RDONLY)
    a = libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    pages = (n + PG - 1) // PG
    vec = ctypes.create_string_buffer(pages)
    assert libc.mincore(a, n, vec) == 0, ctypes.get_errno()
    r = sum(1 for b in vec.raw if b & 1)
    libc.munmap(a, n); os.close(fd)
    tot += pages; res += r
    print(f"{p.name}: {100*r/pages:.1f}% resident")
print(f"TOTAL {100*res/tot:.1f}% resident of {tot*PG/2**30:.1f} GiB")
