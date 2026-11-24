#!/usr/bin/env python3
"""Replay an expert trace (bench.py with offload writes runs/<id>/trace.json) against
cache budgets: hit rate and GB read per policy, without running the model.

usage: python3 simcache.py <trace.json> <expert_mb> <budget_gb>...
"""
import json, sys
from collections import OrderedDict, Counter

def lru(trace, cap):
    c, hit, miss = OrderedDict(), 0, 0
    for L, ex in trace:
        for e in ex:
            k = (L, e)
            if k in c: c.move_to_end(k); hit += 1
            else:
                miss += 1; c[k] = 1
        while len(c) > cap: c.popitem(last=False)
    return hit, miss

def main():
    trace = json.load(open(sys.argv[1])); mb = float(sys.argv[2])
    for g in map(float, sys.argv[3:]):
        cap = int(g * 1024 / mb)
        h, m = lru(trace, cap)
        print(f"budget {g:5.1f} GB ({cap} experts): LRU hit {h/(h+m):.3f}, read {m*mb/1024:.1f} GB")

main()
