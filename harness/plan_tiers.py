#!/usr/bin/env python3
"""Pick, per memory budget, how many leading layers keep all their experts resident
(no per-layer sync, no misses) and how many slots the other layers get, by replaying
an expert trace. Early layers route to many different experts; deep layers reuse a
few, so the leading layers are the ones worth pinning.

usage: python3 plan_tiers.py <trace.json> <expert_mb> <budget_gb>...
time model (from measurements in README): frame = base + syncs*sync_ms + read_mb/ssd
"""
import json, sys
from collections import OrderedDict

BASE_MS, SYNC_MS, SSD_GBS = 21.0, 0.55, 5.0

def per_layer_misses(dec, S):
    out = [0] * 40
    for L in range(40):
        c = OrderedDict()
        for l, ex in dec:
            if l != L: continue
            for e in ex:
                if e in c: c.move_to_end(e)
                else:
                    out[L] += 1; c[e] = 1
            while len(c) > S: c.popitem(last=False)
    return out

def main():
    tr = json.load(open(sys.argv[1])); mb = float(sys.argv[2])
    dec = [(L, ex) for L, ex in tr if len(ex) <= 8]
    frames = sum(1 for L, _ in dec if L == 0)
    memo = {}
    for g in map(float, sys.argv[3:]):
        B = int(g * 1024 / mb); best = None
        for N in range(0, 41):
            if 256 * N > B: break
            S = 256 if N == 40 else min(256, (B - 256 * N) // (40 - N))
            if N < 40 and S < 8: continue
            if S not in memo: memo[S] = per_layer_misses(dec, S)
            miss = sum(memo[S][N:]) / frames
            ms = BASE_MS + (40 - N) * SYNC_MS + miss * mb / 1024 / SSD_GBS * 1000
            if best is None or ms < best[0]: best = (ms, N, S, miss)
        ms, N, S, miss = best
        print(json.dumps({"budget_gb": g, "pin_layers": N, "slots": S, "miss_per_frame": round(miss, 1),
                          "est_rtf": round(ms / 1000 * 21.5, 2)}))

main()
