#!/usr/bin/env python3
"""Drama 3 RSI harness：一个候选跑完 manifest 里冻结的台词组，记加载、内存、每句耗时与分段计时。

用法: python -I bench.py <candidate.json> <outdir>
candidate.json: {"id", "build", "knobs": {...}, "src": "<mlx_speech 源码目录（含 mlx_speech 包的上一层）>"}
输出: <outdir>/wav/<line>_s<seed>.wav, <outdir>/stats.json
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
# D3_MANIFEST 可换台词组（如 stress.json），默认冻结的 manifest.json
MAN = json.loads((HERE / os.environ.get("D3_MANIFEST", "manifest.json")).read_text())
GB = 1024**3
KNOBS = {"dense_int8", "wired_limit_gb", "cache_limit_gb", "clear_cache_after_line", "warmup",
         # experts on demand (offload.py)
         "pin_layers", "slots", "prefetch", "nocache", "lean_head", "codec_tile", "memory_limit_gb"}


def main() -> None:
    cand = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); (out / "wav").mkdir(parents=True, exist_ok=True)
    knobs = cand.get("knobs", {})
    bad = set(knobs) - KNOBS
    if bad:
        raise SystemExit(f"unknown knobs {bad}")
    sys.path.insert(0, str(Path(cand.get("src") or HERE / "src-base")))
    import mlx.core as mx
    import mlx_speech
    from mlx_speech.audio import write_wav
    from mlx_speech.generation import fish_drama_3 as gen
    from mlx_speech.tts._adapters.fish_drama_3 import FishDrama3Adapter

    if knobs.get("wired_limit_gb"):
        mx.set_wired_limit(int(knobs["wired_limit_gb"] * GB))
    if knobs.get("memory_limit_gb"):
        mx.set_memory_limit(int(knobs["memory_limit_gb"] * GB))
    if knobs.get("cache_limit_gb") is not None:
        mx.set_cache_limit(int(knobs["cache_limit_gb"] * GB))

    model_dir = str(HERE / MAN["builds"][cand["build"]]["dir"])  # 相对路径按本目录算
    phase = {"frames_s": 0.0, "codec_s": 0.0}
    orig_gf = gen.FishDrama3Runtime.generate_frames

    def gf(self, *a, **k):
        t = time.perf_counter(); r = orig_gf(self, *a, **k); phase["frames_s"] += time.perf_counter() - t; return r
    gen.FishDrama3Runtime.generate_frames = gf

    t0 = time.perf_counter()
    store = None
    if knobs.get("slots") is not None:
        sys.path.insert(0, str(HERE))
        import offload
        rt, store = offload.load_offloaded(
            model_dir, pin_layers=knobs.get("pin_layers", 0), slots=knobs["slots"], prefetch=knobs.get("prefetch", False), nocache=knobs.get("nocache", True),
            lean_head=knobs.get("lean_head", True), dense_int8=bool(knobs.get("dense_int8")),
            codec_tile=knobs.get("codec_tile", 64))
    else:
        rt = gen.FishDrama3Runtime.from_dir(model_dir, dense_int8=bool(knobs.get("dense_int8")))
    load_s = time.perf_counter() - t0
    codec_decode = rt.codec.decode

    def cd(x):
        t = time.perf_counter(); y = codec_decode(x); mx.eval(y); phase["codec_s"] += time.perf_counter() - t; return y
    rt.codec.decode = cd
    model = FishDrama3Adapter(rt)
    stats = {"id": cand["id"], "build": cand["build"], "knobs": knobs, "src": str(mlx_speech.__file__),
             "load_s": round(load_s, 1), "after_load_active_gb": round(mx.get_active_memory() / GB, 2),
             "rss_after_load_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, 2), "lines": []}
    print(f"[{cand['id']}] load {load_s:.1f}s active {stats['after_load_active_gb']}GB", flush=True)
    t = time.perf_counter()
    ref = model.prepare_reference(str(HERE / MAN["ref_audio"]), reference_text=MAN["ref_text"])
    stats["ref_s"] = round(time.perf_counter() - t, 2)
    if knobs.get("warmup"):
        t = time.perf_counter()
        model.generate(MAN["warmup_text"], reference_audio=ref, seed=0)
        stats["warmup_s"] = round(time.perf_counter() - t, 2)
    mx.reset_peak_memory()
    for seed in MAN["seeds"]:
        for name, text in MAN["lines"]:
            phase["frames_s"] = phase["codec_s"] = 0.0
            t = time.perf_counter()
            res = model.generate(text, reference_audio=ref, seed=seed)
            mx.eval(res.waveform)
            el = time.perf_counter() - t
            if knobs.get("clear_cache_after_line"):
                mx.clear_cache()
            wav = out / "wav" / f"{name}_s{seed}.wav"
            write_wav(wav, res.waveform, sample_rate=res.sample_rate)
            dur = res.waveform.shape[-1] / res.sample_rate
            row = {"name": name, "seed": seed, "audio_s": round(dur, 2), "gen_s": round(el, 3),
                   "frames_s": round(phase["frames_s"], 3), "codec_s": round(phase["codec_s"], 3),
                   "rtf": round(el / max(dur, 1e-3), 3),
                   "mlx_peak_gb": round(mx.get_peak_memory() / GB, 2),
                   "rss_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, 2)}
            if store is not None:
                row["experts"] = store.stats()
            stats["lines"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    L = stats["lines"]
    first = L[0]["gen_s"]
    warm = [r for r in L[1:]]
    stats["summary"] = {
        "load_s": stats["load_s"],
        "peak_gb": max(r["mlx_peak_gb"] for r in L),
        "rss_peak_gb": max(r["rss_peak_gb"] for r in L),
        "first_line_s": first,
        "warm_rtf": round(sum(r["gen_s"] for r in warm) / sum(r["audio_s"] for r in warm), 4),
        "warm_gen_s_mean": round(sum(r["gen_s"] for r in warm) / len(warm), 3),
        "frames_share": round(sum(r["frames_s"] for r in L) / sum(r["gen_s"] for r in L), 3),
        "codec_share": round(sum(r["codec_s"] for r in L) / sum(r["gen_s"] for r in L), 3),
    }
    if store is not None:
        stats["summary"]["experts"] = store.stats()
        (out / "trace.json").write_text(json.dumps(store.trace))
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print("SUMMARY", json.dumps(stats["summary"]), flush=True)


if __name__ == "__main__":
    main()
