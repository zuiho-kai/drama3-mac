#!/usr/bin/env python3
"""Drama 3 RSI 搜索循环（无人值守）：冠军基线两遍 → 写死门槛 → 预置 4 位候选 → 模型提案若干轮。

每轮：propose.py 出候选 → bench.py 测 → quality.py 过音质 → 裁判 → ledger.jsonl 入账（成败都记、记花费）→ 更好才晋级。
用法: python3 loop.py --rounds <轮数>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import os
MLX_PY = os.environ.get("D3_PY", sys.executable)  # 装了 mlx-speech 的 python
ASR_PY = os.environ.get("D3_ASR_PY", sys.executable)  # 装了 mlx-whisper、zhconv 的 python
METRICS = ("load_s", "peak_gb", "warm_rtf")  # 交接：加载时间、内存峰值、合成耗时，越小越好


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def log(*a) -> None:
    print(now(), *a, flush=True)


def man() -> dict:
    return json.loads((HERE / "manifest.json").read_text())


def tail(p: Path) -> str:
    return p.read_text()[-600:]  # 只留错误尾巴进账本


def run_one(cand: dict, run_dir: Path, ref_run: Path | None) -> tuple[dict, dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cf = run_dir / "candidate.json"
    cf.write_text(json.dumps(cand, ensure_ascii=False, indent=1))
    with open(run_dir / "bench.log", "w") as f:
        r = subprocess.run([MLX_PY, "-I", str(HERE / "bench.py"), str(cf), str(run_dir)], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"bench rc={r.returncode}: " + tail(run_dir / "bench.log"))
    args = [ASR_PY, "-I", str(HERE / "quality.py"), str(run_dir)] + ([str(ref_run)] if ref_run else [])
    with open(run_dir / "quality.log", "w") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"quality rc={r.returncode}: " + tail(run_dir / "quality.log"))
    return (json.loads((run_dir / "stats.json").read_text())["summary"],
            json.loads((run_dir / "quality.json").read_text())["summary"])


def ledger() -> list[dict]:
    p = HERE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(row: dict) -> None:
    row["ts"] = now()
    with open(HERE / "ledger.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def baseline() -> None:
    """冠军跑两遍（A/B），量单遍波动，冻结门槛。门槛全部由这两遍实测得出，写进 manifest 后不再改。"""
    m = man()
    if m.get("thresholds"):
        return
    champ = json.loads((HERE / "champion/champion.json").read_text())
    res = {}
    for tag in ("A", "B"):
        d = HERE / "runs" / f"champion-{tag}"
        if (d / "quality.json").exists():
            res[tag] = (json.loads((d / "stats.json").read_text())["summary"], json.loads((d / "quality.json").read_text())["summary"])
        else:
            log("baseline", tag)
            res[tag] = run_one(champ, d, HERE / "runs/champion-A" if tag == "B" else None)
            append({"id": f"champion-{tag}", "category": "基线", "build": champ["build"], "knobs": champ["knobs"],
                    "summary": res[tag][0], "quality": res[tag][1], "verdict": "baseline", "cost_usd": 0})
    (sa, qa), (sb, _) = res["A"], res["B"]
    qrows = json.loads((HERE / "runs/champion-A/quality.json").read_text())["rows"]
    per_seed: dict[str, list[float]] = {}
    for r in qrows:
        per_seed.setdefault(r["wav"].rsplit("_s", 1)[1], []).append(r["char_acc"])
    means = [sum(v) / len(v) for v in per_seed.values()]
    spread = max(means) - min(means)
    th = {
        "rule": "晋级 = 音质两条都过 且 至少一项指标比冠军好出单遍波动 且 没有一项差出单遍波动（交接：快过单遍波动才晋级）",
        "noise": {k: abs(sa[k] - sb[k]) for k in METRICS},  # 单遍波动：冠军同配置两遍之差（实测）
        "seed_acc_spread": spread,  # 冠军两个种子各自字准确率均值之差（实测）
        "char_acc_mean_min": qa["char_acc_mean"] - spread,  # 音质一：不低于冠军减去换种子的差
        "mel_db_mean_max": qa["self_seed_mel_db_mean"],  # 音质二：与冠军同句同种子的距离不超过冠军自己换种子的距离
        "quality_ref_run": str(HERE / "runs/champion-A"),
        "written_at": now(),
    }
    m["thresholds"] = th
    (HERE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    champ.update(run_dir=str(HERE / "runs/champion-A"), summary=sa)
    (HERE / "champion/champion.json").write_text(json.dumps(champ, ensure_ascii=False, indent=1))
    log("thresholds frozen", json.dumps(th, ensure_ascii=False))


def judge(s: dict, q: dict) -> tuple[str, str]:
    th = man()["thresholds"]
    cs = json.loads((HERE / "champion/champion.json").read_text())["summary"]
    if q["char_acc_mean"] < th["char_acc_mean_min"]:
        return "reject", f"字准确率 {q['char_acc_mean']:.3f} < 门槛 {th['char_acc_mean_min']:.3f}"
    if q.get("mel_db_mean", 0) > th["mel_db_mean_max"]:
        return "reject", f"梅尔谱距离 {q['mel_db_mean']} dB > 门槛 {th['mel_db_mean_max']}"
    better, worse = [], []
    for k in METRICS:
        d = s[k] - cs[k]
        if d < -th["noise"][k]:
            better.append(f"{k} {cs[k]}→{s[k]}")
        elif d > th["noise"][k]:
            worse.append(f"{k} {cs[k]}→{s[k]}")
    if worse:
        return "reject", "变差：" + "；".join(worse) + ("；变好：" + "；".join(better) if better else "")
    if not better:
        return "inconclusive", "没有一项好出单遍波动"
    return "better", "；".join(better)


def evaluate(cand: dict, round_tag: str) -> None:
    th = man()["thresholds"]
    run_dir = HERE / "runs" / cand["id"]
    row = {k: cand.get(k) for k in ("id", "idea", "category", "changes_audio", "build", "knobs", "key", "cost_usd")}
    row["round"] = round_tag
    row["patch"] = bool((cand.get("patch") or "").strip())
    try:
        s, q = run_one(cand, run_dir, Path(th["quality_ref_run"]))
        v, why = judge(s, q)
        row.update(summary=s, quality=q, verdict=v, reason=why)
    except Exception as e:  # noqa: BLE001  失败也入账
        row.update(verdict="error", reason=str(e)[-600:])
    append(row)
    log("verdict", cand["id"], row["verdict"], row.get("reason"))
    if row["verdict"] == "better":
        c = {k: cand[k] for k in ("id", "build", "knobs", "src")}
        c.update(run_dir=str(run_dir), summary=row["summary"], promoted_at=now())
        (HERE / "champion/champion.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
        log("PROMOTED", cand["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, required=True)
    a = ap.parse_args()
    baseline()
    if "preset-int4" not in {r["id"] for r in ledger()}:
        # 交接第 5 项：4 位量化单独报，预置一条，不经提案
        evaluate({"id": "preset-int4", "idea": "专家 4 位 g64，权重约减半、可放本机盘免 NAS 加载",
                  "category": "量化", "changes_audio": True, "build": "int4", "knobs": {},
                  "src": str(HERE / "src-base"), "key": "preset-int4", "cost_usd": 0}, "preset")
    stamp = dt.datetime.now().strftime("%m%d%H%M")
    for i in range(1, a.rounds + 1):
        cid = f"r{stamp}-{i}"
        log("propose", cid)
        r = subprocess.run([sys.executable, str(HERE / "propose.py"), cid], capture_output=True, text=True)
        print(r.stdout[-1500:], r.stderr[-800:], flush=True)
        cdir = HERE / "cands" / cid
        if r.returncode or not (cdir / "candidate.json").exists():
            rej = json.loads((cdir / "reject.json").read_text()) if (cdir / "reject.json").exists() else {}
            append({"id": cid, "round": i, "verdict": "propose_fail",
                    "reason": str(rej.get("errors") or r.stderr)[-600:], "cost_usd": rej.get("cost_usd", 0)})
            continue
        evaluate(json.loads((cdir / "candidate.json").read_text()), str(i))
    spent = sum(r.get("cost_usd") or 0 for r in ledger())
    log("LOOP DONE", f"spent_usd={spent:.4f}", (HERE / "champion/champion.json").read_text())


if __name__ == "__main__":
    main()
