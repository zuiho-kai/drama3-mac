#!/usr/bin/env python3
"""提案：让模型（claude -p，无工具）按冠军 profile 和账本出一个候选，一次只出一件。

用法: python3 propose.py <cand_id>  → 写 cands/<cand_id>/candidate.json（有补丁时另建 cands/<cand_id>/src）
格式校验：JSON 可解析、knobs/build 合法、补丁 git apply --check 通过、不与账本里已试过的重复；不过只带错误重试一次（H3 教训）。
花费从 claude 输出的 total_cost_usd 取，写进 candidate.json 的 cost_usd。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAN = json.loads((HERE / "manifest.json").read_text())
KNOB_DOC = {
    "dense_int8": "bool，加载后把稠密 bf16 投影打包成 int8（上游自带开关，改数值，走音质线）",
    "wired_limit_gb": "float，mx.set_wired_limit，把权重钉在显存里",
    "cache_limit_gb": "float，mx.set_cache_limit，MLX 释放后缓存的上限（影响峰值）",
    "clear_cache_after_line": "bool，每句合成后 mx.clear_cache()",
    "warmup": "bool，正式台词前先合成一句短句预热（预热耗时单独记，不计入首句）",
}
SRC_FILES = ["mlx_speech/generation/fish_drama_3.py", "mlx_speech/models/fish_drama_3/model.py",
             "mlx_speech/models/fish_drama_3/checkpoint.py", "mlx_speech/models/fish_drama_3/backbone.py"]


def ledger() -> list[dict]:
    p = HERE / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def key(build: str, knobs: dict, patch: str) -> str:
    return hashlib.sha256(json.dumps([build, knobs, patch.strip()], sort_keys=True).encode()).hexdigest()[:12]


def prompt(err: str | None) -> str:
    champ = json.loads((HERE / "champion/champion.json").read_text())
    cstats = json.loads((HERE / champ["run_dir"] / "stats.json").read_text())
    hist = [{k: r.get(k) for k in ("id", "idea", "category", "build", "knobs", "verdict", "reason", "summary")}
            for r in ledger()[-20:]]
    src = "\n\n".join(f"### {f}\n```python\n{(HERE / 'src-base' / f).read_text()}\n```" for f in SRC_FILES)
    builds = {k: v["note"] for k, v in MAN["builds"].items()}
    return f"""你是 Mac 推理优化的搜索提案器。目标：降低 Fish Drama 3（35B MoE，3B 激活，MLX 移植）在本机 M5 Max 128GB 上的开销——加载时间、内存峰值、每句合成耗时——语气和音质不能退。
本机盘剩 {shutil.disk_usage("/").free // 1024**3}G；int8 权重 41G 只能放 NAS（SMB，加载约 500 秒）。

当前冠军：{json.dumps(champ, ensure_ascii=False)}
冠军 profile（每句 gen_s 分为 frames_s=骨干+音频解码器逐帧生成，codec_s=声码器解码；load_s 为整模加载）：
{json.dumps(cstats['summary'], ensure_ascii=False)}
冠军每句：{json.dumps(cstats['lines'][:6], ensure_ascii=False)}

可选权重版本（build）：{json.dumps(builds, ensure_ascii=False)}
可用开关（knobs）：{json.dumps(KNOB_DOC, ensure_ascii=False)}
也可以给 src-base（mlx_speech 包源码）出一个 unified diff 补丁（路径以 a/mlx_speech/... b/mlx_speech/... 开头），比如加载方式、缓存、mx.compile、预分配等。

最近的尝试（同样的组合别再交）：
{json.dumps(hist, ensure_ascii=False)}

规则：
1. 一次只出一个想法。先看 profile 哪个阶段最费时/最占内存，从那里下手，IDEA 里写清假设和预计省多少。
2. 会改数值结果（量化、精度、跳计算）的标 changes_audio=true，要过音质门槛；只改加载/缓存/调度的标 false。
3. 只输出一个 JSON 对象，不要别的文字：
{{"idea": "假设 → 为什么能省", "category": "加载|内存|解码|声码器|量化|其他", "changes_audio": true/false, "build": "<build 名>", "knobs": {{...}}, "patch": "<unified diff 或空串>"}}

相关源码：
{src}
""" + (f"\n上一次输出没通过校验：{err}\n请修正后重新输出。" if err else "")


def check(c: dict, cdir: Path) -> str | None:
    for k in ("idea", "category", "changes_audio", "build", "knobs", "patch"):
        if k not in c:
            return f"缺字段 {k}"
    if c["build"] not in MAN["builds"]:
        return f"build {c['build']} 不存在"
    if not (HERE / MAN["builds"][c["build"]]["dir"] / "config.json").exists():
        return f"build {c['build']} 还没转好"
    bad = set(c["knobs"]) - set(KNOB_DOC)
    if bad:
        return f"未知开关 {bad}"
    k = key(c["build"], c["knobs"], c["patch"])
    if any(r.get("key") == k for r in ledger()):
        return "与账本里已试过的组合完全相同"
    c["key"] = k
    if c["patch"].strip():
        src = cdir / "src"
        shutil.rmtree(src, ignore_errors=True)
        shutil.copytree(HERE / "src-base", src)
        (cdir / "patch.diff").write_text(c["patch"].rstrip("\n") + "\n")
        r = subprocess.run(["git", "apply", "--check", str(cdir / "patch.diff")], cwd=src, capture_output=True, text=True)
        if r.returncode:
            return "git apply --check 失败：" + r.stderr[-800:]
        subprocess.run(["git", "apply", str(cdir / "patch.diff")], cwd=src, check=True)
        c["src"] = str(src)
    else:
        c["src"] = str(HERE / "src-base")
    return None


def main() -> None:
    cid = sys.argv[1]
    cdir = HERE / "cands" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    err, cost, tries = None, 0.0, []
    for attempt in range(2):
        r = subprocess.run(["claude", "-p", "--model", "opus", "--output-format", "json", "--tools", "",
                            "--no-session-persistence", "--max-budget-usd", "2"],
                           input=prompt(err), capture_output=True, text=True, cwd=cdir)
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            err = "claude 输出不是 JSON：" + (r.stdout + r.stderr)[-300:]
            tries.append(err)
            continue
        cost += out.get("total_cost_usd") or 0
        text = (out.get("result") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in text else text
        try:
            c = json.loads(text[text.find("{"): text.rfind("}") + 1])
            err = check(c, cdir)
        except (json.JSONDecodeError, ValueError) as e:
            err = f"结果不是合法 JSON：{e}"
        tries.append(err)
        if err is None:
            c.update(id=cid, cost_usd=round(cost, 4), attempts=attempt + 1)
            (cdir / "candidate.json").write_text(json.dumps(c, ensure_ascii=False, indent=1))
            print("PROPOSED", json.dumps({k: c[k] for k in ("id", "idea", "build", "knobs", "cost_usd")}, ensure_ascii=False))
            return
    (cdir / "reject.json").write_text(json.dumps({"id": cid, "cost_usd": round(cost, 4), "errors": tries}, ensure_ascii=False, indent=1))
    print("PROPOSE_FAIL", tries[-1])
    sys.exit(3)


if __name__ == "__main__":
    main()
