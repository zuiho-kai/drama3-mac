#!/usr/bin/env python3
"""极端台词盲听包：每句甲/乙同种子各一条，随机标 A/B，文件名不写版本；答案和客观数另写。
用法: python -I blind.py <版本甲运行目录> <版本乙运行目录>（如 runs/stress-int8 runs/stress-int4）
输出: blind/listen/<序号>_<台词名>_A.mp3|_B.mp3（给听的人）、blind/answer.md（答案+每句客观数，另交）；需要 ffmpeg
"""
import json, os, random, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ["D3_MANIFEST"] = "stress.json"
sys.path.insert(0, str(HERE))
import quality  # noqa: E402  复用同一个梅尔谱距离
SEED = 1234                      # 盲听用的种子（两个版本同种子）
R8, R4 = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()  # 甲 = 参照版本，乙 = 候选版本
out = HERE / "blind"; lis = out / "listen"
lis.mkdir(parents=True, exist_ok=True)
for f in lis.glob("*.mp3"): f.unlink()
q8 = {r["wav"]: r for r in json.loads((R8 / "quality.json").read_text())["rows"]}
q4 = {r["wav"]: r for r in json.loads((R4 / "quality.json").read_text())["rows"]}
rng = random.Random(20261114)    # 打乱用的随机数种子，写进答案方便复查
rows = []
for i, (name, _) in enumerate(quality.MAN["lines"], 1):
    w = f"{name}_s{SEED}.wav"
    a_is_4 = rng.random() < 0.5
    pair = {"A": R4 if a_is_4 else R8, "B": R8 if a_is_4 else R4}
    for tag, run in pair.items():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(run / "wav" / w), "-b:a", "160k",
                        str(lis / f"{i:02d}_{name}_{tag}.mp3")], check=True)
    noise = quality.dist(R8 / "wav" / f"{name}_s1234.wav", R8 / "wav" / f"{name}_s5678.wav")
    mel = [q4[f"{name}_s{s}.wav"]["mel_db"] for s in (1234, 5678)]
    rows.append(dict(i=i, name=name, four="A" if a_is_4 else "B", mel=mel, noise=round(noise, 2),
                     acc8=q8[w]["char_acc"], acc4=q4[w]["char_acc"], dur=q4[w]["dur_ratio"]))
top = sorted(rows, key=lambda r: -sum(r["mel"]))[:3]
L = ["# 极端台词盲听答案（先别看，听完再对）", "",
     f"打乱用随机数种子 20261114；两个版本都用合成种子 {SEED}、同一参考音、同一采样参数。", "",
     "| 序号 | 台词 | 乙是 | 甲是 | 乙与甲梅尔距离 dB（种子1234 / 5678） | 甲自己换种子的距离 dB | 对字率 甲 / 乙 | 时长比 乙/甲 |",
     "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    L.append(f"| {r['i']:02d} | {r['name']} | {r['four']} | {'B' if r['four']=='A' else 'A'} | {r['mel'][0]:.2f} / {r['mel'][1]:.2f} | {r['noise']:.2f} | {r['acc8']:.3f} / {r['acc4']:.3f} | {r['dur']:.3f} |")
L += ["", "差最大的三句（两种子梅尔距离之和）：" + "、".join(f"{r['i']:02d} {r['name']}" for r in top),
      "", "梅尔距离 = 长时平均对数梅尔谱的均方根差（去掉整体音量），反映音色和频响。「甲自己换种子的距离」是同一版本换一次随机种子的自然差别，拿来当尺子：乙与甲的距离比它小的句子，换版本的差别不比换种子大；比它大的句子最可能听得出来。",
      "比尺子大的句子：" + "、".join(f"{r['i']:02d} {r['name']}（{sum(r['mel'])/2:.2f} 对 {r['noise']:.2f}）" for r in rows if sum(r['mel'])/2 > r['noise'])]
(out / "answer.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
