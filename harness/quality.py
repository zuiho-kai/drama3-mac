#!/usr/bin/env python3
"""音质门槛：转写对字（mlx_whisper large-v3-turbo）+ 与冠军同句同种子输出的客观距离（长时平均对数梅尔谱，dB）。

用法: python -I quality.py（装了 mlx-whisper、zhconv 的环境） <run_dir> [<champion_run_dir>]
写 <run_dir>/quality.json：每条 wav 的转写、字准确率、与冠军的梅尔谱距离、是否与冠军逐字节相同。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import mlx_whisper
import numpy as np
import wave
import zhconv

HERE = Path(__file__).resolve().parent
import os
# D3_MANIFEST 可换台词组（如 stress.json），默认冻结的 manifest.json
MAN = json.loads((HERE / os.environ.get("D3_MANIFEST", "manifest.json")).read_text())
PUNCT = re.compile(r"[\s，。、！？…～~,.!?：:；;“”\"'（）()\-—]")
# 来源：沿用 Drama 3 codec 每帧 2048 个采样（ref/mlx-speech/docs/fish-drama-3.md 第 17 行），跳步取四分之一帧
N_FFT, HOP = 2048, 2048 // 4
# 来源：沿用 Whisper 的 80 个梅尔带
N_MEL = 80


def norm(t: str) -> str:
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"<[^>]*>", "", t)  # 成对标签 <whisper> <fast> 等不读出来，不算字
    return PUNCT.sub("", zhconv.convert(t, "zh-cn"))


def char_acc(ref: str, hyp: str) -> float:
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev, d[j] = d[j], cur
    return max(0.0, 1 - d[len(b)] / max(1, len(a)))


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dtype = {2: np.int16, 4: np.int32}[sw]
    y = np.frombuffer(raw, dtype=dtype).astype(np.float32) / np.iinfo(dtype).max
    return (y.reshape(-1, ch).mean(1) if ch > 1 else y), sr


def mel_ltas(path: Path) -> np.ndarray:
    """长时平均对数梅尔谱（dB）：与说话内容和节奏无关，反映音色与频响。"""
    y, sr = read_wav(path)
    win = np.hanning(N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    energy = spec.sum(1)
    spec = spec[energy > energy.max() * 1e-4]  # 来源：-40 dB 以下当静音帧去掉（自定，只影响去静音，不当门槛）
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731  来源：标准 HTK 梅尔公式
    edges = 700 * (10 ** (np.linspace(mel(0), mel(sr / 2), N_MEL + 2) / 2595) - 1)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    fb = np.zeros((N_MEL, len(freqs)), np.float32)
    for k in range(N_MEL):
        lo, c, hi = edges[k], edges[k + 1], edges[k + 2]
        fb[k] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    ltas = 10 * np.log10((spec @ fb.T).mean(0) + 1e-10)
    return ltas - ltas.mean()  # 去掉整体音量差


def dist(a: Path, b: Path) -> float:
    return float(np.sqrt(np.mean((mel_ltas(a) - mel_ltas(b)) ** 2)))


def main() -> None:
    run = Path(sys.argv[1])
    champ = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    texts = dict(MAN["lines"])
    rows = []
    for wav in sorted((run / "wav").glob("*.wav")):
        name = wav.stem.rsplit("_s", 1)[0]
        hyp = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     language="zh", condition_on_previous_text=False)["text"]
        row = {"wav": wav.name, "asr": hyp, "char_acc": round(char_acc(texts[name], hyp), 3),
               "sha": hashlib.sha256(wav.read_bytes()).hexdigest()[:16]}
        cw = champ / "wav" / wav.name if champ else None
        if cw and cw.exists():
            row["mel_db"] = round(dist(wav, cw), 3)
            row["same_as_champion"] = hashlib.sha256(cw.read_bytes()).hexdigest()[:16] == row["sha"]
            row["dur_ratio"] = round(len(read_wav(wav)[0]) / len(read_wav(cw)[0]), 3)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    acc = [r["char_acc"] for r in rows]
    summ = {"char_acc_mean": round(float(np.mean(acc)), 4), "char_acc_min": min(acc)}
    mels = [r["mel_db"] for r in rows if "mel_db" in r]
    if mels:
        summ.update(mel_db_mean=round(float(np.mean(mels)), 3), mel_db_max=max(mels),
                    bit_exact=all(r.get("same_as_champion") for r in rows))
    # 同句两个种子之间的距离 = 换一次采样的自然波动，冠军这一项用来定门槛
    s = MAN["seeds"]
    summ["self_seed_mel_db_mean"] = round(float(np.mean(
        [dist(run / "wav" / f"{n}_s{s[0]}.wav", run / "wav" / f"{n}_s{s[1]}.wav") for n, _ in MAN["lines"]
         if (run / "wav" / f"{n}_s{s[1]}.wav").exists()])), 3)
    (run / "quality.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1))
    print("QSUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
