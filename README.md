<h1 align="center">drama3-mac</h1>

<p align="center">
  <strong>Fish Audio Drama 3 — the 35B-MoE expressive TTS — running on a Mac, with a self-improving speed and memory loop.</strong><br>
  4-bit experts fit the whole model in <b>25.5 GB</b> on disk and <b>29.5 GB</b> peak memory, load cold from SSD in <b>3.3 s</b>,
  and speak about <b>2.2× faster than real time</b>.
</p>

<p align="center">
  <a href="#why-drama3-mac">Why</a> ·
  <a href="#download-prebuilt-weights">Download</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#pick-a-config-for-your-mac">Pick by Memory</a> ·
  <a href="#stress-lines">Stress Lines</a> ·
  <a href="#built-with-inferrsi">InferRSI</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="README.zh.md">中文</a>
</p>

## Why drama3-mac

- **Fits a 48 GB Mac.** Upstream Drama 3 wants a 48 GB+ NVIDIA GPU; the 4-bit build peaks at **29.5 GB** of unified memory.
- **Half the download.** **25.5 GB** of weights instead of 41 GB, prebuilt — [download it from the release](#download-prebuilt-weights) and skip the conversion.
- **Loads in 3.3 s, speaks 2.2× real time.** Cold from the internal SSD on an M5 Max, RTF 0.45.
- **Same voice.** Char accuracy 0.974, identical to the 8-bit build; on the stress set 4-bit moves the spectrum less than re-rolling the seed. [Listen yourself](#samples).
- **Measured, not guessed.** A frozen harness and an InferRSI loop: every number below is a real run, every attempt is booked in a ledger.

Built with Fish Audio.

## Download Prebuilt Weights

The 4-bit build is attached to release [`v0.1.0-int4`](https://github.com/zuiho-kai/drama3-mac/releases/tag/v0.1.0-int4)
as 1.9 GB parts of one tar (GitHub caps a release file at 2 GB). Joining needs about 55 GB free; delete the parts afterwards.

```sh
gh release download v0.1.0-int4 -R zuiho-kai/drama3-mac -D dl
(cd dl && shasum -a 256 -c SHA256SUMS)
mkdir -p harness/models && cat dl/drama3-mac-int4-g64.tar.part-* | tar -xf - -C harness/models
# → harness/models/int4-g64 (model, codec-mlx/, LICENSE, NOTICE)
```

## Samples

Same line, seed and reference voice; 8-bit experts vs 4-bit experts.

| Line | 8-bit | 4-bit |
| --- | --- | --- |
| Coaxing `[撒娇，拖长尾音]` | [int8-coaxing.mp3](samples/int8-coaxing.mp3) | [int4-coaxing.mp3](samples/int4-coaxing.mp3) |
| Hurt `[委屈]` | [int8-hurt.mp3](samples/int8-hurt.mp3) | [int4-hurt.mp3](samples/int4-hurt.mp3) |

## About

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) is an acting TTS: write a
director's note in brackets — `[撒娇，拖长尾音]`, `[whispering, scared]` — and a cloned voice
performs the line. Upstream it targets 48 GB+ NVIDIA GPUs.

**drama3-mac** runs it on Apple silicon. It builds on the open-source
[mlx-speech](https://github.com/appautomaton/mlx-speech) MLX port of Drama 3 and adds:

- a **4-bit expert build** that halves the weights so the model lives on the internal SSD;
- a **frozen benchmark harness** — fixed lines, seeds and reference voice; load time, peak memory,
  per-line time split into frame generation and codec; transcription accuracy and mel-spectrum
  distance against the champion;
- an **unattended search loop** in the [InferRSI](https://github.com/InferMatrixLab/InferRSI)
  shape: a model proposes one change, the harness measures it, a judge promotes or rejects it,
  and every attempt lands in a ledger with its cost;
- a **stress set** (30-second monologue, tongue twister, whisper, scream, sobbing, code-switching,
  numbers and rare characters, long pauses) for A/B listening between builds.

## News

- **2026-11-14:** 4-bit experts: 25.5 GB on disk, 29.5 GB peak, **3.3 s** cold load from SSD
  (page cache verified empty with `mincore` before the run).
- **2026-11-14:** First InferRSI round: 3 model proposals for **$0.55**. The champion warms up and
  wires the weights: first line **3.65 s → 2.61 s**, output byte-identical.

## Performance

<!-- latest-performance:start -->
Test machine: **Apple M5 Max, 128 GB, macOS 27**, mlx-speech `2ecabcd`. Every row runs the same
frozen set: 6 Mandarin lines (coaxing, tsundere, hurt, weather report, explanation, greeting),
seeds 1234 and 5678, one reference voice.

- **Load (cold)** — weights read with nothing in the page cache.
- **First line** — first line after loading, in a fresh process.
- **Speed** — seconds of compute per second of audio (real-time factor), mean of lines 2–12.
- **Peak** — MLX peak memory while generating.

| Build | Weights on disk | Peak | Load (cold) | First line | Speed (RTF) | Char accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-speech default, 8-bit experts | 41 GB | 44.4 GB | 509 s ¹ | 3.65 s | 0.489 | 0.974 |
| **+ warmup + wired weights** (InferRSI champion) | 41 GB | **42.5 GB** | — | **2.61 s** | **0.463** | 0.974 |
| **4-bit experts** | **25.5 GB** | **29.5 GB** | **3.3 s** | 3.21 s | **0.453** | 0.974 |
<!-- latest-performance:end -->

¹ The 8-bit weights were read from a network drive (SMB) on the test machine.

Char accuracy is Whisper large-v3-turbo transcription against the script, punctuation and
bracketed directions stripped. Raw numbers for every run are in [`results/`](results/).

## Quickstart

Requires macOS on Apple silicon, `git` and [`uv`](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/zuiho-kai/drama3-mac && cd drama3-mac
scripts/setup.sh
```

Use the [prebuilt 4-bit weights](#download-prebuilt-weights), or download the official release and convert it once (FP8 experts → MLX; nothing needs PyTorch):

```sh
.venv/bin/hf download fishaudio/drama-3 --local-dir raw/drama-3

# 4-bit experts (25.5 GB)
.venv/bin/python harness/convert4.py --input-dir raw/drama-3 \
    --output-dir harness/models/mlx-int4 --expert-bits 4 --group-size 64

# 8-bit experts (41 GB, mlx-speech default)
.venv/bin/python vendor/mlx-speech/scripts/convert/fish_drama_3.py \
    --input-dir raw/drama-3 --output-dir harness/models/mlx-int8
```

`convert4.py` is the mlx-speech converter with one change: each tensor is evaluated as it enters
its shard, so saving a shard never becomes one long GPU command (which hit a Metal timeout).

Speak a line:

```sh
.venv/bin/python vendor/mlx-speech/scripts/generate/fish_drama_3.py \
    --model-dir harness/models/mlx-int4 \
    --text "[撒娇，拖长尾音，带点鼻音] 哎呀，你就再陪我玩一会儿嘛～" \
    --reference-audio voice.wav --reference-text "Transcript of voice.wav." \
    -o out.wav
```

## Pick a Config for Your Mac

| Mac memory | Build | Peak memory | Speed (RTF) |
| --- | --- | --- | --- |
| **48 GB** | 4-bit experts | **29.5 GB** | 0.45 |
| **64 GB** | 8-bit experts | **44.4 GB** | 0.49 |
| **96 GB and up** | 8-bit experts, warmup + wired weights | **42.5 GB** | 0.46, first line 2.6 s |

The wired setup is one call before loading — `mx.set_wired_limit(56 * 2**30)` — plus one short
warmup line after it; the harness exposes both as knobs (`wired_limit_gb`, `warmup`).

## Run the Harness

Put a reference clip at `harness/refs/reference.wav` and its transcript in `ref_text` of
`harness/manifest.json`, then:

```sh
cd harness
echo '{"id":"int4","build":"int4","knobs":{"warmup":true,"wired_limit_gb":56},"src":"src-base"}' > int4.json
../.venv/bin/python -I bench.py int4.json runs/int4            # load, peak, per-line time
../.venv/bin/python -I quality.py runs/int4 runs/champion-A    # char accuracy, mel distance
```

The search loop (needs the `claude` CLI for proposals; cost is read from its output and booked):

```sh
D3_PY=../.venv/bin/python D3_ASR_PY=../.venv/bin/python python3 -u loop.py --rounds 3
```

Cold-load check: `coldify.py <dir>` rewrites a weight folder with `F_NOCACHE` so nothing is left in
the page cache, and `resident.py <dir>` prints how much of it is resident.

## Stress Lines

<!-- stress:start -->
`harness/stress.json` holds eight lines built to pull builds apart: a 31-second monologue, a
tongue twister at full speed, a whispered escape plan, a cockroach scream, sobbing, a developer
complaint in Chinese and English, a news read with dates, π and rare characters, and a confession
with long pauses. Run it with `D3_MANIFEST=stress.json`; `blind.py <runA> <runB>` turns two runs into an A/B
listening pack with the answers in a separate file.

8-bit vs 4-bit on the stress set, same seeds (1234 and 5678), same reference:

| | 8-bit experts | 4-bit experts |
| --- | --- | --- |
| 31-second monologue | 14.4 s | **13.9 s** |
| Speed (RTF), lines 2–16 | 0.459 | **0.444** |
| Peak memory | 44.5 GB | **29.4 GB** |
| Char accuracy | 0.956 | 0.954 |
| Mel distance to the other build | — | 2.38 dB |
| Mel distance, same build, seed 1234 vs 5678 | 3.35 dB | 3.19 dB |

Across the set, switching to 4-bit moves the spectrum less than re-rolling the seed does.
<!-- stress:end -->

## Built with InferRSI

drama3-mac is a showcase for [**InferRSI**](https://github.com/InferMatrixLab/InferRSI), a
workbench for recursive self-improvement of inference engines: propose a change, measure it
against a frozen baseline, keep it only if it clears the gate.

- **Frozen gates, set before the run.** The champion runs twice; the spread between the two runs
  becomes the noise floor, and the quality gates (char accuracy, mel distance) are written into
  the manifest before any candidate starts.
- **One change per proposal.** A model reads the champion's profile — 87% of each line is frame
  generation, 10% the codec — and the ledger of past attempts, then returns one knob set or one
  patch against the mlx-speech source. Malformed or duplicate proposals are rejected before they
  run.
- **Every attempt booked.** Wins, losses and the dollar cost of each proposal go into
  [`results/ledger.jsonl`](results/ledger.jsonl). Round one: 3 proposals, $0.55, two promotions.

## Roadmap

- **More search rounds** — async frame pipelining, `mx.compile` on the frame loop, mixed 4/6-bit
  experts chosen per layer.
- **Experts on demand** — Drama 3 activates 3B of 35B per token; stream cold experts from SSD.
- **48 GB → 32 GB** — 4-bit dense projections on top of 4-bit experts.
- **A resident server** with a one-line installer.

## Credits and License

The harness and scripts here are MIT — see [LICENSE](LICENSE). The Drama 3 MLX port, converter and
generate script come from [mlx-speech](https://github.com/appautomaton/mlx-speech) (MIT); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights are not in the git tree. The prebuilt 4-bit weights in the release are a Derivative Work of
Fish Audio Drama 3, distributed under the Fish Audio Community License v1.0 (copy and NOTICE inside the
folder; the NOTICE says how they were modified). Built with Fish Audio. Original weights:
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3).
