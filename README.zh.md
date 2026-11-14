<h1 align="center">drama3-mac</h1>

<p align="center">
  <strong>在 Mac 上跑 Fish Audio Drama 3（35B MoE 会演戏的 TTS），带一条自己越跑越省的搜索循环。</strong><br>
  专家 4 位版整模只占 <b>25.5 GB</b> 盘、峰值 <b>29.5 GB</b> 内存，从本机 SSD 冷加载 <b>3.3 秒</b>，
  说话速度约是<b>实时的 2.2 倍</b>。
</p>

<p align="center">
  <a href="#为什么用-drama3-mac">为什么</a> ·
  <a href="#下载现成权重">下载</a> ·
  <a href="#性能">性能</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#按内存选配置">按内存选配置</a> ·
  <a href="#极端台词">极端台词</a> ·
  <a href="#用-inferrsi-打磨">InferRSI</a> ·
  <a href="#路线图">路线图</a> ·
  <a href="README.md">English</a>
</p>

## 为什么用 drama3-mac

- **48 GB 的 Mac 就能跑。** 官方 Drama 3 要 48 GB 以上的 NVIDIA 显卡；四位版峰值只占 **29.5 GB** 统一内存。
- **下载量少一半。** 权重 **25.5 GB**，不是 41 GB，而且已经转好——[直接从 Release 下](#下载现成权重)，不用自己转换。
- **3.3 秒加载，2.2 倍实时。** M5 Max 上从本机 SSD 冷加载，实时率 0.45。
- **声音不打折。** 对字率 0.974，和 8 位版一样；极端台词上，换四位对频谱的影响比换个随机种子还小。[自己听](#样音)。
- **数字都是真跑的。** 冻结的测速工具加 InferRSI 循环：下面每个数都来自真实运行，每次尝试都记进账本。

Built with Fish Audio.

## 下载现成权重

四位版挂在 Release [`v0.1.0-int4`](https://github.com/zuiho-kai/drama3-mac/releases/tag/v0.1.0-int4)，
是一个 tar 切成的 1.9 GB 分片（GitHub 单个 Release 文件上限 2 GB）。合并时要约 55 GB 空闲，合完可删分片。

```sh
gh release download v0.1.0-int4 -R zuiho-kai/drama3-mac -D dl
(cd dl && shasum -a 256 -c SHA256SUMS)
mkdir -p harness/models && cat dl/drama3-mac-int4-g64.tar.part-* | tar -xf - -C harness/models
# → harness/models/int4-g64（模型、codec-mlx/、LICENSE、NOTICE）
```

## 样音

同一句台词、同一种子、同一参考音；专家 8 位对比专家 4 位。

| 台词 | 8 位 | 4 位 |
| --- | --- | --- |
| 撒娇 `[撒娇，拖长尾音]` | [int8-coaxing.mp3](samples/int8-coaxing.mp3) | [int4-coaxing.mp3](samples/int4-coaxing.mp3) |
| 委屈 `[委屈]` | [int8-hurt.mp3](samples/int8-hurt.mp3) | [int4-hurt.mp3](samples/int4-hurt.mp3) |

## 简介

[Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) 是会演戏的 TTS：在方括号里写一句说戏，
比如 `[撒娇，拖长尾音]`、`[害怕地小声说]`，克隆出来的声音就照着演。官方面向 48 GB 以上的 N 卡。

**drama3-mac** 让它在苹果芯片上跑起来。底子是开源 [mlx-speech](https://github.com/appautomaton/mlx-speech)
里现成的 Drama 3 MLX 移植，在上面加了：

- **专家 4 位版**：权重减半，整个模型放得进本机 SSD；
- **冻结的测速工具**：台词、种子、参考音固定不变；记加载时间、内存峰值、每句耗时（拆成逐帧生成和声码器两段），
  音质用转写对字率和与冠军输出的梅尔谱距离；
- **无人值守的搜索循环**，照 [InferRSI](https://github.com/InferMatrixLab/InferRSI) 的形状：模型出一个改法，
  工具测，裁判决定晋级还是淘汰，每次尝试连同花费记进账本；
- **极端台词组**：30 秒长独白、绕口令、耳语、尖叫、嚎啕大哭、中英夹杂、数字和生僻字、长停顿，用来对比不同版本。

## 动态

- **2026-11-14：** 专家 4 位版：盘上 25.5 GB、峰值 29.5 GB，从 SSD 冷加载 **3.3 秒**（开跑前用 `mincore` 核过页缓存为空）。
- **2026-11-14：** 第一轮 InferRSI：模型提了 3 个方案，花 **0.55 美元**。冠军开了预热并把权重钉在内存里：
  首句 **3.65 秒 → 2.61 秒**，输出逐字节不变。

## 性能

<!-- latest-performance:start -->
测试机：**Apple M5 Max，128 GB，macOS 27**，mlx-speech `2ecabcd`。每一行跑同一组冻结台词：
6 句普通话（撒娇、傲娇、委屈、天气播报、说明、问候），种子 1234 和 5678，同一段参考音。

- **加载（冷）**：页缓存里没有权重时的读入时间。
- **首句**：新进程加载完的第一句。
- **速度**：每秒音频要算几秒（实时率），取第 2–12 句平均。
- **峰值**：合成时 MLX 的内存峰值。

| 版本 | 盘上大小 | 峰值 | 加载（冷） | 首句 | 速度（实时率） | 对字率 |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-speech 默认，专家 8 位 | 41 GB | 44.4 GB | 509 秒 ¹ | 3.65 秒 | 0.489 | 0.974 |
| **+ 预热 + 权重常驻**（InferRSI 冠军） | 41 GB | **42.5 GB** | — | **2.61 秒** | **0.463** | 0.974 |
| **专家 4 位** | **25.5 GB** | **29.5 GB** | **3.3 秒** | 3.21 秒 | **0.453** | 0.974 |
<!-- latest-performance:end -->

¹ 测试机上 8 位权重放在网络盘（SMB）上读。

对字率用 Whisper large-v3-turbo 转写后对照台词，去掉标点和方括号说戏。每次运行的原始数字在 [`results/`](results/)。

## 快速开始

需要苹果芯片的 macOS、`git` 和 [`uv`](https://docs.astral.sh/uv/)。

```sh
git clone https://github.com/zuiho-kai/drama3-mac && cd drama3-mac
scripts/setup.sh
```

用[现成的四位权重](#下载现成权重)，或下载官方权重，转换一次（FP8 专家 → MLX，不需要 PyTorch）：

```sh
.venv/bin/hf download fishaudio/drama-3 --local-dir raw/drama-3

# 专家 4 位（25.5 GB）
.venv/bin/python harness/convert4.py --input-dir raw/drama-3 \
    --output-dir harness/models/mlx-int4 --expert-bits 4 --group-size 64

# 专家 8 位（41 GB，mlx-speech 默认）
.venv/bin/python vendor/mlx-speech/scripts/convert/fish_drama_3.py \
    --input-dir raw/drama-3 --output-dir harness/models/mlx-int8
```

`convert4.py` 就是 mlx-speech 的转换器，只改一处：每个张量进分片时先算出来，存分片时不会变成一条超长的 GPU 命令（原来会触发 Metal 超时）。

说一句：

```sh
.venv/bin/python vendor/mlx-speech/scripts/generate/fish_drama_3.py \
    --model-dir harness/models/mlx-int4 \
    --text "[撒娇，拖长尾音，带点鼻音] 哎呀，你就再陪我玩一会儿嘛～" \
    --reference-audio voice.wav --reference-text "voice.wav 的原文。" \
    -o out.wav
```

## 按内存选配置

| Mac 内存 | 版本 | 峰值内存 | 速度（实时率） |
| --- | --- | --- | --- |
| **48 GB** | 专家 4 位 | **29.5 GB** | 0.45 |
| **64 GB** | 专家 8 位 | **44.4 GB** | 0.49 |
| **96 GB 及以上** | 专家 8 位，预热 + 权重常驻 | **42.5 GB** | 0.46，首句 2.6 秒 |

权重常驻就是加载前调一次 `mx.set_wired_limit(56 * 2**30)`，加载后先合成一句短句预热；测速工具里这两项是开关（`wired_limit_gb`、`warmup`）。

## 跑测速工具

把参考音放到 `harness/refs/reference.wav`，原文写进 `harness/manifest.json` 的 `ref_text`，然后：

```sh
cd harness
echo '{"id":"int4","build":"int4","knobs":{"warmup":true,"wired_limit_gb":56},"src":"src-base"}' > int4.json
../.venv/bin/python -I bench.py int4.json runs/int4            # 加载、峰值、每句耗时
../.venv/bin/python -I quality.py runs/int4 runs/champion-A    # 对字率、梅尔谱距离
```

搜索循环（出方案要 `claude` 命令行，花费从它的输出里读并入账）：

```sh
D3_PY=../.venv/bin/python D3_ASR_PY=../.venv/bin/python python3 -u loop.py --rounds 3
```

测冷加载：`coldify.py <目录>` 用 `F_NOCACHE` 把权重目录重写一遍，页缓存里不留；`resident.py <目录>` 打印还有多少在缓存里。

## 极端台词

<!-- stress:start -->
`harness/stress.json` 里是八句专门拉开版本差距的台词：31 秒长独白、全速绕口令、耳语逃跑计划、见到蟑螂的尖叫、
嚎啕大哭、程序员中英夹杂吐槽、带日期圆周率和生僻字的新闻播报、带长停顿的告白。用 `D3_MANIFEST=stress.json` 跑；
`blind.py <甲> <乙>` 把两次运行做成 A/B 盲听包，答案单独一个文件。

极端台词上 8 位对 4 位，同种子（1234 和 5678）、同参考音：

| | 专家 8 位 | 专家 4 位 |
| --- | --- | --- |
| 31 秒长独白 | 14.4 秒 | **13.9 秒** |
| 速度（实时率），第 2–16 句 | 0.459 | **0.444** |
| 峰值内存 | 44.5 GB | **29.4 GB** |
| 对字率 | 0.956 | 0.954 |
| 与另一版本的梅尔谱距离 | — | 2.38 dB |
| 同一版本种子 1234 对 5678 的梅尔谱距离 | 3.35 dB | 3.19 dB |

整组来看，换成 4 位对频谱的改变比换一次随机种子还小。
<!-- stress:end -->

## 用 InferRSI 打磨

drama3-mac 是 [**InferRSI**](https://github.com/InferMatrixLab/InferRSI) 的示范项目。InferRSI 是推理引擎递归自我改进的工作台：
提一个改法，对着冻结的基线测，过了门槛才留下。

- **门槛开跑前定死。** 冠军先跑两遍，两遍之差就是噪声线；音质门槛（对字率、梅尔谱距离）在任何候选开跑前写进 manifest。
- **一次只改一件。** 模型读冠军的分段耗时（每句 87% 在逐帧生成，10% 在声码器）和以往尝试的账本，只交一组开关或一个针对
  mlx-speech 源码的补丁；格式不对或重复的方案开跑前就退回。
- **每次尝试都记账。** 成功、失败和每个方案花了多少钱都进 [`results/ledger.jsonl`](results/ledger.jsonl)。第一轮：3 个方案，0.55 美元，两次晋级。

## 路线图

- **更多搜索轮次**：逐帧异步流水线、逐帧循环 `mx.compile`、按层选 4/6 位混合专家。
- **专家按需加载**：Drama 3 每个词只激活 35B 里的 3B，冷门专家从 SSD 流式读。
- **48 GB → 32 GB**：专家 4 位之外，稠密投影也压到 4 位。
- **常驻服务**，加一行安装。

## 致谢与许可

这里的测速工具和脚本用 MIT 许可，见 [LICENSE](LICENSE)。Drama 3 的 MLX 移植、转换器和合成脚本来自
[mlx-speech](https://github.com/appautomaton/mlx-speech)（MIT），见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
git 仓库里不放模型权重。Release 里的四位权重是 Fish Audio Drama 3 的衍生作品，按 Fish Audio Community
License v1.0 分发（许可证和 NOTICE 在文件夹里，NOTICE 写明了怎么改的）。Built with Fish Audio。原版权重：
[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3)。
