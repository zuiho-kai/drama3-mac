#!/usr/bin/env python3
"""4 位专家转换：调 mlx-speech 现成转换器，只把每个张量在加入分片时先 eval，
避免整片一次提交导致 Metal GPU Timeout（14:21 首次转换在第一片 save 时报 kIOGPUCommandBufferCallbackErrorTimeout）。"""
import sys
from pathlib import Path
import mlx.core as mx
import os
# MLX_SPEECH_DIR = mlx-speech 源码仓库根目录（scripts/setup.sh 克隆到 ../vendor/mlx-speech）
sys.path.insert(0, str(Path(os.environ.get("MLX_SPEECH_DIR", Path(__file__).resolve().parents[1] / "vendor/mlx-speech")) / "scripts/convert"))
import fish_drama_3 as conv  # noqa: E402
_add = conv.ShardWriter.add
def add(self, name, value):
    mx.eval(value)
    _add(self, name, value)
conv.ShardWriter.add = add
conv.main()
