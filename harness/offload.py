"""Experts on demand: run Drama 3 with the routed experts left on SSD.

Same idea as flash-moe (github.com/danveloper/flash-moe) and AiiStream Q3.6
(github.com/apirathaiya/aiistream-q3.6, Qwen3.6-35B-A3B on a 16 GB M5): the router
picks 8 of 256 experts per layer per frame; only those are read, straight out of
the stacked expert tensors in the stock safetensors shards (no repacking).

Everything that is not a routed expert stays resident. Read experts are kept in an
LRU cache with a byte budget. On a real small Mac that job is done by the OS page
cache; here the budget stands in for it, and with nocache=True every miss is read
with F_NOCACHE so the 128 GB test machine's page cache cannot help.

Byte-for-byte check: offloaded runs give the same codes as the resident model
(prefill builds a temporary [256, ...] layer so gather_qmm runs the resident shapes;
decode rows are unsorted and per-expert, so slot order does not matter).

lean_head: lm_head keeps only the rows sampling can draw (eos + the semantic band),
and token embeddings are looked up from a memory map of the shard. Both give
the same numbers as the resident tensors.
"""
from __future__ import annotations

import fcntl
import json
import os
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

F_NOCACHE = 48
PROF = bool(os.environ.get("D3_PROF"))
PROJ = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")
NP_DTYPE = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def _headers(model_dir: Path) -> dict[str, tuple[Path, int, list[int], str]]:
    """tensor name -> (shard, absolute byte offset, shape, dtype)."""
    out = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with shard.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        for k, v in head.items():
            if k != "__metadata__":
                out[k] = (shard, 8 + n + v["data_offsets"][0], v["shape"], v["dtype"])
    return out


class ExpertReader:
    """Reads one expert's 9 tensors (gate/up/down x weight/scales/biases) out of the
    stacked tensors in the shards with pread; F_NOCACHE keeps the page cache out."""

    def __init__(self, headers, layers: list[int], nocache: bool, threads: int = 8):
        self.fds, self.meta = {}, {}
        for L in layers:
            for p in PROJ:
                for part in PARTS:
                    shard, off, shape, dt = headers[f"backbone.layers.{L}.mlp.experts.{p}.{part}"]
                    if shard not in self.fds:
                        fd = os.open(shard, os.O_RDONLY)
                        if nocache:
                            fcntl.fcntl(fd, F_NOCACHE, 1)
                        self.fds[shard] = fd
                    per = int(np.prod(shape[1:])) * np.dtype(NP_DTYPE[dt]).itemsize
                    self.meta[(L, PROJ.index(p) * 3 + PARTS.index(part))] = (
                        self.fds[shard], off, per, tuple(shape[1:]), dt)
        self.pool = ThreadPoolExecutor(threads)
        self.hits = self.misses = 0
        self.read_bytes, self.read_s = 0, 0.0
        self.prefetched = self.prefetch_hits = 0
        self.prof = {"eval_ids": 0.0, "fill": 0.0, "calls": 0}
        self.trace: list[tuple[int, tuple[int, ...]]] = []  # (layer, experts) per call

    def _read(self, L: int, e: int) -> list[np.ndarray]:
        out = []
        for i in range(9):
            fd, base, per, shape, dt = self.meta[(L, i)]
            buf = os.pread(fd, per, base + e * per)
            out.append(np.frombuffer(buf, dtype=NP_DTYPE[dt]).reshape(shape))
        return out

    def prefetch(self, L: int, e: int):
        self.prefetched += 1
        return self.pool.submit(self._read, L, e)

    def read(self, L: int, experts: list[int], pending: dict | None = None) -> list[mx.array]:
        """9 arrays [len(experts), ...]. Worker threads only read bytes; MLX arrays
        are made on the caller's thread (MLX streams are per thread). pending holds
        reads already started by the previous layer's prefetch."""
        t = time.perf_counter()
        pending = pending or {}
        self.prefetch_hits += sum(1 for e in experts if e in pending)
        futs = [pending.pop(e) if e in pending else self.pool.submit(self._read, L, e)
                for e in experts]
        raws = [f.result() for f in futs]
        out = []
        for i in range(9):
            m = mx.array(np.stack([r[i] for r in raws]))
            out.append(m.view(mx.bfloat16) if self.meta[(L, i)][4] == "BF16" else m)
        self.read_bytes += sum(a.nbytes for r in raws for a in r)
        self.read_s += time.perf_counter() - t
        return out

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None,
                "read_gb": round(self.read_bytes / 2**30, 3), "read_s": round(self.read_s, 2),
                "prefetched": self.prefetched, "prefetch_hits": self.prefetch_hits,
                "prof": {k: round(v, 3) for k, v in self.prof.items()}}


class OffloadedExperts(nn.Module):
    """Drop-in for RoutedExperts with `slots` experts of the layer resident.

    The resident experts live in fixed slot tensors [slots, ...] that gather_qmm reads
    directly, so a hit copies nothing; a miss overwrites the least recently used slot.
    The router's picks must reach the CPU to find misses: one sync per layer per frame.
    Prefill rows pick far more experts than there are slots; they run on the slot
    tensors plus a temporary stack of the missing experts, and leave the slots alone.
    """

    def __init__(self, layer: int, reader: ExpertReader, slots: int, group_size: int, bits: int):
        super().__init__()
        self._L, self._r, self._S = layer, reader, slots
        self._gs, self._bits = group_size, bits
        self._pool = None  # 9 arrays [slots, ...], filled on first use
        self._slot_of = {}  # expert -> slot
        self._lru: OrderedDict = OrderedDict()  # slot -> expert, oldest first
        self._pending: dict = {}  # expert -> read started by the previous layer
        self._next = None  # (next layer's router, next OffloadedExperts) when prefetching

    def _qmm(self, x, w, s, b, ids, sort):
        return mx.gather_qmm(x, w, s, b, rhs_indices=ids, transpose=True,
                             group_size=self._gs, bits=self._bits, sorted_indices=sort)

    def _fill(self, need: list[int]) -> None:
        """Make every expert in need resident (len(need) <= slots)."""
        miss = [e for e in need if e not in self._slot_of]
        r = self._r
        r.hits += len(need) - len(miss)
        r.misses += len(miss)
        for e in need:
            if e in self._slot_of:
                self._lru.move_to_end(self._slot_of[e])
        if not miss:
            self._pending.clear()
            return
        new = r.read(self._L, miss, self._pending)
        self._pending.clear()
        if self._pool is None:  # first call: the slots start as copies of these experts
            pad = self._S - len(miss)
            self._pool = [mx.concatenate([a, mx.zeros((pad, *a.shape[1:]), a.dtype)]) if pad else a
                          for a in new]
            slots = list(range(len(miss)))
            for sl in range(len(miss), self._S):
                self._lru[sl] = None
                self._lru.move_to_end(sl, last=False)
        else:
            keep = set(need)
            slots = []
            for sl, e in self._lru.items():
                if e is None or e not in keep:
                    slots.append(sl)
                    if len(slots) == len(miss):
                        break
            idx = mx.array(slots, dtype=mx.uint32)
            for i in range(9):
                self._pool[i][idx] = new[i]
            for sl in slots:
                old = self._lru[sl]
                if old is not None:
                    del self._slot_of[old]
        for sl, e in zip(slots, miss):
            self._slot_of[e] = sl
            self._lru[sl] = e
            self._lru.move_to_end(sl)
        mx.eval(self._pool)

    def __call__(self, x: mx.array, ids: mx.array, weights: mx.array) -> mx.array:
        from mlx_speech.models.fish_drama_3.layers import (
            SORTED_EXPERT_MIN_PAIRS, swiglu, weighted_expert_sum)
        r = self._r
        t0 = time.perf_counter()
        decode = ids.size < SORTED_EXPERT_MIN_PAIRS
        if decode and self._next is not None:
            # AiiStream's guess: the next layer's router applied to this layer's input
            gate, nxt = self._next
            pred = mx.argpartition(-gate(x), kth=ids.shape[-1] - 1, axis=-1)[:, : ids.shape[-1]]
            mx.eval(ids, pred)
            for e in np.unique(np.array(pred)).tolist():
                if e not in nxt._slot_of and e not in nxt._pending:
                    nxt._pending[e] = r.prefetch(nxt._L, e)
        else:
            mx.eval(ids)
        t1 = time.perf_counter()
        ids_np = np.array(ids)
        uniq = np.unique(ids_np).tolist()
        r.trace.append((self._L, tuple(uniq)))
        if len(uniq) <= self._S and decode:
            self._fill(uniq)
            tensors = self._pool
            lut = np.zeros(256, np.uint32)
            for e in uniq:
                lut[e] = self._slot_of[e]
        else:  # prefill: one temporary [256, ...] layer, each routed expert at its own
            # index (gather_qmm then runs the resident shapes, so rounding is identical);
            # unused rows repeat a routed one; the slots are left alone.
            have = [e for e in uniq if e in self._slot_of]
            miss = [e for e in uniq if e not in self._slot_of]
            r.hits += len(have); r.misses += len(miss)
            new = r.read(self._L, miss)
            n0 = self._S if self._pool is not None else 0
            src = {e: self._slot_of[e] for e in have}
            src.update({e: n0 + j for j, e in enumerate(miss)})
            perm = mx.array([src.get(e, src[uniq[0]]) for e in range(256)], dtype=mx.uint32)
            tensors = []
            for i in range(9):
                both = mx.concatenate([self._pool[i], new[i]]) if self._pool is not None else new[i]
                tensors.append(both[perm])
                mx.eval(tensors[-1])  # free each concatenation before the next
            del new
            lut = np.arange(256, dtype=np.uint32)
        local = mx.array(lut[ids_np])
        r.prof["eval_ids"] += t1 - t0; r.prof["fill"] += time.perf_counter() - t1; r.prof["calls"] += 1
        rows, top_k = local.shape
        expanded = x[:, None, None, :]
        sort = local.size >= SORTED_EXPERT_MIN_PAIRS
        lid = local
        if sort:
            flat = local.flatten()
            order = mx.argsort(flat)
            expanded = expanded.flatten(0, -3)[order // top_k]
            lid = flat[order]
        g = self._qmm(expanded, *tensors[0:3], lid, sort)
        u = self._qmm(expanded, *tensors[3:6], lid, sort)
        out = self._qmm(swiglu(g, u), *tensors[6:9], lid, sort)
        if sort:
            out = out[mx.argsort(order)].reshape(rows, top_k, 1, -1)
        return weighted_expert_sum(out.squeeze(-2), weights, x)


class MappedEmbedding(nn.Module):
    """nn.Embedding read from a memory map of the shard instead of resident."""

    def __init__(self, shard: Path, offset: int, shape, dt: str):
        super().__init__()
        self._mm = np.memmap(shard, dtype=NP_DTYPE[dt], mode="r", offset=offset, shape=tuple(shape))
        self._bf16 = dt == "BF16"

    def __call__(self, ids: mx.array) -> mx.array:
        rows = mx.array(np.ascontiguousarray(self._mm[np.array(ids)]))
        return rows.view(mx.bfloat16) if self._bf16 else rows


def load_offloaded(model_dir, *, pin_layers: int, slots: int, prefetch: bool = False,
                   nocache: bool = True,
                   lean_head: bool = True, dense_int8: bool = False, codec_tile: int = 64):
    from mlx_speech.checkpoints.sharded import load_state_dict
    from mlx_speech.generation.fish_drama_3 import FishDrama3Runtime, Tokenizer
    from mlx_speech.models.fish_drama_3.config import FishDrama3Config
    from mlx_speech.models.fish_drama_3.model import (
        DENSE_INT8_GROUP_SIZE, FishDrama3Model, is_dense_int8_target)
    from mlx_speech.models.fish_s2_pro.codec import FishS2Codec

    model_dir = Path(model_dir)
    config = FishDrama3Config.from_path(model_dir)
    q = config.expert_quantization
    model = FishDrama3Model(config)
    heads = _headers(model_dir)
    n_layers = config.text.num_hidden_layers
    off_layers = list(range(pin_layers, n_layers))
    store = ExpertReader(heads, off_layers, nocache)
    bb = model.backbone
    for L in off_layers:
        bb.layers[L].mlp.experts = OffloadedExperts(L, store, slots, q.group_size, q.bits)
    if prefetch:
        for L in off_layers[:-1]:
            bb.layers[L].mlp.experts._next = (bb.layers[L + 1].mlp.gate, bb.layers[L + 1].mlp.experts)
    off_prefix = tuple(f"backbone.layers.{L}.mlp.experts." for L in off_layers)
    skip = lambda k: k.startswith(off_prefix) or (lean_head and k in (
        "backbone.embed_tokens.weight", "backbone.lm_head.weight"))
    weights = load_state_dict(model_dir).weights  # mx.load is lazy: skipped tensors are never read
    if lean_head:
        cand = np.array(model.audio_decoder.main_candidates)
        shard, off, shape, dt = heads["backbone.lm_head.weight"]
        mm = np.memmap(shard, dtype=NP_DTYPE[dt], mode="r", offset=off, shape=tuple(shape))
        w_c = mx.array(np.ascontiguousarray(mm[cand]))
        w_c = w_c.view(mx.bfloat16) if dt == "BF16" else w_c
        del mm
        bb.lm_head = nn.Linear(1, 1, bias=False)  # placeholder, never called
        cand_mx = model.audio_decoder.main_candidates
        full_logits = type(bb).logits

        def logits(hidden, token_ids, _w=w_c.T):
            if token_ids is cand_mx:
                return hidden @ _w
            raise RuntimeError("lean_head only serves main_candidates")
        bb.logits = logits
        shard, off, shape, dt = heads["backbone.embed_tokens.weight"]
        bb.embed_tokens = MappedEmbedding(shard, off, shape, dt)
    kept = [(k, v) for k, v in weights.items() if not skip(k)]
    model.load_weights(kept, strict=False)
    have = {k for k, _ in kept}
    want = {k for k, _ in nn.utils.tree_flatten(model.parameters())}
    missing = want - have - ({"backbone.lm_head.weight"} if lean_head else set())
    if missing:
        raise ValueError(f"weights not loaded: {sorted(missing)[:5]}")
    del weights, kept
    if dense_int8:
        nn.quantize(model, group_size=DENSE_INT8_GROUP_SIZE, bits=8, mode="affine",
                    class_predicate=is_dense_int8_target)
    mx.eval(model.parameters())
    # The codec decodes in time tiles with a halo (exact, see its Decoder): 64-step
    # tiles give the same samples as 256 and cut the decode peak from 3.5 to 1.8 GB.
    from mlx_speech.models.fish_s2_pro import codec_model
    codec_model.Decoder.TILE_STEPS = codec_tile
    codec = FishS2Codec.from_dir(model_dir / "codec-mlx")
    rt = FishDrama3Runtime(model=model,
                           tokenizer=Tokenizer.from_file(str(model_dir / "tokenizer.json")),
                           codec=codec)
    return rt, store
