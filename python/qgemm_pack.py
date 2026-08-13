#!/usr/bin/env python3
"""Host pack / unpack / reference dequant matching include/qgemm/pack.hpp.

INT4-g128 (and g64), MXFP4, NVFP4. Sequential nibble order, same E2M1 / E8M0 /
E4M3 tables and encoders as pack.hpp. Stdlib only; numpy is optional for
ndarray inputs. Do not require torch or triton.

    python3 python/qgemm_pack.py
"""

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, MutableSequence, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Group / block sizes  (pack.hpp)
# ---------------------------------------------------------------------------

INT4_GROUP_128 = 128
INT4_GROUP_64 = 64
MXFP4_BLOCK = 32
NVFP4_BLOCK = 16

# Exact finite positive E2M1 magnitudes. Index is the 3-bit magnitude code.
E2M1_POS_TABLE: Tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)

# pack.hpp: 0.7071067811865476f  (sqrt(1/2) as float32)
_E8M0_MANT_THRESH = struct.unpack("<f", struct.pack("<f", 0.7071067811865476))[0]

BytesLike = Union[bytes, bytearray, memoryview, MutableSequence[int]]
WeightIn = Union[Sequence[float], Iterable[float]]


def _f32(x: float) -> float:
    """Round through IEEE-754 binary32, matching host `float`."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def _signbit(v: float) -> bool:
    return math.copysign(1.0, v) < 0.0


def _nearbyint_to_int(x: float) -> int:
    """std::nearbyint + static_cast<int> (default ties-to-even)."""
    return int(round(x))


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def packed_nibbles_bytes(n: int, k: int) -> int:
    if (n * k) & 1:
        raise ValueError("packed_nibbles_bytes: N*K must be even")
    return (n * k) // 2


def int4_scale_count(n: int, k: int, group: int) -> int:
    if group <= 0 or k % group != 0:
        raise ValueError("int4_scale_count: K must be divisible by group")
    return n * (k // group)


def mxfp4_scale_count(n: int, k: int) -> int:
    if k % MXFP4_BLOCK != 0:
        raise ValueError("mxfp4_scale_count: K must be divisible by 32")
    return n * (k // MXFP4_BLOCK)


def nvfp4_scale_count(n: int, k: int) -> int:
    if k % NVFP4_BLOCK != 0:
        raise ValueError("nvfp4_scale_count: K must be divisible by 16")
    return n * (k // NVFP4_BLOCK)


def int4_total_bytes(n: int, k: int, group: int) -> int:
    return packed_nibbles_bytes(n, k) + int4_scale_count(n, k, group) * 2


def mxfp4_total_bytes(n: int, k: int) -> int:
    return packed_nibbles_bytes(n, k) + mxfp4_scale_count(n, k)


def nvfp4_total_bytes(n: int, k: int) -> int:
    return packed_nibbles_bytes(n, k) + nvfp4_scale_count(n, k)


def int4_bytes_per_weight(group: int) -> float:
    return (4.0 + 16.0 / float(group)) / 8.0


def mxfp4_bytes_per_weight() -> float:
    return (4.0 + 8.0 / 32.0) / 8.0  # 0.53125


def nvfp4_bytes_per_weight() -> float:
    return (4.0 + 8.0 / 16.0) / 8.0  # 0.5625


# ---------------------------------------------------------------------------
# Nibble pack  — even linear index = low nibble, odd = high nibble
# ---------------------------------------------------------------------------

def pack_nibble(data: MutableSequence[int], linear_index: int, nibble: int) -> None:
    byte_i = linear_index >> 1
    v = nibble & 0xF
    if (linear_index & 1) == 0:
        data[byte_i] = (data[byte_i] & 0xF0) | v
    else:
        data[byte_i] = (data[byte_i] & 0x0F) | (v << 4)


def unpack_nibble(data: BytesLike, linear_index: int) -> int:
    b = int(data[linear_index >> 1]) & 0xFF
    if (linear_index & 1) == 0:
        return b & 0x0F
    return (b >> 4) & 0x0F


# ---------------------------------------------------------------------------
# IEEE FP16 <-> FP32  (bit-identical to pack.hpp)
# ---------------------------------------------------------------------------

def fp16_bits_to_float(h: int) -> float:
    h = h & 0xFFFF
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            out = sign << 31
        else:
            m = mant
            e = 127 - 15 + 1
            while (m & 0x400) == 0:
                m <<= 1
                e -= 1
            m &= 0x3FF
            out = (sign << 31) | (e << 23) | (m << 13)
    elif exp == 31:
        out = (sign << 31) | 0x7F800000 | (mant << 13)
    else:
        out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
    return _bits_to_f32(out)


def float_to_fp16_bits(f: float) -> int:
    x = _f32_bits(_f32(f))
    sign = (x >> 16) & 0x8000
    exp = int((x >> 23) & 0xFF) - 127 + 15
    mant = x & 0x7FFFFF

    if ((x >> 23) & 0xFF) == 0xFF:
        m = 0x200 if mant else 0
        return (sign | 0x7C00 | m) & 0xFFFF
    if exp <= 0:
        if exp < -10:
            return sign & 0xFFFF
        mant |= 0x800000
        shift = 14 - exp
        half_m = mant >> shift
        rem = mant & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (half_m & 1)):
            half_m += 1
        return (sign | half_m) & 0xFFFF
    if exp >= 31:
        return (sign | 0x7C00) & 0xFFFF
    half_m = mant >> 13
    if (mant & 0x1FFF) > 0x1000 or ((mant & 0x1FFF) == 0x1000 and (half_m & 1)):
        half_m += 1
        if half_m == 0x400:
            half_m = 0
            exp += 1
            if exp >= 31:
                return (sign | 0x7C00) & 0xFFFF
    return (sign | (exp << 10) | half_m) & 0xFFFF


# ---------------------------------------------------------------------------
# E2M1 (FP4)  — MXFP4 and NVFP4
# ---------------------------------------------------------------------------

def e2m1_decode(code: int) -> float:
    c = code & 0xF
    mag = E2M1_POS_TABLE[c & 0x7]
    if c & 0x8:
        return -mag if mag != 0.0 else -0.0
    return mag


def e2m1_encode(v: float) -> int:
    v = _f32(v)
    if not math.isfinite(v):
        return 0xF if _signbit(v) else 0x7
    a = abs(v)
    best = 0
    best_err = abs(a - E2M1_POS_TABLE[0])
    for i in range(1, 8):
        err = abs(a - E2M1_POS_TABLE[i])
        if err < best_err:  # ties keep the lower code (pack.hpp)
            best_err = err
            best = i
    code = best
    if best != 0 and _signbit(v):
        code |= 0x8
    return code


# ---------------------------------------------------------------------------
# E8M0 scale (OCP MX)
# ---------------------------------------------------------------------------

def e8m0_decode(bits: int) -> float:
    bits = bits & 0xFF
    if bits == 0xFF:
        return math.nan
    return math.ldexp(1.0, bits - 127)


def e8m0_encode(scale: float) -> int:
    scale = _f32(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        if not math.isfinite(scale):
            return 0xFF
        return 0
    _m, e = math.frexp(scale)
    m = _f32(math.ldexp(scale, -e))
    pow_ = e
    if m < _E8M0_MANT_THRESH:
        pow_ = e - 1
    bits = pow_ + 127
    if bits < 0:
        bits = 0
    if bits > 254:
        bits = 254
    return bits & 0xFF


# ---------------------------------------------------------------------------
# E4M3 (FP8) — NVFP4 block scale
# ---------------------------------------------------------------------------

def e4m3_decode(bits: int) -> float:
    bits = bits & 0xFF
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    if exp == 0xF and mant == 0x7:
        return math.nan
    if exp == 0:
        mag = math.ldexp(float(mant), -9)
    else:
        mag = math.ldexp(1.0 + float(mant) * 0.125, exp - 7)
    return -mag if sign else mag


def e4m3_encode(v: float) -> int:
    v = _f32(v)
    if not math.isfinite(v):
        return 0xFF if _signbit(v) else 0x7F
    a = abs(v)
    if a == 0.0:
        return 0x80 if _signbit(v) else 0x00
    best = 0
    best_err = math.inf
    for code in range(128):
        if code == 0x7F:
            continue
        cand = e4m3_decode(code)
        err = abs(a - cand)
        if err < best_err:
            best_err = err
            best = code
    out = best
    if v < 0.0:
        out |= 0x80
    return out & 0xFF


# ---------------------------------------------------------------------------
# INT4 nibble <-> signed value  (two's complement, [-8, 7])
# ---------------------------------------------------------------------------

def int4_from_nibble(n: int) -> int:
    n = n & 0xF
    return n - 16 if (n & 0x8) else n


def int4_to_nibble(v: int) -> int:
    if v < -8:
        v = -8
    if v > 7:
        v = 7
    return v & 0xF


# ---------------------------------------------------------------------------
# Packed containers
# ---------------------------------------------------------------------------

@dataclass
class PackedInt4:
    N: int = 0
    K: int = 0
    group: int = INT4_GROUP_128
    data: bytearray = field(default_factory=bytearray)
    scales: List[int] = field(default_factory=list)  # uint16 FP16 bits


@dataclass
class PackedMxfp4:
    N: int = 0
    K: int = 0
    data: bytearray = field(default_factory=bytearray)
    scales: List[int] = field(default_factory=list)  # uint8 E8M0


@dataclass
class PackedNvfp4:
    N: int = 0
    K: int = 0
    data: bytearray = field(default_factory=bytearray)
    scales: List[int] = field(default_factory=list)  # uint8 E4M3
    tensor_scale: float = 1.0


def _shape_from_numpy(w) -> Optional[Tuple[int, int]]:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return None
    if isinstance(w, np.ndarray) and w.ndim == 2:
        n, k = int(w.shape[0]), int(w.shape[1])
        return n, k
    return None


def _weights_f32(w: WeightIn, n: Optional[int], k: Optional[int]) -> Tuple[int, int, List[float]]:
    inferred = _shape_from_numpy(w)
    if inferred is not None:
        in_n, in_k = inferred
        if n is None:
            n = in_n
        if k is None:
            k = in_k
        if n != in_n or k != in_k:
            raise ValueError(f"W shape {(in_n, in_k)} does not match N={n} K={k}")
    if n is None or k is None:
        raise ValueError("N and K are required when W is not a 2-D numpy array")
    try:
        import numpy as np  # type: ignore

        if isinstance(w, np.ndarray):
            flat = np.asarray(w, dtype=np.float32).reshape(-1)
            vals = [_f32(float(x)) for x in flat.tolist()]
        else:
            vals = [_f32(x) for x in w]
    except ImportError:
        vals = [_f32(x) for x in w]
    if len(vals) != n * k:
        raise ValueError(f"W length {len(vals)} != N*K ({n}*{k})")
    return n, k, vals


# ---------------------------------------------------------------------------
# Pack from FP32
# ---------------------------------------------------------------------------

def pack_int4(
    w: WeightIn,
    n: Optional[int] = None,
    k: Optional[int] = None,
    group: int = INT4_GROUP_128,
) -> PackedInt4:
    if group not in (INT4_GROUP_64, INT4_GROUP_128):
        raise ValueError("pack_int4: group must be 64 or 128")
    n, k, w32 = _weights_f32(w, n, k)
    if k % group != 0:
        raise ValueError("pack_int4: K must be divisible by group")
    out = PackedInt4(N=n, K=k, group=group)
    out.data = bytearray(packed_nibbles_bytes(n, k))
    out.scales = [0] * int4_scale_count(n, k, group)
    groups_per_row = k // group
    seven = _f32(7.0)
    for ni in range(n):
        for g in range(groups_per_row):
            k0 = g * group
            amax = _f32(0.0)
            for t in range(group):
                amax = _f32(max(amax, abs(w32[ni * k + k0 + t])))
            scale = _f32(amax / seven) if amax > 0.0 else _f32(1.0)
            out.scales[ni * groups_per_row + g] = float_to_fp16_bits(scale)
            inv = _f32(1.0 / scale)
            for t in range(group):
                x = w32[ni * k + k0 + t]
                q = _nearbyint_to_int(_f32(x * inv))
                if q < -8:
                    q = -8
                if q > 7:
                    q = 7
                pack_nibble(out.data, ni * k + k0 + t, int4_to_nibble(q))
    return out


def pack_mxfp4(
    w: WeightIn, n: Optional[int] = None, k: Optional[int] = None
) -> PackedMxfp4:
    n, k, w32 = _weights_f32(w, n, k)
    if k % MXFP4_BLOCK != 0:
        raise ValueError("pack_mxfp4: K must be divisible by 32")
    out = PackedMxfp4(N=n, K=k)
    out.data = bytearray(packed_nibbles_bytes(n, k))
    out.scales = [0] * mxfp4_scale_count(n, k)
    blocks = k // MXFP4_BLOCK
    six = _f32(6.0)
    for ni in range(n):
        for b in range(blocks):
            k0 = b * MXFP4_BLOCK
            amax = _f32(0.0)
            for t in range(MXFP4_BLOCK):
                amax = _f32(max(amax, abs(w32[ni * k + k0 + t])))
            target = _f32(amax / six) if amax > 0.0 else e8m0_decode(127)
            sb = e8m0_encode(target)
            out.scales[ni * blocks + b] = sb
            scale = e8m0_decode(sb)
            inv = _f32(1.0 / scale)
            for t in range(MXFP4_BLOCK):
                x = _f32(w32[ni * k + k0 + t] * inv)
                pack_nibble(out.data, ni * k + k0 + t, e2m1_encode(x))
    return out


def pack_nvfp4(
    w: WeightIn,
    n: Optional[int] = None,
    k: Optional[int] = None,
    tensor_scale: float = 1.0,
) -> PackedNvfp4:
    n, k, w32 = _weights_f32(w, n, k)
    if k % NVFP4_BLOCK != 0:
        raise ValueError("pack_nvfp4: K must be divisible by 16")
    ts = _f32(tensor_scale)
    if not (ts > 0.0) or not math.isfinite(ts):
        raise ValueError("pack_nvfp4: tensor_scale must be finite and > 0")
    out = PackedNvfp4(N=n, K=k, tensor_scale=ts)
    out.data = bytearray(packed_nibbles_bytes(n, k))
    out.scales = [0] * nvfp4_scale_count(n, k)
    inv_tensor = _f32(1.0 / ts)
    blocks = k // NVFP4_BLOCK
    six = _f32(6.0)
    for ni in range(n):
        for b in range(blocks):
            k0 = b * NVFP4_BLOCK
            amax = _f32(0.0)
            for t in range(NVFP4_BLOCK):
                ww = _f32(w32[ni * k + k0 + t] * inv_tensor)
                amax = _f32(max(amax, abs(ww)))
            target = _f32(amax / six) if amax > 0.0 else _f32(1.0)
            sb = e4m3_encode(target)
            out.scales[ni * blocks + b] = sb
            scale = e4m3_decode(sb)
            if scale != 0.0 and math.isfinite(scale):
                inv = _f32(1.0 / scale)
            else:
                inv = _f32(0.0)
            for t in range(NVFP4_BLOCK):
                x = _f32(_f32(w32[ni * k + k0 + t] * inv_tensor) * inv)
                pack_nibble(out.data, ni * k + k0 + t, e2m1_encode(x))
    return out


# ---------------------------------------------------------------------------
# Reference dequant → FP32  (flat row-major N*K)
# ---------------------------------------------------------------------------

def dequant_int4(p: PackedInt4, out: Optional[MutableSequence[float]] = None) -> List[float]:
    nelt = p.N * p.K
    if out is None:
        out_list: List[float] = [0.0] * nelt
    else:
        out_list = out  # type: ignore[assignment]
    groups_per_row = p.K // p.group
    for ni in range(p.N):
        for kk in range(p.K):
            g = kk // p.group
            scale = fp16_bits_to_float(p.scales[ni * groups_per_row + g])
            q = int4_from_nibble(unpack_nibble(p.data, ni * p.K + kk))
            out_list[ni * p.K + kk] = _f32(float(q) * scale)
    return out_list if out is None else out  # type: ignore[return-value]


def dequant_mxfp4(p: PackedMxfp4, out: Optional[MutableSequence[float]] = None):
    nelt = p.N * p.K
    if out is None:
        out_list: List[float] = [0.0] * nelt
    else:
        out_list = out  # type: ignore[assignment]
    blocks = p.K // MXFP4_BLOCK
    for ni in range(p.N):
        for kk in range(p.K):
            b = kk // MXFP4_BLOCK
            scale = e8m0_decode(p.scales[ni * blocks + b])
            elem = e2m1_decode(unpack_nibble(p.data, ni * p.K + kk))
            out_list[ni * p.K + kk] = _f32(elem * scale)
    return out_list if out is None else out


def dequant_nvfp4(p: PackedNvfp4, out: Optional[MutableSequence[float]] = None):
    nelt = p.N * p.K
    if out is None:
        out_list: List[float] = [0.0] * nelt
    else:
        out_list = out  # type: ignore[assignment]
    blocks = p.K // NVFP4_BLOCK
    ts = _f32(p.tensor_scale)
    for ni in range(p.N):
        for kk in range(p.K):
            b = kk // NVFP4_BLOCK
            bscale = e4m3_decode(p.scales[ni * blocks + b])
            elem = e2m1_decode(unpack_nibble(p.data, ni * p.K + kk))
            out_list[ni * p.K + kk] = _f32(_f32(elem * bscale) * ts)
    return out_list if out is None else out


# ---------------------------------------------------------------------------
# Already-quantized codes
# ---------------------------------------------------------------------------

def pack_int4_codes(
    q: Sequence[int],
    scales_f32: Sequence[float],
    n: int,
    k: int,
    group: int,
) -> PackedInt4:
    out = PackedInt4(N=n, K=k, group=group)
    out.data = bytearray(packed_nibbles_bytes(n, k))
    scount = int4_scale_count(n, k, group)
    if len(scales_f32) != scount:
        raise ValueError("pack_int4_codes: bad scale count")
    if len(q) != n * k:
        raise ValueError("pack_int4_codes: bad q length")
    out.scales = [float_to_fp16_bits(_f32(s)) for s in scales_f32]
    for i in range(n * k):
        pack_nibble(out.data, i, int4_to_nibble(int(q[i])))
    return out


def pack_mxfp4_codes(
    elem_codes: Sequence[int], scale_bits: Sequence[int], n: int, k: int
) -> PackedMxfp4:
    out = PackedMxfp4(N=n, K=k)
    out.data = bytearray(packed_nibbles_bytes(n, k))
    scount = mxfp4_scale_count(n, k)
    if len(scale_bits) != scount:
        raise ValueError("pack_mxfp4_codes: bad scale count")
    if len(elem_codes) != n * k:
        raise ValueError("pack_mxfp4_codes: bad code length")
    out.scales = [int(s) & 0xFF for s in scale_bits]
    for i in range(n * k):
        pack_nibble(out.data, i, int(elem_codes[i]))
    return out


def pack_nvfp4_codes(
    elem_codes: Sequence[int],
    scale_bits: Sequence[int],
    n: int,
    k: int,
    tensor_scale: float = 1.0,
) -> PackedNvfp4:
    out = PackedNvfp4(N=n, K=k, tensor_scale=_f32(tensor_scale))
    out.data = bytearray(packed_nibbles_bytes(n, k))
    scount = nvfp4_scale_count(n, k)
    if len(scale_bits) != scount:
        raise ValueError("pack_nvfp4_codes: bad scale count")
    if len(elem_codes) != n * k:
        raise ValueError("pack_nvfp4_codes: bad code length")
    out.scales = [int(s) & 0xFF for s in scale_bits]
    for i in range(n * k):
        pack_nibble(out.data, i, int(elem_codes[i]))
    return out


def unpack_int4_codes(p: PackedInt4) -> List[int]:
    return [int4_from_nibble(unpack_nibble(p.data, i)) for i in range(p.N * p.K)]


def unpack_e2m1_codes(data: BytesLike, n: int, k: int) -> List[int]:
    return [unpack_nibble(data, i) for i in range(n * k)]


# ---------------------------------------------------------------------------
# Self-test (pytest-free)
# ---------------------------------------------------------------------------

def _close(a: float, b: float, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def self_test() -> int:
    fails = 0

    def expect(cond: bool, msg: str) -> None:
        nonlocal fails
        if not cond:
            print(f"FAIL: {msg}", file=sys.stderr)
            fails += 1

    def expect_near(a: float, b: float, tol: float, msg: str) -> None:
        expect(abs(a - b) <= tol, f"{msg}: |{a} - {b}| > {tol}")

    # E2M1 table
    for c in range(16):
        if c == 0x8:
            continue
        back = e2m1_encode(e2m1_decode(c))
        expect(back == c, f"e2m1 roundtrip code {c:#x} -> {back:#x}")
    expect(e2m1_encode(e2m1_decode(0x8)) == 0x0, "e2m1 signed-zero encodes to 0")
    expect_near(e2m1_decode(0x0), 0.0, 0.0, "e2m1 0")
    expect_near(e2m1_decode(0x1), 0.5, 0.0, "e2m1 0.5")
    expect_near(e2m1_decode(0x7), 6.0, 0.0, "e2m1 6")
    expect_near(e2m1_decode(0xF), -6.0, 0.0, "e2m1 -6")
    expect_near(e2m1_decode(0x8), 0.0, 0.0, "e2m1 -0")
    expect(_f32_bits(e2m1_decode(0x8)) == 0x80000000, "e2m1 0x8 is -0.0 bits")
    # ties keep the lower magnitude code
    expect(e2m1_encode(0.25) == 0, "e2m1 tie 0.25 -> 0")
    expect(e2m1_encode(2.5) == 4, "e2m1 tie 2.5 -> 2.0")
    expect(e2m1_encode(5.0) == 6, "e2m1 tie 5 -> 4.0")
    expect(e2m1_encode(float("inf")) == 0x7, "e2m1 +inf")
    expect(e2m1_encode(float("-inf")) == 0xF, "e2m1 -inf")

    # E8M0
    expect_near(e8m0_decode(127), 1.0, 0.0, "e8m0 127")
    expect_near(e8m0_decode(128), 2.0, 0.0, "e8m0 128")
    expect_near(e8m0_decode(126), 0.5, 0.0, "e8m0 126")
    expect(math.isnan(e8m0_decode(0xFF)), "e8m0 0xFF is NaN")
    expect(e8m0_encode(1.0) == 127, "e8m0 encode 1")
    expect(e8m0_encode(2.0) == 128, "e8m0 encode 2")
    expect(e8m0_encode(0.5) == 126, "e8m0 encode 0.5")
    expect(e8m0_encode(3.0) == 129, "e8m0 encode 3 -> 4")
    expect(e8m0_encode(0.0) == 0, "e8m0 encode 0")
    expect(e8m0_encode(float("nan")) == 0xFF, "e8m0 encode nan")

    # E4M3
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            continue
        v = e4m3_decode(c)
        back = e4m3_encode(v)
        expect(back == c, f"e4m3 roundtrip {c:#04x} -> {back:#04x} (v={v})")
    expect_near(e4m3_decode(0x00), 0.0, 0.0, "e4m3 0")
    expect_near(e4m3_decode(0x38), 1.0, 1e-6, "e4m3 1.0")
    expect(math.isnan(e4m3_decode(0x7F)), "e4m3 0x7F NaN")
    expect(math.isnan(e4m3_decode(0xFF)), "e4m3 0xFF NaN")
    expect(e4m3_encode(0.0) == 0x00, "e4m3 +0")
    expect(e4m3_encode(-0.0) == 0x80, "e4m3 -0")

    # FP16
    for v in (0.0, 1.0, -1.0, 0.5, 2.0, 7.0, 0.0001, 65504.0):
        back = fp16_bits_to_float(float_to_fp16_bits(v))
        if v == 0.0:
            expect_near(back, 0.0, 0.0, "fp16 0")
        else:
            expect(
                abs(back - v) <= abs(v) * 1e-3 + 1e-6,
                f"fp16 {v} -> {back}",
            )
    expect(float_to_fp16_bits(0.0) == 0, "fp16 +0 bits")
    expect(float_to_fp16_bits(-0.0) == 0x8000, "fp16 -0 bits")
    expect(float_to_fp16_bits(float("inf")) == 0x7C00, "fp16 +inf")

    # INT4 nibbles
    for v in range(-8, 8):
        expect(int4_from_nibble(int4_to_nibble(v)) == v, f"int4 nibble {v}")

    # Nibble order
    buf = bytearray(4)
    pack_nibble(buf, 0, 0xA)
    pack_nibble(buf, 1, 0xB)
    pack_nibble(buf, 2, 0xC)
    pack_nibble(buf, 3, 0xD)
    expect(buf[0] == 0xBA, f"nibble byte0 {buf[0]:#04x}")
    expect(buf[1] == 0xDC, f"nibble byte1 {buf[1]:#04x}")
    expect(unpack_nibble(buf, 0) == 0xA, "unpack 0")
    expect(unpack_nibble(buf, 1) == 0xB, "unpack 1")
    expect(unpack_nibble(buf, 2) == 0xC, "unpack 2")
    expect(unpack_nibble(buf, 3) == 0xD, "unpack 3")

    # INT4 code probe: q=1, scale=prime 2
    n, k, group = 4, 256, 128
    q = [1] * (n * k)
    ngrp = int4_scale_count(n, k, group)
    # first 64 primes from correctness.hpp (enough for this shape)
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
        257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    ]
    scales = [float(primes[i % len(primes)]) for i in range(ngrp)]
    packed = pack_int4_codes(q, scales, n, k, group)
    got = dequant_int4(packed)
    s0 = fp16_bits_to_float(float_to_fp16_bits(2.0))
    expect_near(got[0], s0, 1e-3, "int4 prime[0]")

    # INT4 one-hot
    qh = [0] * (8 * 128)
    qh[3 * 128 + 17] = 5
    ph = pack_int4_codes(qh, [2.0] * int4_scale_count(8, 128, 128), 8, 128, 128)
    gh = dequant_int4(ph)
    expect_near(gh[3 * 128 + 17], 5.0 * fp16_bits_to_float(float_to_fp16_bits(2.0)), 1e-4, "int4 one-hot")
    expect_near(gh[0], 0.0, 0.0, "int4 one-hot zero")

    # MXFP4 one-hot: code 0x5 (=3) * e8m0 128 (=2) → 6
    codes = [0] * (4 * 32)
    codes[2 * 32 + 7] = 0x5
    pm = pack_mxfp4_codes(codes, [128] * mxfp4_scale_count(4, 32), 4, 32)
    gm = dequant_mxfp4(pm)
    expect_near(gm[2 * 32 + 7], 6.0, 0.0, "mxfp4 one-hot")
    expect_near(gm[0], 0.0, 0.0, "mxfp4 one-hot zero")

    # MXFP4 ramp, scale=1
    ramp = [i % 8 for i in range(2 * 32)]
    pr = pack_mxfp4_codes(ramp, [127] * mxfp4_scale_count(2, 32), 2, 32)
    gr = dequant_mxfp4(pr)
    expect_near(gr[0], 0.0, 0.0, "mx ramp 0")
    expect_near(gr[1], 0.5, 0.0, "mx ramp 0.5")
    expect_near(gr[7], 6.0, 0.0, "mx ramp 6")

    # NVFP4 prime-ish: elem=1, scale=e4m3(2), tensor=2
    nc, kc = 2, 32
    nv_codes = [0x2] * (nc * kc)
    nv_scales = [e4m3_encode(2.0) for _ in range(nvfp4_scale_count(nc, kc))]
    pn = pack_nvfp4_codes(nv_codes, nv_scales, nc, kc, tensor_scale=2.0)
    gn = dequant_nvfp4(pn)
    expect_near(gn[0], e4m3_decode(e4m3_encode(2.0)) * 2.0, 1e-5, "nvfp4 prime")

    # INT4 quantize roundtrip
    n, k = 4, 128
    w = [float((i % 15) - 7) for i in range(n * k)]
    pi = pack_int4(w, n, k, 128)
    gi = dequant_int4(pi)
    ok = all(_close(a, b, 0.05, 0.05) for a, b in zip(gi, w))
    expect(ok, "int4 quantize roundtrip")

    # MXFP4 exact-set roundtrip
    n, k = 2, 32
    w = [e2m1_decode(i % 16) * 4.0 for i in range(n * k)]
    px = pack_mxfp4(w, n, k)
    gx = dequant_mxfp4(px)
    ok = all(_close(a, b, 1e-5, 1e-5) for a, b in zip(gx, w))
    expect(ok, "mxfp4 exact-set roundtrip")

    # NVFP4 quantize roundtrip
    n, k = 2, 16
    w = [e2m1_decode((i * 3) % 16) * 1.0 for i in range(n * k)]
    pv = pack_nvfp4(w, n, k, tensor_scale=1.0)
    gv = dequant_nvfp4(pv)
    ok = all(_close(a, b, 0.05, 0.05) for a, b in zip(gv, w))
    expect(ok, "nvfp4 quantize roundtrip")

    # unpack_int4_codes
    n, k, group = 2, 16, 16
    q = [(i % 16) - 8 for i in range(n * k)]
    scales = [1.0] * int4_scale_count(n, k, group)
    pc = pack_int4_codes(q, scales, n, k, group)
    back_q = unpack_int4_codes(pc)
    expect(back_q == q, "unpack_int4_codes")

    # awkward K=96 (div by 32, not 128) MXFP4 one-hot
    n, k = 96, 96
    expect(k % 128 != 0 and k % 32 == 0, "awkward K setup")
    codes = [0] * (n * k)
    codes[10 * k + 33] = 0x6
    pa = pack_mxfp4_codes(codes, [127] * mxfp4_scale_count(n, k), n, k)
    ga = dequant_mxfp4(pa)
    expect_near(ga[10 * k + 33], 4.0, 0.0, "awkward K mxfp4")

    # byte accounting
    expect(abs(mxfp4_bytes_per_weight() - 0.53125) < 1e-12, "mx bpw")
    expect(abs(nvfp4_bytes_per_weight() - 0.5625) < 1e-12, "nv bpw")
    expect(abs(int4_bytes_per_weight(128) - 0.515625) < 1e-12, "int4 g128 bpw")
    expect(abs(int4_bytes_per_weight(64) - 0.53125) < 1e-12, "int4 g64 bpw")

    # API rejects
    threw = False
    try:
        pack_int4([0.0] * (2 * 32), 2, 32, 128)
    except ValueError:
        threw = True
    expect(threw, "INT4-g128 rejects K=32")
    threw = False
    try:
        pack_mxfp4([0.0] * (2 * 16), 2, 16)
    except ValueError:
        threw = True
    expect(threw, "MXFP4 rejects K=16")

    if fails:
        print(f"qgemm_pack: {fails} failure(s)", file=sys.stderr)
        return 1
    print("qgemm_pack: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    del argv
    return self_test()


if __name__ == "__main__":
    raise SystemExit(main())
