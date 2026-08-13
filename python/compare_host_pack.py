#!/usr/bin/env python3
"""Compare python/qgemm_pack.py encodings against include/qgemm/pack.hpp.

If a C++ compiler is available this script compiles a tiny throwaway dumper
that includes pack.hpp (no Makefile / CUDA / include/ edits) and checks
E2M1 / E8M0 / E4M3 tables plus packed INT4-g128, MXFP4, and NVFP4 buffers
against the Python implementation.

    python3 python/compare_host_pack.py
    python3 python/compare_host_pack.py --skip-cpp

If the host dump cannot be built, this prints how to compare by hand and
still runs Python-only consistency checks (does not fail the skip path).
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

_PY = Path(__file__).resolve().parent
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

import qgemm_pack as qp  # noqa: E402

# ---------------------------------------------------------------------------
# How to compare if this script cannot compile pack.hpp
# ---------------------------------------------------------------------------

HOWTO = """
How to compare Python pack against pack.hpp
===========================================

1. Python self-test (stdlib, no torch/triton):

     python3 python/qgemm_pack.py

   Tables and pack/dequant must match include/qgemm/pack.hpp
   (E2M1_POS_TABLE, E8M0 2^(bits-127), E4M3 bias-7, sequential nibbles).

2. Host C++ tests (optional; does not modify Makefile):

     g++ -std=c++17 -O0 -I include tests/test_pack_roundtrip.cpp \\
         -o /tmp/test_pack_roundtrip
     /tmp/test_pack_roundtrip

3. Bit-exact dump (what this script does automatically):

     g++ -std=c++17 -O0 -I include dump_pack.cpp -o /tmp/dump_pack
     /tmp/dump_pack

   dump_pack.cpp should #include "qgemm/pack.hpp" and print:

     * e2m1_decode(0..15) as IEEE-754 binary32 hex
     * e8m0_encode/decode of {0.5, 1, 2, 3}
     * e4m3_decode(0..255) as hex or "nan"
     * pack_int4 / pack_mxfp4 / pack_nvfp4 of a known W as hex data + scales

   Then compare those lines to qgemm_pack.e2m1_* / pack_*.

4. Do not edit include/qgemm/pack.hpp to make Python match; Python follows
   the host header.
""".strip()


DUMP_SRC = r"""
#include "qgemm/pack.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static void f32hex(float f) {
  if (std::isnan(f)) {
    std::printf("nan");
    return;
  }
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  std::printf("%08x", u);
}

static std::uint32_t f32u(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

int main() {
  std::printf("E2M1_DECODE\n");
  for (int c = 0; c < 16; ++c) {
    std::printf("%d ", c);
    f32hex(qgemm::e2m1_decode(static_cast<std::uint8_t>(c)));
    std::printf("\n");
  }

  std::printf("E2M1_ENCODE\n");
  const float e2_in[] = {0.f, 0.25f, 0.5f, 0.75f, 1.f, 2.5f, 3.f,
                         5.f, 6.f, -6.f, 7.f};
  for (float v : e2_in) {
    std::printf("%.9g %u\n", static_cast<double>(v),
                static_cast<unsigned>(qgemm::e2m1_encode(v)));
  }
  std::printf("negzero %u\n",
              static_cast<unsigned>(qgemm::e2m1_encode(-0.0f)));

  std::printf("E8M0\n");
  const float e8_in[] = {0.5f, 1.f, 2.f, 3.f, 4.f, 0.f};
  for (float v : e8_in) {
    std::printf("enc %.9g %u\n", static_cast<double>(v),
                static_cast<unsigned>(qgemm::e8m0_encode(v)));
  }
  for (int b : {0, 1, 126, 127, 128, 129, 254, 255}) {
    std::printf("dec %d ", b);
    f32hex(qgemm::e8m0_decode(static_cast<std::uint8_t>(b)));
    std::printf("\n");
  }

  std::printf("E4M3_DECODE\n");
  for (int c = 0; c < 256; ++c) {
    std::printf("%d ", c);
    f32hex(qgemm::e4m3_decode(static_cast<std::uint8_t>(c)));
    std::printf("\n");
  }
  std::printf("E4M3_ENCODE_RT\n");
  for (int c = 0; c < 256; ++c) {
    if ((c & 0x7F) == 0x7F) continue;
    const float v = qgemm::e4m3_decode(static_cast<std::uint8_t>(c));
    const std::uint8_t back = qgemm::e4m3_encode(v);
    std::printf("%d %u\n", c, static_cast<unsigned>(back));
  }

  std::printf("FP16\n");
  const float h_in[] = {0.f, 1.f, -1.f, 0.5f, 2.f, 7.f, 0.0001f, 65504.f,
                        0.1f};
  for (float v : h_in) {
    const std::uint16_t h = qgemm::float_to_fp16_bits(v);
    std::printf("to %08x %04x\n", f32u(v), static_cast<unsigned>(h));
  }
  std::printf("to_negzero %04x\n",
              static_cast<unsigned>(qgemm::float_to_fp16_bits(-0.0f)));

  // Known W — same construction as python/compare_host_pack.py
  {
    const std::size_t N = 4, K = 128;
    std::vector<float> W(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
      W[i] = static_cast<float>(static_cast<int>(i % 15) - 7);
    }
    auto p = qgemm::pack_int4(W.data(), N, K, 128);
    std::printf("PACK_INT4 %zu %zu\n", p.data.size(), p.scales.size());
    std::printf("data ");
    for (auto b : p.data) std::printf("%02x", b);
    std::printf("\nscales ");
    for (auto s : p.scales) std::printf("%04x", s);
    std::printf("\n");
  }
  {
    const std::size_t N = 2, K = 32;
    std::vector<float> W(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
      W[i] = qgemm::e2m1_decode(static_cast<std::uint8_t>(i % 16)) * 4.0f;
    }
    auto p = qgemm::pack_mxfp4(W.data(), N, K);
    std::printf("PACK_MXFP4 %zu %zu\n", p.data.size(), p.scales.size());
    std::printf("data ");
    for (auto b : p.data) std::printf("%02x", b);
    std::printf("\nscales ");
    for (auto s : p.scales) std::printf("%02x", s);
    std::printf("\n");
  }
  {
    const std::size_t N = 2, K = 16;
    std::vector<float> W(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
      W[i] = qgemm::e2m1_decode(static_cast<std::uint8_t>((i * 3) % 16));
    }
    auto p = qgemm::pack_nvfp4(W.data(), N, K, 1.0f);
    std::printf("PACK_NVFP4 %zu %zu %.9g\n", p.data.size(), p.scales.size(),
                static_cast<double>(p.tensor_scale));
    std::printf("data ");
    for (auto b : p.data) std::printf("%02x", b);
    std::printf("\nscales ");
    for (auto s : p.scales) std::printf("%02x", s);
    std::printf("\n");
  }
  {
    // Mixed fractional INT4 so fp16 scale bits are exercised.
    const std::size_t N = 1, K = 128;
    std::vector<float> W(N * K);
    for (std::size_t i = 0; i < N * K; ++i) {
      W[i] = static_cast<float>(i) * 0.01f - 0.64f;
    }
    auto p = qgemm::pack_int4(W.data(), N, K, 128);
    std::printf("PACK_INT4_FRAC %zu %zu\n", p.data.size(), p.scales.size());
    std::printf("data ");
    for (auto b : p.data) std::printf("%02x", b);
    std::printf("\nscales ");
    for (auto s : p.scales) std::printf("%04x", s);
    std::printf("\n");
  }
  return 0;
}
"""


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    root = here.parent
    pack = root / "include" / "qgemm" / "pack.hpp"
    if not pack.is_file():
        raise SystemExit(f"error: pack.hpp not found at {pack}")
    return root


def _f32_hex(v: float) -> str:
    if math.isnan(v):
        return "nan"
    u = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    return f"{u:08x}"


def _hex_bytes(data) -> str:
    return "".join(f"{int(b) & 0xFF:02x}" for b in data)


def _hex_u16(vals) -> str:
    return "".join(f"{int(s) & 0xFFFF:04x}" for s in vals)


def python_dump_lines() -> list[str]:
    lines: list[str] = []
    lines.append("E2M1_DECODE")
    for c in range(16):
        lines.append(f"{c} {_f32_hex(qp.e2m1_decode(c))}")

    lines.append("E2M1_ENCODE")
    for v in (0.0, 0.25, 0.5, 0.75, 1.0, 2.5, 3.0, 5.0, 6.0, -6.0, 7.0):
        # Format matches C++ %.9g of a float32-promoted double.
        vg = float(struct.unpack("<f", struct.pack("<f", v))[0])
        lines.append(f"{_fmt_g(vg)} {qp.e2m1_encode(v)}")
    lines.append(f"negzero {qp.e2m1_encode(-0.0)}")

    lines.append("E8M0")
    for v in (0.5, 1.0, 2.0, 3.0, 4.0, 0.0):
        vg = float(struct.unpack("<f", struct.pack("<f", v))[0])
        lines.append(f"enc {_fmt_g(vg)} {qp.e8m0_encode(v)}")
    for b in (0, 1, 126, 127, 128, 129, 254, 255):
        lines.append(f"dec {b} {_f32_hex(qp.e8m0_decode(b))}")

    lines.append("E4M3_DECODE")
    for c in range(256):
        lines.append(f"{c} {_f32_hex(qp.e4m3_decode(c))}")
    lines.append("E4M3_ENCODE_RT")
    for c in range(256):
        if (c & 0x7F) == 0x7F:
            continue
        lines.append(f"{c} {qp.e4m3_encode(qp.e4m3_decode(c))}")

    lines.append("FP16")
    for v in (0.0, 1.0, -1.0, 0.5, 2.0, 7.0, 0.0001, 65504.0, 0.1):
        vf = struct.unpack("<f", struct.pack("<f", v))[0]
        bits = struct.unpack("<I", struct.pack("<f", vf))[0]
        lines.append(f"to {bits:08x} {qp.float_to_fp16_bits(v):04x}")
    lines.append(f"to_negzero {qp.float_to_fp16_bits(-0.0):04x}")

    n, k = 4, 128
    w = [float((i % 15) - 7) for i in range(n * k)]
    p = qp.pack_int4(w, n, k, 128)
    lines.append(f"PACK_INT4 {len(p.data)} {len(p.scales)}")
    lines.append("data " + _hex_bytes(p.data))
    lines.append("scales " + _hex_u16(p.scales))

    n, k = 2, 32
    w = [qp.e2m1_decode(i % 16) * 4.0 for i in range(n * k)]
    p = qp.pack_mxfp4(w, n, k)
    lines.append(f"PACK_MXFP4 {len(p.data)} {len(p.scales)}")
    lines.append("data " + _hex_bytes(p.data))
    lines.append("scales " + _hex_bytes(p.scales))

    n, k = 2, 16
    w = [qp.e2m1_decode((i * 3) % 16) for i in range(n * k)]
    p = qp.pack_nvfp4(w, n, k, tensor_scale=1.0)
    lines.append(f"PACK_NVFP4 {len(p.data)} {len(p.scales)} {_fmt_g(p.tensor_scale)}")
    lines.append("data " + _hex_bytes(p.data))
    lines.append("scales " + _hex_bytes(p.scales))

    n, k = 1, 128
    w = [float(i) * 0.01 - 0.64 for i in range(n * k)]
    p = qp.pack_int4(w, n, k, 128)
    lines.append(f"PACK_INT4_FRAC {len(p.data)} {len(p.scales)}")
    lines.append("data " + _hex_bytes(p.data))
    lines.append("scales " + _hex_u16(p.scales))
    return lines


def _fmt_g(v: float) -> str:
    """Approximate C/C++ default %g / %.9g for this small value set."""
    return format(float(v), ".9g")


def _run_cpp_dump(root: Path) -> tuple[str | None, str]:
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if cxx is None:
        return None, "no C++ compiler (g++/c++/clang++) on PATH"
    inc = root / "include"
    try:
        with tempfile.TemporaryDirectory(prefix="qgemm-pack-dump-") as td:
            src = Path(td) / "dump_pack.cpp"
            exe = Path(td) / "dump_pack"
            src.write_text(DUMP_SRC)
            comp = subprocess.run(
                [cxx, "-std=c++17", "-O0", f"-I{inc}", str(src), "-o", str(exe)],
                check=False,
                capture_output=True,
                text=True,
            )
            if comp.returncode != 0:
                err = (comp.stderr or comp.stdout or "").strip()
                return None, f"compile failed:\n{err}"
            run = subprocess.run(
                [str(exe)],
                check=False,
                capture_output=True,
                text=True,
            )
            if run.returncode != 0:
                err = (run.stderr or run.stdout or "").strip()
                return None, f"dump exited {run.returncode}: {err}"
            return run.stdout, ""
    except OSError as e:
        return None, f"compiler invocation failed: {e}"


def _compare_dumps(host: str, py_lines: list[str]) -> list[str]:
    host_lines = [ln.rstrip("\n") for ln in host.splitlines() if ln.strip() != ""]
    py_clean = [ln for ln in py_lines if ln.strip() != ""]
    diffs: list[str] = []
    n = max(len(host_lines), len(py_clean))
    if len(host_lines) != len(py_clean):
        diffs.append(
            f"line count host={len(host_lines)} python={len(py_clean)}"
        )
    for i in range(n):
        h = host_lines[i] if i < len(host_lines) else "<missing>"
        p = py_clean[i] if i < len(py_clean) else "<missing>"
        if h != p:
            diffs.append(f"L{i + 1}: host={h!r} python={p!r}")
            if len(diffs) >= 20:
                diffs.append("... (further mismatches truncated)")
                break
    return diffs


def python_only_checks() -> int:
    return qp.self_test()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skip-cpp",
        action="store_true",
        help="do not compile pack.hpp; print HOWTO and run Python self-test",
    )
    args = p.parse_args(argv)

    print("python self-test:")
    rc = python_only_checks()
    if rc != 0:
        return rc

    if args.skip_cpp:
        print()
        print(HOWTO)
        print()
        print("compare_host_pack: skipped C++ dump (--skip-cpp)")
        return 0

    root = _repo_root()
    host, err = _run_cpp_dump(root)
    if host is None:
        print()
        print("C++ dump unavailable:", err)
        print()
        print(HOWTO)
        print()
        print("compare_host_pack: SKIP (python self-test passed)")
        return 0

    diffs = _compare_dumps(host, python_dump_lines())
    if diffs:
        print("compare_host_pack: MISMATCH vs pack.hpp", file=sys.stderr)
        for d in diffs:
            print(" ", d, file=sys.stderr)
        return 1
    print("compare_host_pack: OK (python matches pack.hpp dump)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
