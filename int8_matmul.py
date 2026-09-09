"""int8 quantized matrix multiply on the analog crossbar.

Symmetric int8 weights and activations, exact int32 accumulation as the reference, and an
int8 requantized output -- the workload a real compute-in-memory inference accelerator
runs. The analog path underneath is `fp_matmul.matmul`, unchanged: this module only
quantizes going in and requantizes coming out.

    weights      s_w = max|A| / 127,  A_q = clip(round(A / s_w), -127, 127)
    activations  s_x = max|B| / 127,  B_q = clip(round(B / s_x), -127, 127)
    reference    C_ref = A_q @ B_q, exact in int64, asserted to fit int32
    analog       C_analog = fp_matmul.matmul(A_q, B_q, ...)
    output       s_y = max|C_ref| / 127,  y = clip(round(C / s_y), -127, 127)

`s_w` is per-tensor or **per output channel** (`per_channel=True`, one scale per row of
`A`, which is one scale per column pair of the tile). Per-channel is the standard scheme in
modern quantized inference and it is the interesting case here: the per-channel scale is a
multiplier the dequantization path already applies, so folding a measured per-column IR-drop
gain into it costs no hardware at all -- see `fit_gain` and docs/MATMUL.md.

Why the int8 *output* is the metric and the int32 accumulator is not: recovering the
accumulator exactly needs 1 LSB out of 127*127*N of full scale, ~19 bits at N = 32, and the
analog path delivers ~7. That is unreachable and stays unreachable. The requantized int8
output only needs 1 LSB out of 255 of the output range, ~0.4%, which is the same order as
the measured analog error -- so it is the metric where the answer is neither trivially yes
nor trivially no. `bits_for_exact` and `Int8Metrics.bits_needed` state the gap numerically.
"""
from typing import NamedTuple

import numpy as np

import crossbar as C
import fp_matmul as F

INT8_MAX = 127


def quantize(X, axis=None):
    """Symmetric int8: `(X_q, s)` with `X ~= s * X_q` and `X_q` integer-valued float64.

    `axis=None` is per-tensor and `s` is 0-d; `axis=1` on a weight matrix gives one scale
    per output channel and `s` is (M,1), so both broadcast against `X`. Clipping is a
    no-op for symmetric max scaling -- `max|X|/s` is exactly 127 -- and is kept only so a
    caller who passes their own scale cannot silently overflow the range.
    """
    X = np.asarray(X, dtype=float)
    peak = np.abs(X).max() if axis is None else np.abs(X).max(axis=axis, keepdims=True)
    s = np.where(peak > 0.0, peak, 1.0) / INT8_MAX
    return np.clip(np.rint(X / s), -INT8_MAX, INT8_MAX), s


def requantize(Cacc, s_y: float) -> np.ndarray:
    """int32 accumulator -> symmetric int8 output, round-to-nearest."""
    return np.clip(np.rint(np.asarray(Cacc, dtype=float) / s_y),
                   -INT8_MAX, INT8_MAX).astype(np.int64)


def bits_for_exact(N: int) -> float:
    """Worst-case bits needed to resolve one accumulator LSB: `log2(2 * 127**2 * N)`.

    The 2 is because resolving an integer needs half an LSB, not a whole one. This is the
    operand-independent bound; `Int8Metrics.bits_needed` is the same quantity against the
    peak a particular matrix pair actually reaches, which is always smaller.
    """
    return float(np.log2(2 * INT8_MAX ** 2 * N))


class Int8Result(NamedTuple):
    c_ref: np.ndarray      # exact integer accumulator, int64
    c_analog: np.ndarray   # the analog path's estimate of it, float64
    y_ref: np.ndarray      # c_ref requantized to int8
    y_analog: np.ndarray   # c_analog requantized to int8
    s_y: float             # output requantization step, in accumulator LSBs


class Int8Metrics(NamedTuple):
    exact: bool          # round(c_analog) == c_ref elementwise, no tolerance at all
    match: float         # fraction of int8 outputs that are exactly right
    max_lsb: int         # worst int8 output deviation, in output LSBs
    bits_max: float      # accumulator bits recovered, max-norm
    bits_rms: float      # accumulator bits recovered, rms-norm
    bits_needed: float   # log2(2 * max|c_ref|): exact int32 recovery for *this* pair


def run_q(A_q, B_q, **kw) -> Int8Result:
    """The analog path for operands that are already int8.

    Split out from `matmul` so a sweep can quantize once and re-run the same `A_q`/`B_q`
    across parasitic settings -- otherwise every point would compare against a different
    integer reference. `**kw` goes to `fp_matmul.matmul`.
    """
    A_q, B_q = np.asarray(A_q, dtype=float), np.asarray(B_q, dtype=float)
    c_ref = A_q.astype(np.int64) @ B_q.astype(np.int64)
    peak = int(np.abs(c_ref).max())
    if peak > np.iinfo(np.int32).max:
        raise ValueError(f"accumulator peak {peak} overflows int32")

    c_analog = F.matmul(A_q, B_q, **kw)
    s_y = peak / INT8_MAX if peak > 0 else 1.0
    return Int8Result(c_ref, c_analog, requantize(c_ref, s_y),
                      requantize(c_analog, s_y), s_y)


def matmul(A, B, per_channel: bool = False, **kw) -> Int8Result:
    """Quantize `A` and `B` to int8, run the analog engine, requantize the output."""
    A_q, _ = quantize(A, axis=1 if per_channel else None)
    B_q, _ = quantize(B)
    return run_q(A_q, B_q, **kw)


def calibrate(r: Int8Result, gain) -> Int8Result:
    """Divide a per-output-channel gain out of the accumulator and requantize again.

    No new simulation: the array produced `c_analog` already and the calibration is a
    digital multiply on top of it. That is the whole point -- on hardware it folds into the
    per-channel weight scale, so it costs neither a multiplier nor a second read.
    """
    c = r.c_analog / np.asarray(gain, dtype=float).reshape(-1, 1)
    return r._replace(c_analog=c, y_analog=requantize(c, r.s_y))


def metrics(r: Int8Result) -> Int8Metrics:
    """Everything the study reports, from one `Int8Result`."""
    d = r.y_analog - r.y_ref
    mx, rms = F.rel_errors(r.c_analog, r.c_ref)
    peak = float(np.abs(r.c_ref).max())
    return Int8Metrics(
        exact=bool(np.array_equal(np.rint(r.c_analog).astype(np.int64), r.c_ref)),
        match=float((d == 0).mean()),
        max_lsb=int(np.abs(d).max()),
        bits_max=F.effective_bits(mx),
        bits_rms=F.effective_bits(rms),
        bits_needed=float(np.log2(2 * peak)) if peak > 0 else 0.0,
    )


def fit_gain(c_analog, c_ref, per_channel: bool = True) -> np.ndarray:
    """Best-fit droop gain, (M,), per output channel or one scalar broadcast to all.

    Uses `crossbar.error_metrics`' closed form `a = (got.ref)/(ref.ref)`, the argmin of
    ||got - a*ref||, which `verify_mna.py` already cross-checks against `lstsq`.
    `error_metrics` also computes elementwise percentages this does not use, and an
    accumulator entry of exactly zero would make those infinite -- hence the errstate.

    A real system does not need new hardware to apply this: per-channel weight
    quantization already multiplies each output channel by its own `s_w`, so `s_w/a` is the
    same instruction. What it does need is the calibration measurement itself, and `a`
    depends on the activation statistics, not only on the array -- see the
    `calibration transfer` panel in scripts/int8_accuracy.py.
    """
    got, ref = np.asarray(c_analog, dtype=float), np.asarray(c_ref, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if not per_channel:
            return np.full(ref.shape[0], C.error_metrics(got.ravel(), ref.ravel())[1])
        return np.array([C.error_metrics(got[m], ref[m])[1] for m in range(ref.shape[0])])


if __name__ == "__main__":
    import time
    from dataclasses import replace

    rng = np.random.default_rng(0)
    M, N, K = 12, 32, 6
    A, B = rng.standard_normal((M, N)), rng.standard_normal((N, K))
    print(f"A {A.shape} @ B {B.shape} int8 on a 16x16 tile, unipolar two-pass, batched")
    print(f"exact int32 recovery would need {bits_for_exact(N):.1f} bits worst case")

    for r in (0.0, 0.25, 1.0, 5.0):
        xbar = replace(F.IDEAL_XBAR, r_row=r, r_col=r)
        for per_channel in (False, True):
            A_q, _ = quantize(A, axis=1 if per_channel else None)
            B_q, _ = quantize(B)
            t0 = time.perf_counter()
            res = run_q(A_q, B_q, xbar=xbar, rundir=f"/tmp/int8_demo_r{r:g}_{per_channel:d}")
            wall = time.perf_counter() - t0
            m = metrics(res)
            cal = metrics(calibrate(res, fit_gain(res.c_analog, res.c_ref,
                                                  per_channel=per_channel)))
            print(f"  r={r:>4.2f}  {'per-channel' if per_channel else 'per-tensor ':11s} | "
                  f"int8 match {100*m.match:6.2f}%  worst {m.max_lsb:3d} LSB | "
                  f"accum {m.bits_rms:5.2f} of {m.bits_needed:5.2f} bits needed | "
                  f"calibrated match {100*cal.match:6.2f}%  worst {cal.max_lsb:3d} LSB"
                  f"{'  | int32 EXACT' if m.exact else ''} | {wall:.2f} s")
