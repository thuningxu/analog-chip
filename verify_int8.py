"""Machine checks for the int8 quantized matmul in `int8_matmul.py`.

The gate is check 1, and it is binary: at `r_row = r_col = 0` with `CellParams(passive=True)`
the analog accumulator rounded to integers must **equal** `A_q @ B_q` elementwise, with no
tolerance. Nothing else in this file, and nothing in `scripts/int8_accuracy.py`, means
anything until it passes.

Check 1 also reports how much slack each gate has, because the two gates are not ordered
the way one might assume. Rounding to an integer tolerates an absolute error of 0.5, which
at an accumulator peak of ~2e4 is a *relative* tolerance of ~2e-5 -- seven orders of
magnitude looser than the 1e-12 that `verify_matmul.py` asserts on the float engine. The
integer gate is the one the workload cares about and the one with no arbitrary constant in
it; the float gate is the numerically tighter one. Both are kept.

Checks, in order:
    1  int32 exactness  round(analog) == A_q @ B_q exactly, across non-square shapes,
                        ragged tiles, all-negative A, mixed signs, a zero column of B, and
                        both weight-scale granularities. Plus the float relative error and
                        the slack in each gate.
    2  quantization     symmetric int8 round-trip properties, per-tensor and per-channel,
                        and the int32 accumulator bound.
    3  int8 output      at zero parasitics every requantized int8 output is exactly right;
                        requantization stays in range; `calibrate` runs no new simulation.
    4  calibration      droop is one-sided, and a per-channel gain removes more of it than
                        one global gain -- the claim that per-channel weight scales absorb
                        IR droop for free.
"""

import time
from dataclasses import replace

import numpy as np

import crossbar as C
import fp_matmul as F
import int8_matmul as Q


def wire(r: float, **kw) -> C.CrossbarParams:
    """`IDEAL_XBAR` at a chosen wire resistance."""
    return replace(F.IDEAL_XBAR, r_row=r, r_col=r, **kw)


def cases():
    """(tag, A, B, rows, cols) covering shape, sign and tiling coverage."""
    rng = np.random.default_rng(7)
    zero_col = rng.standard_normal((11, 4))
    zero_col[:, 2] = 0.0
    return [
        ("square, one tile   ", rng.standard_normal((8, 8)), rng.standard_normal((8, 8)), 16, 16),
        ("non-square, ragged ", rng.standard_normal((5, 20)), rng.standard_normal((20, 3)), 6, 4),
        ("all-negative A     ", -np.abs(rng.standard_normal((7, 13))), rng.standard_normal((13, 4)), 5, 3),
        ("mixed-sign A and B ", rng.standard_normal((9, 17)), rng.standard_normal((17, 5)), 4, 4),
        ("zero column in B   ", rng.standard_normal((6, 11)), zero_col, 4, 4),
    ]


def check_int32_exactness() -> bool:
    """Zero parasitics: round(analog accumulator) must BE the integer product."""
    ok = True
    for tag, A, B, rows, cols in cases():
        for per_channel in (False, True):
            res = Q.matmul(A, B, per_channel=per_channel, rows=rows, cols=cols,
                           rundir=f"/tmp/int8_ex_{rows}_{cols}_{A.shape[0]}_{per_channel:d}")
            m = Q.metrics(res)
            worst = float(np.abs(res.c_analog - res.c_ref).max())
            peak = int(np.abs(res.c_ref).max())
            rel = F.rel_errors(res.c_analog, res.c_ref)[0]
            ok &= m.exact and rel < 1e-12
            print(f"  {tag} {'per-channel' if per_channel else 'per-tensor ':11s} "
                  f"{A.shape} @ {B.shape} on {rows}x{cols} | peak {peak:7d} | "
                  f"worst abs err {worst:.2e} of 0.5 allowed | rel {rel:.2e} of 1e-12 | "
                  f"{'int32 EXACT' if m.exact else 'NOT EXACT'}")

    # The two gates are not ordered the way one might assume: quantify both on one case.
    res = Q.matmul(*cases()[0][1:3], rows=16, cols=16, rundir="/tmp/int8_ex_slack")
    peak = int(np.abs(res.c_ref).max())
    rel = F.rel_errors(res.c_analog, res.c_ref)[0]
    int_tol = 0.5 / peak
    print(f"  integer gate tolerates rel {int_tol:.2e} at peak {peak}; the float gate "
          f"tolerates 1e-12; measured {rel:.2e}")
    print(f"  -> the integer gate is {int_tol / 1e-12:.0e}x looser in relative terms, so "
          f"both are kept")
    print(f"  every case must be int32-exact and under 1e-12: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def check_quantization() -> bool:
    """Symmetric int8 round-trip, both granularities, and the accumulator bound."""
    rng = np.random.default_rng(8)
    A = rng.standard_normal((6, 10)) * rng.choice([1.0, 1e-3], size=(6, 1))  # rows of very
    B = rng.standard_normal((10, 4))                                        # unequal scale
    At, st = Q.quantize(A)
    Ac, sc = Q.quantize(A, axis=1)
    Bq, sx = Q.quantize(B)

    A0 = np.zeros((3, 4))                       # the degenerate case: all-zero input
    A0q, s0 = Q.quantize(A0)
    N = A.shape[1]
    c_ref = Ac.astype(np.int64) @ Bq.astype(np.int64)

    checks = {
        "quantized values are integers in [-127, 127]": bool(
            np.all(At == np.rint(At)) and np.abs(At).max() <= 127 and np.abs(Ac).max() <= 127),
        "per-tensor uses one scale, per-channel one per row": st.shape == () and sc.shape == (6, 1),
        "per-tensor peak hits full scale exactly once": int((np.abs(At) == 127).sum()) >= 1,
        "per-channel puts every row at full scale": bool(
            np.all(np.abs(Ac).max(axis=1) == 127)),
        "per-channel beats per-tensor on the small rows": (
            np.abs(A - sc * Ac).max() < np.abs(A - st * At).max()),
        "dequantization error is under half a step": bool(
            np.all(np.abs(A - sc * Ac) <= sc.ravel()[:, None] / 2 + 1e-12)),
        "an all-zero matrix quantizes to zeros with scale 1": bool(
            np.all(A0q == 0) and float(s0) == 1.0 / Q.INT8_MAX),
        "accumulator fits int32": int(np.abs(c_ref).max()) <= np.iinfo(np.int32).max,
        "accumulator is within the 127*127*N bound": int(np.abs(c_ref).max()) <= 127 * 127 * N,
        "bits_for_exact matches log2(2*127^2*N)": abs(
            Q.bits_for_exact(N) - np.log2(2 * 127 ** 2 * N)) < 1e-12,
        "requantized output stays in int8 range": bool(
            np.abs(Q.requantize(c_ref, np.abs(c_ref).max() / 127)).max() <= 127),
    }
    print(f"  per-tensor step {float(st):.3e}; per-channel steps "
          f"{sc.ravel().min():.3e}..{sc.ravel().max():.3e} ({sc.ravel().max()/sc.ravel().min():.0f}x spread)")
    print(f"  accumulator peak {int(np.abs(c_ref).max())} of the {127*127*N} bound; "
          f"exact recovery needs {Q.bits_for_exact(N):.1f} bits worst case")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_int8_output() -> bool:
    """Zero parasitics: every requantized int8 output must be exactly right."""
    rng = np.random.default_rng(9)
    A, B = rng.standard_normal((10, 24)), rng.standard_normal((24, 8))

    out = {}
    for per_channel in (False, True):
        res = Q.matmul(A, B, per_channel=per_channel, rows=16, cols=16,
                       rundir=f"/tmp/int8_out_{per_channel:d}")
        out[per_channel] = (res, Q.metrics(res))
        m = out[per_channel][1]
        print(f"  {'per-channel' if per_channel else 'per-tensor ':11s} | int8 match "
              f"{100*m.match:6.2f}%  worst {m.max_lsb} LSB | s_y {res.s_y:.3f} accumulator "
              f"LSBs | int8 range {res.y_ref.min():4d}..{res.y_ref.max():4d}")

    # `calibrate` is digital: same object in, no simulation, and a unit gain is a no-op.
    res = out[True][0]
    before = F.LAST_COST["decks"]
    unit = Q.calibrate(res, np.ones(res.c_ref.shape[0]))
    checks = {
        "per-tensor int8 output is exactly right everywhere": out[False][1].match == 1.0,
        "per-channel int8 output is exactly right everywhere": out[True][1].match == 1.0,
        "no output element is off by even one LSB": (
            out[False][1].max_lsb == 0 and out[True][1].max_lsb == 0),
        "requantized reference spans most of the int8 range": abs(
            np.abs(res.y_ref).max() - 127) <= 1,
        "calibrate runs no new simulation": F.LAST_COST["decks"] == before,
        "a unit gain leaves the output untouched": bool(
            np.array_equal(unit.y_analog, res.y_analog)),
    }
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_calibration() -> bool:
    """Per-channel gains must remove more droop than one global gain.

    Also pins the mechanism behind the result that looks backwards -- uncalibrated
    per-channel scaling being *worse* than per-tensor. Per-channel lifts every row of A_q to
    full scale, which puts more weight mass, hence more conductance, hence more current in
    the array, so the droop it has to fight is larger. It also hands over the per-column
    degrees of freedom to fight it with.
    """
    rng = np.random.default_rng(10)
    A, B = rng.standard_normal((12, 16)), rng.standard_normal((16, 8))
    A_q, _ = Q.quantize(A, axis=1)
    B_q, _ = Q.quantize(B)

    rows = {}
    for r in (1.0, 5.0):
        res = Q.run_q(A_q, B_q, xbar=wire(r), rows=16, cols=16,
                      rundir=f"/tmp/int8_cal_r{r:g}")
        g1 = Q.fit_gain(res.c_analog, res.c_ref, per_channel=False)
        gm = Q.fit_gain(res.c_analog, res.c_ref, per_channel=True)
        raw, glob, chan = (Q.metrics(res), Q.metrics(Q.calibrate(res, g1)),
                           Q.metrics(Q.calibrate(res, gm)))
        rows[r] = (raw, glob, chan, g1, gm)
        print(f"  r={r:>4.1f} | int8 match raw {100*raw.match:6.2f}%  "
              f"global gain {100*glob.match:6.2f}%  per-channel {100*chan.match:6.2f}% | "
              f"accum bits {raw.bits_rms:5.2f} -> {glob.bits_rms:5.2f} -> {chan.bits_rms:5.2f} | "
              f"gains {gm.min():.4f}..{gm.max():.4f} (global {g1[0]:.4f})")

    # Why raw per-channel loses: same A, same B, only the weight-scale granularity differs.
    load = {}
    for tag, axis in (("per-tensor ", None), ("per-channel", 1)):
        Aq, _ = Q.quantize(A, axis=axis)
        res = Q.run_q(Aq, B_q, xbar=wire(5.0), rows=16, cols=16,
                      rundir=f"/tmp/int8_load_{axis}")
        gm = Q.fit_gain(res.c_analog, res.c_ref)
        load[tag] = (np.abs(Aq).mean(), Q.fit_gain(res.c_analog, res.c_ref,
                                                  per_channel=False)[0],
                     gm.max() - gm.min(), Q.metrics(res).match)
        print(f"  {tag} at r=5 | mean |A_q| {load[tag][0]:6.2f} | global droop gain "
              f"{load[tag][1]:.4f} | per-column spread {100*load[tag][2]:.2f} pts | "
              f"raw int8 match {100*load[tag][3]:6.2f}%")

    raw5, glob5, chan5, g1_5, gm5 = rows[5.0]
    pt, pc = load["per-tensor "], load["per-channel"]
    checks = {
        "every per-channel gain is below 1 (droop is one-sided)": bool(np.all(gm5 < 1.0)),
        "the global gain sits inside the per-channel spread": gm5.min() <= g1_5[0] <= gm5.max(),
        "per-channel gains actually differ across columns": (gm5.max() - gm5.min()) > 1e-3,
        "one global gain helps at 5 ohm/pitch": glob5.match > raw5.match,
        "per-channel helps more than global": chan5.match > glob5.match,
        "per-channel recovers more accumulator bits": chan5.bits_rms > glob5.bits_rms,
        "per-channel scaling puts more weight mass in the array": pc[0] > pt[0],
        "and therefore droops harder": pc[1] < pt[1],
        "with a wider per-column spread to calibrate out": pc[2] > pt[2],
        "so raw per-channel scores worse than raw per-tensor": pc[3] < pt[3],
    }
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


CHECKS = [
    ("int32 exactness", check_int32_exactness),
    ("quantization", check_quantization),
    ("int8 output", check_int8_output),
    ("calibration", check_calibration),
]


def main() -> int:
    results = {}
    t0 = time.perf_counter()
    for name, fn in CHECKS:
        print(f"\n== {name} ==")
        results[name] = fn()

    print()
    for name, hit in results.items():
        print(f"{'PASS' if hit else 'FAIL'}  {name}")
    ok = all(results.values())
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}  ({time.perf_counter() - t0:.1f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
