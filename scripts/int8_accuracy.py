"""Regenerate docs/figures/int8-accuracy.png from real ngspice runs.

Every number plotted and printed here comes out of a simulation this script runs. Run it
from the repo root:

    uv run scripts/int8_accuracy.py

Six panels:

    1  requantized int8 output match rate vs wire resistance, for four quantization and
       calibration schemes;
    2  the same runs' worst output deviation, in int8 LSBs -- a 1-LSB miss and a 10-LSB
       miss are different failures;
    3  accumulator bits recovered vs the bits exact int32 recovery would need;
    4  array height: match rate vs the number of rows accumulated in the analog domain,
       which is the design limit `r_wire` alone does not give you;
    5  0T1R against 1T1R (generic placeholder and real sky130 nfet_01v8), with and without
       per-channel calibration -- Ron compression is a per-cell nonlinearity, so a
       per-column gain should not absorb it as cleanly as IR droop;
    6  calibration transfer: per-channel gains fitted on one activation batch, applied to
       a different one. In-sample calibration numbers are not a deployable claim.

Plot style, the palette and `save` come from `make_figures`; the circuit comes from
`crossbar` via `fp_matmul`; the quantization comes from `int8_matmul`. Nothing is
duplicated here.

Panel 5 needs a real sky130 model library, so the script probes for one up front and fails
with `sky130_root()`'s actionable message rather than writing a partial figure.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crossbar as C
import fp_matmul as F
import int8_matmul as Q
from make_figures import BLUE, GREEN, INK, PURPLE, VERMILLION, save, setup_style

import matplotlib.pyplot as plt

R_SWEEP = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)   # ohm per cell pitch
TILE = (16, 16)                              # rows x cols, i.e. a 16 x 32 physical array
SHAPE = (16, 32, 8)                          # A (M,N) @ B (N,K) for the wire sweep
N_SWEEP = (4, 8, 16, 32, 64)                 # analog-accumulated rows; tile is N x 8
N_SHAPE = (8, 8)                             # M, K for the height sweep
N_COLS = 8
N_R = 0.5                                    # ohm/pitch for the height sweep
CELL_TILE = (8, 8)                           # the 1T1R panel: 8 x 16 physical
CELL_SHAPE = (8, 16, 4)

# Four schemes. `chan` is the one the brief predicts wins: per-channel weight scales carry
# one multiplier per output channel already, so the calibration is free on hardware.
SCHEMES = (
    ("per-tensor, raw", False, None, BLUE, "o-"),
    ("per-tensor + one global gain", False, False, PURPLE, "s--"),
    ("per-channel, raw", True, None, VERMILLION, "^-"),
    ("per-channel + per-channel gain", True, True, GREEN, "D-"),
)


def operands(shape, seed=0):
    """i.i.d. standard-normal A and B. One matrix pair is not an inference study, and per
    REPORT.md 5.3 i.i.d. weights are the *optimistic* structure for the 1T1R panel."""
    M, N, K = shape
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N)), rng.standard_normal((N, K))


def evaluate(A, B, xbar, tile, tag):
    """{scheme: Int8Metrics} plus the fitted gains, from one analog run per granularity.

    Two simulations per configuration, not four: calibration is a digital multiply on an
    accumulator the array already produced, so `Q.calibrate` adds no deck.
    """
    B_q, _ = Q.quantize(B)
    runs, out, gains = {}, {}, {}
    for per_channel in (False, True):
        A_q, _ = Q.quantize(A, axis=1 if per_channel else None)
        runs[per_channel] = Q.run_q(A_q, B_q, xbar=xbar, rows=tile[0], cols=tile[1],
                                    rundir=f"/tmp/int8_acc_{tag}_{per_channel:d}")
    for name, per_channel, cal, _, _ in SCHEMES:
        res = runs[per_channel]
        if cal is None:
            out[name] = Q.metrics(res)
        else:
            g = Q.fit_gain(res.c_analog, res.c_ref, per_channel=cal)
            gains[name] = g
            out[name] = Q.metrics(Q.calibrate(res, g))
    return out, gains, runs


def report(tag, out, gains):
    for name, *_ in SCHEMES:
        m = out[name]
        g = gains.get(name)
        span = f" | gains {g.min():.4f}..{g.max():.4f}" if g is not None else ""
        print(f"  {tag:>16s}  {name:31s} | int8 match {100*m.match:6.2f}%  "
              f"worst {m.max_lsb:3d} LSB | accum {m.bits_rms:5.2f} of {m.bits_needed:5.2f} "
              f"bits needed{'  EXACT int32' if m.exact else ''}{span}")


# --------------------------------------------------------------------------
# 1 + 2 + 3: wire resistance.
# --------------------------------------------------------------------------
def sweep_wire(A, B):
    print(f"  A {A.shape} @ B {B.shape} int8 on a {TILE[0]}x{TILE[1]} tile "
          f"({TILE[0]}x{2*TILE[1]} physical), 0T1R, unipolar two-pass, batched")
    print(f"  exact int32 recovery needs {Q.bits_for_exact(SHAPE[1]):.1f} bits worst case; "
          f"the int8 output needs half an LSB of 127, i.e. {100/254:.2f}% of full scale")
    rows = []
    for r in R_SWEEP:
        t0 = time.perf_counter()
        out, gains, _ = evaluate(A, B, replace(F.IDEAL_XBAR, r_row=r, r_col=r), TILE,
                                 f"r{r:g}")
        print(f"  -- r = {r:g} ohm/pitch, {F.LAST_COST['decks']} decks/run, "
              f"{time.perf_counter() - t0:.2f} s --")
        report(f"r={r:g}", out, gains)
        rows.append(out)
    return rows


def series(rows, field):
    """(len(SCHEMES), len(R_SWEEP)) of one Int8Metrics field."""
    return np.array([[getattr(row[name], field) for row in rows] for name, *_ in SCHEMES])


def fig_match(ax, rows):
    r, y = np.array(R_SWEEP), series(rows, "match") * 100
    for (name, _, _, colour, style), row in zip(SCHEMES, y):
        ax.plot(r, row, style, color=colour, lw=1.8, ms=5, label=name)
    ax.axhline(100, color=INK, lw=1.0, ls=":")
    ax.set_ylim(0, 108)
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("int8 outputs exactly correct  [%]")
    ax.set_title("Requantized int8 output fidelity\n"
                 "per-channel scales absorb the droop; raw per-channel is worse than "
                 "raw per-tensor")
    ax.legend(loc="lower left", framealpha=0.95)


def fig_lsb(ax, rows):
    r, y = np.array(R_SWEEP), series(rows, "max_lsb")
    for (name, _, _, colour, style), row in zip(SCHEMES, y):
        ax.plot(r, row, style, color=colour, lw=1.8, ms=5, label=name)
    ax.axhline(1, color=INK, lw=1.0, ls=":")
    ax.annotate("1 LSB", xy=(r[-1], 1), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=INK)
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("worst output deviation  [int8 LSBs]")
    ax.set_title("How wrong the wrong outputs are\n"
                 "per-channel calibration never misses by more than one LSB")
    ax.legend(loc="upper left", framealpha=0.95)


def fig_bits(ax, rows):
    r, y = np.array(R_SWEEP), series(rows, "bits_rms")
    for (name, _, _, colour, style), row in zip(SCHEMES, y):
        ax.plot(r, row, style, color=colour, lw=1.8, ms=5, label=name)
    need = rows[0]["per-channel, raw"].bits_needed
    worst = Q.bits_for_exact(SHAPE[1])
    ax.axhline(worst, color=INK, lw=1.2, ls="--")
    ax.annotate(f"exact int32, worst-case operands: {worst:.1f} bits", xy=(r[-1], worst),
                xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.axhline(need, color=INK, lw=1.2, ls="-.")
    ax.annotate(f"exact int32, this matrix pair: {need:.1f} bits", xy=(r[-1], need),
                xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.axhline(np.log2(254), color=VERMILLION, lw=1.2, ls=":")
    ax.annotate(f"int8 output, half an LSB: {np.log2(254):.1f} bits",
                xy=(r[-1], np.log2(254)), xytext=(-4, 4), textcoords="offset points",
                ha="right", fontsize=8.5, color=VERMILLION)
    ax.set_ylim(0, worst + 2)
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("accumulator bits recovered, rms  [-]")
    ax.set_title("Exact int32 accumulation is out of reach at every nonzero $r_{wire}$\n"
                 "int8 output fidelity is the reachable target, and only just")
    # r = 0 runs off the top at the float64 solver floor; the near-vertical segment says so
    # without a label, and every other free patch of this panel already carries a threshold.
    ax.legend(loc="lower left", framealpha=0.95, fontsize=8)


# --------------------------------------------------------------------------
# 4: array height -- how many rows can accumulate in the analog domain.
# --------------------------------------------------------------------------
def sweep_height(seed=1):
    M, K = N_SHAPE
    print(f"  A ({M},N) @ B (N,{K}) int8, tile N x {N_COLS} (N x {2*N_COLS} physical), "
          f"r = {N_R} ohm/pitch, one contraction block so all N rows sum in the analog domain")
    out = []
    for N in N_SWEEP:
        A, B = operands((M, N, K), seed=seed)
        t0 = time.perf_counter()
        got, gains, _ = evaluate(A, B, replace(F.IDEAL_XBAR, r_row=N_R, r_col=N_R),
                                 (N, N_COLS), f"n{N}")
        print(f"  -- N = {N}, {F.LAST_COST['decks']} decks/run, "
              f"{time.perf_counter() - t0:.2f} s --")
        report(f"N={N}", got, gains)
        out.append(got)
    return out


def fig_height(ax, rows):
    n = np.array(N_SWEEP)
    for name, _, _, colour, style in SCHEMES:
        ax.plot(n, [100 * row[name].match for row in rows], style, color=colour, lw=1.8,
                ms=5, label=name)
    ax.axhline(100, color=INK, lw=1.0, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(v) for v in n])
    ax.set_ylim(0, 108)
    ax.set_xlabel("rows accumulated in the analog domain, N  [-]")
    ax.set_ylabel("int8 outputs exactly correct  [%]")
    ax.set_title(f"Array height at {N_R:g} $\\Omega$/pitch: the real design limit\n"
                 "beyond this, tiling moves accumulation into exact digital adds")
    ax.legend(loc="lower left", framealpha=0.95)


# --------------------------------------------------------------------------
# 5: the access device.
# --------------------------------------------------------------------------
def sweep_cell(A, B):
    routes = (
        ("passive0", "0T1R\nzero wire R", C.CellParams(passive=True), 0.0),
        ("passive1", "0T1R\n1 $\\Omega$/pitch", C.CellParams(passive=True), 1.0),
        ("generic0", "1T1R generic\nzero wire R", C.CellParams(), 0.0),
        ("sky130_0", "1T1R sky130\nzero wire R", C.CellParams(access="sky130"), 0.0),
    )
    print(f"  A {A.shape} @ B {B.shape} int8 on a {CELL_TILE[0]}x{CELL_TILE[1]} tile "
          f"({CELL_TILE[0]}x{2*CELL_TILE[1]} physical)")
    out = {}
    for tag, label, cell, r in routes:
        t0 = time.perf_counter()
        got, gains, _ = evaluate(A, B, replace(F.IDEAL_XBAR, cell=cell, r_row=r, r_col=r),
                                 CELL_TILE, tag)
        print(f"  -- {tag}, r = {r:g} ohm/pitch, {time.perf_counter() - t0:.1f} s --")
        report(tag, got, gains)
        out[label] = got
    return out


def fig_cell(ax, cells):
    labels = list(cells)
    x = np.arange(len(labels))
    raw = [100 * cells[t]["per-channel, raw"].match for t in labels]
    cal = [100 * cells[t]["per-channel + per-channel gain"].match for t in labels]
    ax.bar(x - 0.19, raw, 0.36, color=VERMILLION, label="per-channel, raw")
    ax.bar(x + 0.19, cal, 0.36, color=GREEN, label="per-channel + per-channel gain")
    for xi, (a, b) in enumerate(zip(raw, cal)):
        for dx, v in ((-0.19, a), (0.19, b)):
            ax.annotate(f"{v:.0f}", xy=(xi + dx, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0, 134)
    ax.set_ylabel("int8 outputs exactly correct  [%]")
    ax.set_title("Access device: Ron compression is not a per-column gain\n"
                 "so a per-column calibration cannot fully remove it")
    ax.legend(loc="upper center", ncol=2, framealpha=0.95, fontsize=8.5)


# --------------------------------------------------------------------------
# 6: does the calibration transfer to activations it was not fitted on?
# --------------------------------------------------------------------------
def sweep_transfer(A, B_cal, B_same, B_relu):
    """Fit per-channel gains on one activation batch, apply them to two others.

    `B_same` is another draw from the same Gaussian: this only tests that the fit is not
    memorising one sample. `B_relu` is a *distribution shift* -- `max(gaussian, 0)`, which
    is what a real layer receives from the ReLU in front of it: non-negative, ~half zeros,
    and therefore a completely different total row current, which is what sets IR droop.
    That is the test that decides whether a factory calibration is deployable.
    """
    A_q, _ = Q.quantize(A, axis=1)
    qs = {k: Q.quantize(v)[0] for k, v in
          (("cal", B_cal), ("same", B_same), ("relu", B_relu))}
    print(f"  gains fitted on a gaussian B {B_cal.shape}, applied to another gaussian draw "
          f"and to a ReLU'd batch ({100*(B_relu == 0).mean():.0f}% zeros, non-negative)")
    out = []
    for r in R_SWEEP:
        xbar = replace(F.IDEAL_XBAR, r_row=r, r_col=r)
        res = {k: Q.run_q(A_q, q, xbar=xbar, rows=TILE[0], cols=TILE[1],
                          rundir=f"/tmp/int8_acc_tr_{k}_r{r:g}") for k, q in qs.items()}
        g = Q.fit_gain(res["cal"].c_analog, res["cal"].c_ref)
        row = [Q.metrics(res["same"]).match,
               Q.metrics(Q.calibrate(res["same"], g)).match,
               Q.metrics(res["relu"]).match,
               Q.metrics(Q.calibrate(res["relu"], g)).match]
        # In-sample refits, for the honest comparison: is transfer costing anything at all?
        ins = [Q.metrics(Q.calibrate(res[k], Q.fit_gain(res[k].c_analog, res[k].c_ref))).match
               for k in ("same", "relu")]
        out.append(row)
        print(f"  r={r:>4.2f} | gaussian raw {100*row[0]:6.2f}% -> transferred "
              f"{100*row[1]:6.2f}% (refit in-sample {100*ins[0]:6.2f}%) | ReLU raw "
              f"{100*row[2]:6.2f}% -> transferred {100*row[3]:6.2f}% (refit in-sample "
              f"{100*ins[1]:6.2f}%)")
    return np.array(out)


def fig_transfer(ax, tr):
    r = np.array(R_SWEEP)
    ax.plot(r, 100 * tr[:, 0], "^-", color=VERMILLION, lw=1.6, ms=5,
            label="gaussian batch, raw")
    ax.plot(r, 100 * tr[:, 1], "o--", color=BLUE, lw=1.8, ms=5,
            label="gaussian batch, transferred gains")
    ax.plot(r, 100 * tr[:, 2], "v:", color=PURPLE, lw=1.6, ms=5,
            label="ReLU batch, raw")
    ax.plot(r, 100 * tr[:, 3], "D-", color=GREEN, lw=1.8, ms=5,
            label="ReLU batch, transferred gains")
    ax.axhline(100, color=INK, lw=1.0, ls=":")
    ax.set_ylim(0, 108)
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("int8 outputs exactly correct  [%]")
    ax.set_title("Calibration transfer: gains fitted on one activation batch,\n"
                 "applied to another draw and to a shifted (ReLU) distribution")
    ax.legend(loc="lower left", framealpha=0.95, fontsize=8)


def main():
    setup_style()
    C.sky130_root()   # fail fast, with the actionable message, if no PDK is installed
    figures = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    A, B = operands(SHAPE)
    print("== int8 fidelity vs wire resistance ==")
    wires = sweep_wire(A, B)

    print("\n== array height ==")
    heights = sweep_height()

    print("\n== 0T1R vs 1T1R ==")
    cells = sweep_cell(*operands(CELL_SHAPE, seed=3))

    print("\n== calibration transfer ==")
    tr = sweep_transfer(A, B, operands(SHAPE, seed=2)[1],
                        np.maximum(operands(SHAPE, seed=4)[1], 0.0))

    fig, ax = plt.subplots(2, 3, figsize=(18.5, 9.8))
    fig_match(ax[0, 0], wires)
    fig_lsb(ax[0, 1], wires)
    fig_bits(ax[0, 2], wires)
    fig_height(ax[1, 0], heights)
    fig_cell(ax[1, 1], cells)
    fig_transfer(ax[1, 2], tr)
    fig.suptitle("int8 matmul on the analog crossbar: symmetric int8 weights and "
                 "activations, exact int32 reference, int8 requantized output\n"
                 "\"exactly correct\" means the int8 value equals the one the integer "
                 "reference produces -- not within a tolerance", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save(fig, figures / "int8-accuracy.png")

    print(f"\ntotal wall clock {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
