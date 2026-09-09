"""Regenerate docs/figures/matmul-accuracy.png from real ngspice runs.

Every number plotted and printed here comes out of a simulation this script runs. Run it
from the repo root:

    uv run scripts/matmul_accuracy.py

Four panels, four questions:

    1  how does the decoded product degrade with wire resistance, as max and rms error
       relative to numpy float64;
    2  the same, converted to effective mantissa bits (-log2 of the relative error) and
       read against FP32 / FP16 / bfloat16;
    3  0T1R against 1T1R -- the placeholder model and the real sky130 nfet_01v8 -- at
       zero wire R, so the access device's Ron compression is the only mechanism;
    4  per-block block-floating-point scaling against one global scale factor.

Plot style, the colour-blind-safe palette and `save` are imported from `make_figures`;
`block_fp_case` is imported from `verify_matmul` so the figure cannot drift away from the
`block fp` check. `matmul_accuracy` never builds a circuit itself -- `fp_matmul` does.

Panel 3 needs a real sky130 model library, so the script probes for one up front and
fails with `sky130_root()`'s actionable message rather than writing a partial figure.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crossbar as C
import fp_matmul as F
from make_figures import BLUE, GREEN, INK, VERMILLION, save, setup_style
from verify_matmul import BFP_DECADES, block_fp_case

import matplotlib.pyplot as plt

R_SWEEP = (0.0, 0.5, 1.0, 2.0, 5.0)   # ohm per cell pitch
TILE = (16, 16)                        # rows x cols, i.e. a 16 x 32 physical array
SHAPE = (16, 64, 8)                    # A (M,N) @ B (N,K) for the wire sweep
CELL_TILE = (8, 8)                     # the 1T1R panel: 8 x 16 physical, 128 FETs/instance
CELL_SHAPE = (8, 16, 4)
FORMATS = {"FP32 mantissa": 24, "FP16 mantissa": 11, "bfloat16 mantissa": 8}
TOP_BITS = 30                          # y limit on both bit axes; exact cases run past it


def matrices(shape, seed=0):
    """i.i.d. standard-normal A and B. Note this is the *optimistic* weight structure for
    the 1T1R panel -- see REPORT.md 5.3, where structure is what breaks the gain fit."""
    M, N, K = shape
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N)), rng.standard_normal((N, K))


def after_gain(got, ref):
    """(best-fit scalar gain a, (max, rms) norm-relative error after dividing it out).

    `crossbar.error_metrics` gives `a = (got.ideal)/(ideal.ideal)` in closed form. Only
    that field is used: its percentage fields are elementwise, and elementwise relative
    error is meaningless on a signed matmul output whose entries nearly cancel.
    """
    a = C.error_metrics(got.ravel(), ref.ravel())[1]
    return a, F.rel_errors(got / a, ref)


def run(A, B, xbar, tile, tag):
    """One matmul, with its cost. Returns (result, wall seconds, decks, solves)."""
    t0 = time.perf_counter()
    got = F.matmul(A, B, xbar=xbar, rows=tile[0], cols=tile[1], rundir=f"/tmp/fpmm_acc_{tag}")
    return got, time.perf_counter() - t0, F.LAST_COST["decks"], F.LAST_COST["solves"]


# --------------------------------------------------------------------------
# 1 + 2: wire resistance, as relative error and as surviving mantissa bits.
# --------------------------------------------------------------------------
def sweep_wire(A, B):
    ref = A @ B
    rows = []
    print(f"  A {A.shape} @ B {B.shape} on a {TILE[0]}x{TILE[1]} tile "
          f"({TILE[0]}x{2*TILE[1]} physical), 0T1R, unipolar two-pass, batched")
    for r in R_SWEEP:
        got, wall, decks, solves = run(A, B, replace(F.IDEAL_XBAR, r_row=r, r_col=r),
                                       TILE, f"r{r:g}")
        mx, rms = F.rel_errors(got, ref)
        a, (mx_a, rms_a) = after_gain(got, ref)
        rows.append((r, mx, rms, a, mx_a, rms_a))
        print(f"  r={r:>4.1f} ohm/pitch | max rel {mx:.3e} ({F.effective_bits(mx):5.2f} bits)"
              f"  rms rel {rms:.3e} ({F.effective_bits(rms):5.2f} bits)"
              f" | gain a {a:.6f} -> max {mx_a:.3e} ({F.effective_bits(mx_a):5.2f} bits)"
              f"  rms {rms_a:.3e} ({F.effective_bits(rms_a):5.2f} bits)"
              f" | {decks} decks, {solves} solves, {wall:.2f} s")
    return np.array(rows)


def fig_wire_error(ax, w):
    r = w[:, 0]
    ax.semilogy(r, w[:, 1], "s--", color=VERMILLION, lw=1.3, ms=4.5, label="max, raw")
    ax.semilogy(r, w[:, 2], "o-", color=BLUE, lw=1.8, ms=4.5, label="rms, raw")
    ax.semilogy(r, w[:, 5], "D-", color=GREEN, lw=1.8, ms=4,
                label="rms, after one gain fit")
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("error relative to numpy float64  [-]")
    ax.set_title("Error vs wire resistance: exact at r = 0, percent-level by 1 $\\Omega$")
    ax.legend(loc="lower right")
    ax.annotate(f"r = 0: {w[0, 1]:.1e}\n(float64 solver floor)", xy=(r[0], w[0, 1]),
                xytext=(14, 4), textcoords="offset points", fontsize=8.5, color=INK)


def fig_wire_bits(ax, w):
    r = w[:, 0]
    bits = np.vectorize(F.effective_bits)
    ax.plot(r, bits(w[:, 1]), "s--", color=VERMILLION, lw=1.3, ms=4.5, label="max, raw")
    ax.plot(r, bits(w[:, 2]), "o-", color=BLUE, lw=1.8, ms=4.5, label="rms, raw")
    ax.plot(r, bits(w[:, 5]), "D-", color=GREEN, lw=1.8, ms=4,
            label="rms, after one gain fit")
    for (name, nbits), style in zip(FORMATS.items(), (":", "--", "-.")):
        ax.axhline(nbits, color=INK, lw=1.0, ls=style)
        ax.annotate(f"{name} = {nbits} bits", xy=(r[-1], nbits), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.set_ylim(0, TOP_BITS)
    ax.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax.set_ylabel("effective mantissa bits, $-\\log_2$(rel. error)  [-]")
    ax.set_title("Surviving precision: nothing but the exact case clears FP16")
    ax.legend(loc="center right")
    ax.annotate(f"r = 0 runs off the top at {bits(w[0, 2]):.0f} bits\n"
                "(the float64 solver floor, not a circuit)",
                xy=(r[0], TOP_BITS), xytext=(30, -20), textcoords="offset points",
                fontsize=8.5, color=INK)


# --------------------------------------------------------------------------
# 3: the access device.  Zero wire R, so Ron is the only mechanism in play.
# --------------------------------------------------------------------------
def sweep_cell(A, B):
    ref = A @ B
    # The 0T1R-at-5-ohm bar is the reason this panel runs its own wire case rather than
    # borrowing a number from `sweep_wire`: same tile, same matrices, so "Ron costs more
    # than 5 ohm/pitch of wire" is a comparison and not an insinuation.
    routes = (
        ("passive", "0T1R\nzero wire R", C.CellParams(passive=True), 0.0),
        ("passive5", "0T1R\nat 5 $\\Omega$/pitch", C.CellParams(passive=True), 5.0),
        ("generic", "1T1R generic\nzero wire R", C.CellParams(), 0.0),
        ("sky130", "1T1R sky130\nzero wire R", C.CellParams(access="sky130"), 0.0),
    )
    print(f"  A {A.shape} @ B {B.shape} on a {CELL_TILE[0]}x{CELL_TILE[1]} tile "
          f"({CELL_TILE[0]}x{2*CELL_TILE[1]} physical)")
    out = {}
    for tag, label, cell, r in routes:
        got, wall, decks, solves = run(
            A, B, replace(F.IDEAL_XBAR, cell=cell, r_row=r, r_col=r), CELL_TILE, tag)
        mx, rms = F.rel_errors(got, ref)
        a, (mx_a, rms_a) = after_gain(got, ref)
        out[label] = (mx, rms, a, mx_a, rms_a)
        print(f"  {tag:9s} r={r:>3.1f} ohm/pitch | max rel {mx:.3e} "
              f"({F.effective_bits(mx):5.2f} bits)  rms rel {rms:.3e} "
              f"({F.effective_bits(rms):5.2f} bits) | gain a {a:.6f} -> rms {rms_a:.3e} "
              f"({F.effective_bits(rms_a):5.2f} bits) | {decks} decks, {solves} solves, "
              f"{wall:.1f} s")
    return out


def fig_cell(ax, cells):
    tags = list(cells)
    x = np.arange(len(tags))
    bits = [F.effective_bits(cells[t][1]) for t in tags]
    bits_a = [F.effective_bits(cells[t][4]) for t in tags]
    ax.bar(x - 0.19, bits, 0.36, color=VERMILLION, label="rms, raw")
    ax.bar(x + 0.19, bits_a, 0.36, color=GREEN, label="rms, after one gain fit")
    for xi, (b, ba) in enumerate(zip(bits, bits_a)):
        for dx, v in ((-0.19, b), (0.19, ba)):
            off = v > TOP_BITS   # clipped bar: label inside it, in white, with a marker
            ax.annotate(f"{v:.1f}" + ("*" if off else ""),
                        xy=(xi + dx, min(v, TOP_BITS)), xytext=(0, -13 if off else 2),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color="white" if off else INK)
    for name, nbits in FORMATS.items():
        ax.axhline(nbits, color=INK, lw=1.0, ls=":")
        ax.annotate(name.split()[0], xy=(len(tags) - 0.55, nbits), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.set_xticks(x, tags, fontsize=8.5)
    ax.set_ylim(0, TOP_BITS)
    ax.set_ylabel("effective mantissa bits  [-]")
    ax.set_title("Series Ron costs more precision than 5 $\\Omega$/pitch of wire\n"
                 "(same tile and same matrices for all four bars; * = clipped)")
    ax.legend(loc="upper right")


# --------------------------------------------------------------------------
# 4: block floating point.  Same case the `block fp` check asserts.
# --------------------------------------------------------------------------
def sweep_block_fp():
    A, B, rows, cols = block_fp_case()
    ref = A @ B
    per = F.matmul(A, B, rows=rows, cols=cols, rundir="/tmp/fpmm_acc_bfp_per")
    glob = F.matmul(A, B, rows=rows, cols=cols, global_scale=True,
                    rundir="/tmp/fpmm_acc_bfp_glob")
    print(f"  A {A.shape} @ B {B.shape} on a {rows}x{cols} tile, output blocks spanning "
          f"1e-0 .. 1e-{BFP_DECADES[-1]}")
    out = []
    for b, d in enumerate(BFP_DECADES):
        s = slice(b * cols, (b + 1) * cols)
        mp, mg = F.rel_errors(per[s], ref[s])[0], F.rel_errors(glob[s], ref[s])[0]
        out.append((d, mp, mg))
        print(f"  block {b} at 1e-{d:<3d} | per-block sA {mp:.3e} "
              f"({F.effective_bits(mp):5.2f} bits) | one global sA {mg:.3e} "
              f"({F.effective_bits(mg):5.2f} bits)")
    return np.array(out)


def fig_block_fp(ax, bfp):
    d, mp, mg = bfp[:, 0], bfp[:, 1], bfp[:, 2]
    bits = np.vectorize(F.effective_bits)
    ax.plot(d, bits(mp), "o-", color=BLUE, lw=1.8, ms=5,
            label="per-block $s_A$ (block floating point)")
    ax.plot(d, bits(mg), "s--", color=VERMILLION, lw=1.5, ms=5, label="one global $s_A$")
    for name, nbits in FORMATS.items():
        ax.axhline(nbits, color=INK, lw=1.0, ls=":")
        ax.annotate(name.split()[0], xy=(d[-1], nbits), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.set_xticks(d, ["$1$"] + [f"$10^{{-{int(v)}}}$" for v in d[1:]])
    ax.set_xlabel("magnitude of this output block of A, relative to the largest  [-]")
    ax.set_ylabel("effective mantissa bits  [-]")
    ax.set_title("Block floating point: a per-block exponent is the dynamic range\n"
                 f"(the small block loses {bits(mp)[-1] - bits(mg)[-1]:.0f} bits "
                 "to a single global scale)")
    ax.legend(loc="center left")


def main():
    setup_style()
    C.sky130_root()   # fail fast, with the actionable message, if no PDK is installed
    figures = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print("== relative error vs wire resistance ==")
    w = sweep_wire(*matrices(SHAPE))

    print("\n== 0T1R vs 1T1R ==")
    cells = sweep_cell(*matrices(CELL_SHAPE, seed=1))

    print("\n== block floating point vs one global scale ==")
    bfp = sweep_block_fp()

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig_wire_error(ax[0, 0], w)
    fig_wire_bits(ax[0, 1], w)
    fig_cell(ax[1, 0], cells)
    fig_block_fp(ax[1, 1], bfp)
    fig.suptitle("Floating-point matmul on the crossbar: what survives the analog path\n"
                 "error is relative to numpy float64 A@B; bits are $-\\log_2$ of it",
                 y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save(fig, figures / "matmul-accuracy.png")

    print(f"\ntotal wall clock {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
