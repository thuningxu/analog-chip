"""Regenerate docs/figures/*.png from real ngspice runs.

Every number plotted here comes out of a simulation this script runs; nothing is
sketched, interpolated or hand-tuned. Run it from the repo root:

    uv run scripts/make_figures.py

Circuit construction is imported from `crossbar`, never duplicated. `demo_case` and
`implied_ron` are imported from `verify_mna` for the same reason: they define the exact
(W, x) pair and the exact Ron extraction that the verification checks and the README
numbers use, so the figures cannot drift away from the values in the docs.

`ron-compression.png` needs a real sky130 model library, so the script probes for one
up front and fails with `sky130_root()`'s actionable message rather than writing three
figures and then dying on the fourth.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or over ssh
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crossbar as C
from verify_mna import demo_case, implied_ron

# Okabe-Ito, colour-blind safe. All five are dark enough to read against the opaque
# white face colour set in `setup_style`, which is what keeps the PNGs legible on
# GitHub's dark theme -- a transparent background with default black text would not be.
BLUE, ORANGE, GREEN, VERMILLION, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
INK = "#333333"
# Where series are told apart by colour alone, use these three: they separate in hue and
# in lightness, so ORANGE next to VERMILLION -- which are close -- never comes up.
SERIES3 = (BLUE, GREEN, VERMILLION)

R_SWEEP = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)   # ohm per cell pitch
WEIGHT_SWEEP = np.linspace(0.0, 1.0, 21)
C_SWEEP = (0.2e-15, 0.5e-15, 2.0e-15)             # F per cell pitch
V_READ = 0.2                                       # V, the top of the demo input range
N_ARRAY = 16                                       # rows/cols for the sky130 array cases


def setup_style():
    """Opaque white everywhere, so a dark-theme reader still sees black-on-white."""
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    })


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------
# 0T1R wire IR drop: one sweep feeds both the error figure and the gain-fit
# decomposition, so the two figures cannot disagree about the same run.
# --------------------------------------------------------------------------
def sweep_wire_r(W, x):
    """Per-column relative error and the `error_metrics` split at every r_wire."""
    rel, metrics = [], []
    for r in R_SWEEP:
        p = C.CrossbarParams(weights=W, r_row=r, r_col=r, cell=C.CellParams(passive=True))
        spice, ideal = C.run_mac(p, x, rundir=f"/tmp/xbar_fig_r{r:g}")
        rel.append(100 * (spice - ideal) / ideal)
        metrics.append(C.error_metrics(spice, ideal))
        droop, a, worst, rms, worst_res = metrics[-1]
        print(f"  r={r:>5.1f} ohm/pitch | droop {droop:8.4f}%  worst {worst:8.4f}%  "
              f"a {a:.4f} | residual rms {rms:6.3f}%  worst {worst_res:6.3f}% "
              f"| least-negative column {rel[-1].max():+.3e}%")
    return np.array(rel), np.array(metrics)


def fig_ir_drop(rel, path):
    r = np.array(R_SWEEP)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.fill_between(r, rel.min(axis=1), rel.max(axis=1), color=BLUE, alpha=0.18,
                     label=f"spread across all {rel.shape[1]} columns")
    ax0.plot(r, rel.mean(axis=1), "o-", color=BLUE, lw=1.8, ms=4.5, label="mean over columns")
    ax0.plot(r, rel.min(axis=1), "s--", color=VERMILLION, lw=1.3, ms=4, label="worst column")
    ax0.axhline(0, color=INK, lw=1.1)
    ax0.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax0.set_ylabel("MAC error vs zero-parasitic ideal  [%]")
    ax0.set_title("IR drop is strictly one-sided: no column ever gains current")
    ax0.legend(loc="lower left")

    for rv, colour, mark in zip((1.0, 5.0, 20.0), SERIES3, ("o", "s", "^")):
        k = R_SWEEP.index(rv)
        ax1.plot(np.arange(rel.shape[1]), rel[k], mark + "-", color=colour, lw=1.4, ms=3.5,
                 label=f"$r_{{wire}}$ = {rv:g} $\\Omega$/pitch")
    ax1.axhline(0, color=INK, lw=1.1)
    ax1.set_xlabel("column index j  [-]")
    ax1.set_ylabel("MAC error vs zero-parasitic ideal  [%]")
    ax1.set_title("Per-column error: the far columns droop hardest")
    ax1.legend(loc="lower left")

    fig.suptitle(f"0T1R wire IR drop, {rel.shape[1]}x{rel.shape[1]} array, "
                 "numpy.default_rng(0) weights, x = rng.random(32) * 0.2 V", y=1.02)
    save(fig, path)


def fig_gain_fit(metrics, path):
    r = np.array(R_SWEEP)
    droop, a, worst, rms, worst_res = (metrics[:, k] for k in range(5))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.plot(r, np.abs(worst), "s--", color=VERMILLION, lw=1.3, ms=4,
             label="worst column, before gain fit")
    ax0.plot(r, np.abs(droop), "o-", color=BLUE, lw=1.8, ms=4.5,
             label="mean droop, before gain fit")
    ax0.plot(r, worst_res, "^--", color=PURPLE, lw=1.3, ms=4,
             label="worst column, after gain fit")
    ax0.plot(r, rms, "D-", color=GREEN, lw=1.8, ms=4,
             label="rms residual, after gain fit")
    ax0.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax0.set_ylabel("magnitude of MAC error  [%]")
    ax0.set_title("Removing one scalar gain removes most of the error")
    ax0.legend(loc="upper left")

    ax1.plot(r, a, "o-", color=BLUE, lw=1.8, ms=4.5)
    ax1.axhline(1.0, color=INK, lw=1.1)
    ax1.set_xlabel("wire resistance per cell pitch  [$\\Omega$]")
    ax1.set_ylabel("best-fit scalar gain a  [-]")
    ax1.set_title("The calibratable part: a = (spice$\\cdot$ideal) / (ideal$\\cdot$ideal)")
    for rv in (1.0, 5.0, 20.0):
        k = R_SWEEP.index(rv)
        # the r=20 point sits on the right spine, so its label has to open leftwards
        dx, ha = (-8, "right") if rv == 20.0 else (8, "left")
        ax1.annotate(f"a = {a[k]:.4f}", xy=(r[k], a[k]), xytext=(dx, 8),
                     textcoords="offset points", ha=ha, fontsize=8.5, color=INK)

    fig.suptitle("Gain droop vs non-calibratable residual, 32x32 0T1R "
                 "(same runs as ir-drop-vs-rwire.png)", y=1.02)
    save(fig, path)


# --------------------------------------------------------------------------
# Access-FET Ron: a per-cell compression curve, plus the array-level consequence,
# which depends on weight structure rather than on Ron alone.
# --------------------------------------------------------------------------
def cell_compression():
    """Effective vs programmed cell conductance across the weight range, on sky130.

    One row and zero wire R makes every column an independent cell held at the same
    read voltage, so a single .op sweeps the whole weight range: G_eff[j] = I[j] / V.
    The passive run is the control -- it must return the programmed conductance
    exactly, which is what proves the deviation is the access FET and not the wires.
    """
    W = (tuple(float(v) for v in WEIGHT_SWEEP),)
    base = dict(weights=W, r_row=0.0, r_col=0.0)
    with warnings.catch_warnings():
        # run_mac warns that its 0T1R reference is invalid for 1T1R. Expected: the
        # reference is discarded here, only the measured current is used.
        warnings.simplefilter("ignore")
        i_1t1r, _ = C.run_mac(C.CrossbarParams(**base, cell=C.CellParams(access="sky130")),
                              (V_READ,), rundir="/tmp/xbar_fig_cmp_sky")
    i_0t1r, _ = C.run_mac(C.CrossbarParams(**base, cell=C.CellParams(passive=True)),
                          (V_READ,), rundir="/tmp/xbar_fig_cmp_pas")

    p = C.CrossbarParams(**base)
    g_prog = p.g_min + WEIGHT_SWEEP * (p.g_max - p.g_min)
    g_1t1r, g_0t1r = i_1t1r / V_READ, i_0t1r / V_READ
    print(f"  0T1R control: max deviation from the programmed conductance "
          f"{np.abs(g_0t1r / g_prog - 1).max():.2e} (want ~0)")
    print("     w   g_prog [uS]  g_1T1R [uS]  compression [%]  implied Ron [ohm]")
    for k in range(0, len(WEIGHT_SWEEP), 4):
        print(f"  {WEIGHT_SWEEP[k]:4.2f} {g_prog[k]*1e6:11.3f} {g_1t1r[k]*1e6:12.4f} "
              f"{100*(1-g_1t1r[k]/g_prog[k]):15.3f} {1/g_1t1r[k]-1/g_prog[k]:17.1f}")
    return p, g_prog, g_0t1r, g_1t1r


def array_compression(ron):
    """Per-column 1T1R error vs the 0T1R ideal, for two weight structures.

    Same array, same bias, same devices: only the weight *structure* differs. With
    unstructured weights each column averages over the compression curve and one
    scalar gain nearly absorbs it; with graded columns each column sits at its own
    point on the curve and the same fit removes almost nothing.
    """
    graded = (tuple(tuple(j / (N_ARRAY - 1) for j in range(N_ARRAY)) for _ in range(N_ARRAY)),
              tuple(0.1 for _ in range(N_ARRAY)))
    out = {}
    for tag, (W, x) in {"uniform random": demo_case(n=N_ARRAY), "graded columns": graded}.items():
        p = C.CrossbarParams(weights=W, r_row=0.0, r_col=0.0,
                             cell=C.CellParams(access="sky130"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # see cell_compression
            spice, ideal = C.run_mac(p, x, rundir=f"/tmp/xbar_fig_arr_{tag.split()[0]}")
        m0 = C.error_metrics(spice, ideal)
        m1 = C.error_metrics(spice, C.ideal_1t1r(p, x, ron))
        out[tag] = (100 * (spice - ideal) / ideal, m0, m1)
        print(f"  {tag:14s} vs 0T1R ideal   : droop {m0[0]:7.3f}%  a {m0[1]:.4f}  "
              f"residual rms {m0[3]:6.3f}%  worst {m0[4]:6.3f}%")
        print(f"  {tag:14s} vs Ron-corrected: droop {m1[0]:7.3f}%  a {m1[1]:.4f}  "
              f"residual rms {m1[3]:6.3f}%  worst {m1[4]:6.3f}%")
    return out


def fig_ron_compression(p, g_prog, g_0t1r, g_1t1r, arrays, ron, path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    gp = g_prog * 1e6
    ax0.plot(gp, 100 * (1 - g_0t1r / g_prog), "o-", color=GREEN, lw=1.5, ms=3.5,
             label="0T1R control (no access FET)")
    ax0.plot(gp, 100 * (1 - g_1t1r / g_prog), "o-", color=VERMILLION, lw=1.8, ms=4.5,
             label="sky130 1T1R, measured")
    ax0.plot(gp, 100 * (1 - 1 / (1 + ron * g_prog)), "--", color=INK, lw=1.2,
             label=f"single-Ron model, Ron = {ron:.0f} $\\Omega$")
    ax0.set_xlabel("programmed cell conductance  [$\\mu$S]")
    ax0.set_ylabel("conductance compression 1 - $G_{eff}/G$  [%]")
    # Title carries the top-axis meaning: a secondary-axis *label* would collide with it.
    ax0.set_title("Per cell: sky130 access FET squeezes high weights, not low ones\n"
                  "(top axis: the programmed weight w behind each conductance)", pad=20)
    ax0.legend(loc="upper left")
    ax0.text(0.98, 0.11, f"measured: {100*(1-g_1t1r[0]/g_prog[0]):.3f}% at "
             f"g_min = {gp[0]:g} $\\mu$S,   {100*(1-g_1t1r[-1]/g_prog[-1]):.3f}% at "
             f"g_max = {gp[-1]:g} $\\mu$S",
             transform=ax0.transAxes, ha="right", va="bottom", fontsize=8.5, color=VERMILLION)
    g_lo, g_hi = p.g_min * 1e6, p.g_max * 1e6
    ax0.secondary_xaxis("top", functions=(lambda g: (g - g_lo) / (g_hi - g_lo),
                                          lambda w: g_lo + w * (g_hi - g_lo)))

    for (tag, (rel, m0, _)), colour in zip(arrays.items(), (BLUE, VERMILLION)):
        ax1.plot(np.arange(len(rel)), rel, "o-", color=colour, lw=1.4, ms=3.5,
                 label=f"{tag} (best-fit gain a = {m0[1]:.4f})")
        ax1.axhline(100 * (m0[1] - 1), color=colour, ls="--", lw=1.1)
    ax1.set_xlabel("column index j  [-]")
    ax1.set_ylabel("MAC error vs 0T1R ideal  [%]")
    ax1.set_title(f"Per array: whether one gain absorbs it is set by weight structure\n"
                  f"({N_ARRAY}x{N_ARRAY}, zero wire R; dashed = the gain each fit found)")
    ax1.legend(loc="lower left", framealpha=0.95)
    resid = "\n".join(f"{tag}: {m0[3]:.2f}% rms, {m0[4]:.2f}% worst"
                      for tag, (_, m0, _) in arrays.items())
    ax1.text(0.98, 0.97, "residual left after the gain fit\n" + resid,
             transform=ax1.transAxes, ha="right", va="top", fontsize=8.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc"))

    fig.text(0.5, -0.06,
             "Uniform random weights are the optimistic case: each column averages over the "
             "compression curve, so a scalar gain nearly removes the error.\nReal trained "
             "weights are structured. Against the Ron-corrected reference ideal_1t1r() both "
             "cases fit to better than 0.03% rms, so the compression is real and modelable.",
             ha="center", va="top", fontsize=8.5, color=INK)
    save(fig, path)


# --------------------------------------------------------------------------
# Wire RC settling: nonzero c_row / c_col turn each line into an RC ladder.
# --------------------------------------------------------------------------
def settle_trace(p, x, tstop, tstep, rundir):
    """(time, summed column current, tau_63) from one transient run.

    The tau_63 extraction matches `verify_mna.settle_tau` -- interpolate the 63.2%
    crossing and subtract the pulse delay -- but keeps the waveform, which that
    function discards, because the figure plots it.
    """
    import hdl21.sim as hs

    res = C.settle_sim(p, x, tstop, tstep).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA,
        rundir=rundir))
    d = res.an[0].data
    t = np.asarray(d["time"])
    i = sum(np.asarray(d[f"i(v.xtop.vvsense{j})"]) for j in range(len(p.weights[0])))
    k = int(np.argmax(i >= 0.632 * i[-1]))
    tau = float(np.interp(0.632 * i[-1], i[k - 1:k + 1], t[k - 1:k + 1]) - 10 * tstep)
    return t, i, tau


def sweep_wire_c(W, x, r=5.0, tstop=40e-12, tstep=20e-15):
    """One transient per wire capacitance, at the r_wire the README quotes."""
    n = len(W)
    op, _ = C.run_mac(C.CrossbarParams(weights=W, r_row=r, r_col=r,
                                       cell=C.CellParams(passive=True)), x,
                      rundir="/tmp/xbar_fig_tran_op")
    print(f"  .op summed column current {op.sum()*1e3:.6f} mA (the transient endpoint)")
    traces = {}
    for c in C_SWEEP:
        p = C.CrossbarParams(weights=W, r_row=r, r_col=r, c_row=c, c_col=c,
                             cell=C.CellParams(passive=True))
        t, i, tau = settle_trace(p, x, tstop, tstep, f"/tmp/xbar_fig_tran_{c:.0e}")
        est = r * c * n * n / 2   # Elmore delay of a uniform line driven from one end
        traces[c] = (t, i, tau, est)
        print(f"  c={c*1e15:4.1f} fF/pitch | tau_63 {tau:.4e} s  vs R*C/2 estimate "
              f"{est:.4e} s ({tau/est:.2f}x) | endpoint {i[-1]*1e3:.6f} mA "
              f"({i[-1]/op.sum():.6f} of .op)")
    return traces, op.sum(), r, tstep


def fig_rc_settling(traces, i_op, r, tstep, path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    t0 = 10 * tstep   # settle_sim delays the input step by 10 rise times

    for (c, (t, i, tau, est)), colour in zip(traces.items(), SERIES3):
        ts = (t - t0) * 1e12
        ax0.plot(ts, i * 1e3, "-", color=colour, lw=1.6,
                 label=f"c = {c*1e15:.1f} fF/pitch, $\\tau_{{63}}$ = {tau*1e12:.3f} ps")
        ax0.plot(tau * 1e12, 0.632 * i[-1] * 1e3, "o", color=colour, ms=6,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax0.axhline(i_op * 1e3, color=INK, lw=1.1, ls=":", label="final value (.op result)")
    ax0.axhline(0.632 * i_op * 1e3, color=INK, lw=1.0, ls="--")
    ax0.annotate("63.2% of final", xy=(39, 0.632 * i_op * 1e3), xytext=(0, 6),
                 textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax0.set_xlim(-1, 40)
    ax0.set_xlabel("time since the input step  [ps]")
    ax0.set_ylabel("summed column current  [mA]")
    ax0.set_title(f"Column current settling, 32x32 at {r:g} $\\Omega$/pitch")
    ax0.legend(loc="lower right")

    for (c, (t, i, tau, est)), colour in zip(traces.items(), SERIES3):
        ts = (t - t0) * 1e12
        ax1.plot(ts, i / i[-1], "-", color=colour, lw=1.6,
                 label=f"c = {c*1e15:.1f} fF/pitch, {tau/est:.2f}x R$\\cdot$C/2")
        ax1.plot(tau * 1e12, 0.632, "o", color=colour, ms=6,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax1.axhline(0.632, color=INK, lw=1.0, ls="--")
    ax1.set_xscale("log")
    ax1.set_xlim(0.05, 40)
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel("time since the input step  [ps]")
    ax1.set_ylabel("column current, normalised to final  [-]")
    ax1.set_title("Log time separates the three time constants")
    ax1.legend(loc="lower right")

    fig.suptitle("Wire-RC settling from c_row / c_col, against the Elmore estimate "
                 "R$_{tot}$C$_{tot}$/2 = r$\\cdot$c$\\cdot$N$^2$/2", y=1.02)
    save(fig, path)


def main():
    setup_style()
    C.sky130_root()   # fail fast, with the actionable message, if no PDK is installed
    figures = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    W, x = demo_case()
    print(f"demo case: {len(W)}x{len(W[0])}, numpy.default_rng(0), "
          f"x in [{min(x):.4f}, {max(x):.4f}] V")

    print("\n== 0T1R wire IR drop sweep ==")
    rel, metrics = sweep_wire_r(W, x)
    fig_ir_drop(rel, figures / "ir-drop-vs-rwire.png")
    fig_gain_fit(metrics, figures / "gain-fit-residual.png")

    print("\n== sky130 access-FET Ron compression ==")
    ron = implied_ron(C.CellParams(access="sky130"), "fig")
    print(f"  implied Ron at the read bias: {ron:.1f} ohm")
    p, g_prog, g_0t1r, g_1t1r = cell_compression()
    arrays = array_compression(ron)
    fig_ron_compression(p, g_prog, g_0t1r, g_1t1r, arrays, ron,
                        figures / "ron-compression.png")

    print("\n== wire RC settling ==")
    traces, i_op, r, tstep = sweep_wire_c(W, x)
    fig_rc_settling(traces, i_op, r, tstep, figures / "rc-settling.png")

    print("\nall figures written to", figures)


if __name__ == "__main__":
    main()
