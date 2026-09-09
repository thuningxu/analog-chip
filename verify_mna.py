"""Independent cross-check of the ngspice crossbar result.

Builds the passive (0T1R) crossbar's nodal-analysis system directly in numpy and
compares its column currents against what ngspice reports for the same
`CrossbarParams`. Two independent solvers agreeing to ~1e-9 relative is strong
evidence the generated netlist has the topology we think it has and that the
SPICE operating point is being read correctly.

Topology being asserted (mirrors `Crossbar`):
    row i:  inp[i] -R_row- row[i][0] -R_row- row[i][1] ... row[i][cols-1]
    col j:  col[j][0] -R_col- col[j][1] ... col[j][rows-1] -R_col- out[j]
    cell:   row[i][j] -1/G[i,j]- col[j][i]
    inp[i] forced to x[i]; out[j] forced to 0 V.

Checks, in order:
    1  mna              0T1R column currents, ngspice vs the numpy MNA solve above.
    2  op regression    the .op error numbers the README quotes, asserted to 2 dp.
    3  error metrics    the gain-droop / residual split, plus an lstsq cross-check
                        of the closed-form best-fit gain.
    4  1T1R generic     the *default* cell (1T1R + placeholder .model) simulates,
                        and its implied Ron matches the level-1 hand estimate.
    5  sky130 wiring    compile -> sky130_fd_pr__nfet_01v8 plus the `.lib ... tt` from
                        `Install.include`, SI geometry scaled to PDK microns, and a
                        clear actionable error when no PDK is found. Deck only.
    6  sky130 sim       the 1T1R .op against the real PDK models, at the *default*
                        SI geometry. Skips (passing) when no model library is
                        installed, so this file still passes on a PDK-less machine.
    7  1T1R compression access-FET Ron compresses the conductance range, and whether
                        one scalar gain absorbs it depends on the weight structure.
    8  wire cap         nonzero c_row/c_col produce RC settling that tracks the
                        analytic R_tot*C_tot/2 estimate and scales linearly in c.
    9  signed tile      the differential column pair recovers
                        (g_max - g_min) * W_signed.T @ x at zero wire R.
"""

import io
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import hdl21 as h
import hdl21.sim as hs
import numpy as np

import crossbar as C


def mna_columns(p: "C.CrossbarParams", x) -> np.ndarray:
    """Column currents [A] into the out[j] virtual grounds, by direct MNA solve."""
    W = np.asarray(p.weights, dtype=float)
    rows, cols = W.shape
    Gcell = p.g_min + W * (p.g_max - p.g_min)
    g_row, g_col = 1.0 / p.r_row, 1.0 / p.r_col
    x = np.asarray(x, dtype=float)

    n_rc = rows * cols
    N = 2 * n_rc
    G = np.zeros((N, N))
    I = np.zeros(N)

    def ir(i, j):  # row-line node index
        return i * cols + j

    def ic(j, i):  # column-line node index
        return n_rc + j * rows + i

    def stamp(a, b, g):
        G[a, a] += g
        G[b, b] += g
        G[a, b] -= g
        G[b, a] -= g

    def stamp_to_source(a, g, v):  # resistor from node a to a forced voltage v
        G[a, a] += g
        I[a] += g * v

    for i in range(rows):
        stamp_to_source(ir(i, 0), g_row, x[i])  # driver into column-0 tap
        for j in range(1, cols):
            stamp(ir(i, j - 1), ir(i, j), g_row)

    for j in range(cols):
        for i in range(1, rows):
            stamp(ic(j, i - 1), ic(j, i), g_col)
        stamp_to_source(ic(j, rows - 1), g_col, 0.0)  # last tap into virtual gnd

    for i in range(rows):
        for j in range(cols):
            stamp(ir(i, j), ic(j, i), Gcell[i, j])

    V = np.linalg.solve(G, I)
    # Current out of the last column tap into the 0 V sense source.
    return np.array([g_col * (V[ic(j, rows - 1)] - 0.0) for j in range(cols)])


def demo_case(n=32, seed=0):
    """The (W, x) pair `crossbar.py`'s __main__ and the README numbers come from."""
    rng = np.random.default_rng(seed)
    W = tuple(tuple(float(v) for v in row) for row in rng.random((n, n)))
    x = tuple(float(v) for v in rng.random(n) * 0.2)
    return W, x


def check_mna() -> bool:
    """0T1R column currents: ngspice vs the direct numpy nodal solve."""
    W, x = demo_case()

    worst = 0.0
    for r in (1.0, 5.0, 20.0):
        p = C.CrossbarParams(
            weights=W, r_row=r, r_col=r, cell=C.CellParams(passive=True)
        )
        spice, ideal = C.run_mac(p, x, rundir=f"/tmp/xbar_v{int(r)}")
        mna = mna_columns(p, x)
        rel = np.abs(spice - mna) / np.abs(mna)
        worst = max(worst, rel.max())
        print(
            f"  r={r:>4} ohm/pitch | ngspice vs numpy-MNA: max rel diff {rel.max():.2e}"
            f" | IR-drop err vs ideal: mean {100*np.abs(spice-ideal).mean()/np.abs(ideal).mean():5.2f}%"
        )

    # Independent check of the r=0 limit: no wire drop => exactly G.T @ x.
    p0 = C.CrossbarParams(
        weights=W, r_row=0.0, r_col=0.0, cell=C.CellParams(passive=True)
    )
    spice0, ideal0 = C.run_mac(p0, x, rundir="/tmp/xbar_v0")
    rel0 = np.abs(spice0 - ideal0).max() / np.abs(ideal0).max()
    print(f"  r=   0 ohm/pitch | ngspice vs analytic G.T@x: max rel diff {rel0:.2e}")

    print(f"  worst rel diff {max(worst, rel0):.2e} (want < 1e-6)")
    return worst < 1e-6 and rel0 < 1e-6


def check_op_regression() -> bool:
    """Pin the .op error magnitudes the README quotes, so refactors cannot drift them."""
    W, x = demo_case()
    want = {0.0: (0.00, 0.00), 1.0: (3.59, 4.72), 5.0: (15.50, 20.01)}

    ok = True
    for r in (0.0, 1.0, 5.0):
        p = C.CrossbarParams(
            weights=W, r_row=r, r_col=r, cell=C.CellParams(passive=True)
        )
        spice, ideal = C.run_mac(p, x, rundir=f"/tmp/xbar_reg{int(r)}")
        droop, _, worst, _, _ = C.error_metrics(spice, ideal)
        got = (round(abs(droop), 2), round(abs(worst), 2))
        ok &= got == want[r]
        print(f"  r={r:>4} ohm/pitch | |mean| {got[0]:5.2f}%  |worst| {got[1]:5.2f}%"
              f"  want {want[r][0]:5.2f}% / {want[r][1]:5.2f}%  {'ok' if got == want[r] else 'DRIFT'}")
    return ok


def check_error_metrics() -> bool:
    """IR drop is one-sided, and most of it is one calibratable gain."""
    W, x = demo_case()
    p = C.CrossbarParams(weights=W, r_row=5.0, r_col=5.0, cell=C.CellParams(passive=True))
    spice, ideal = C.run_mac(p, x, rundir="/tmp/xbar_metrics")
    droop, a, worst, rms, worst_res = C.error_metrics(spice, ideal)

    # `a` in closed form must equal an independent least-squares solve.
    a_ls = float(np.linalg.lstsq(ideal[:, None], spice, rcond=None)[0][0])
    checks = {
        "every column droops (spice < ideal)": bool(np.all(spice < ideal)),
        "signed mean droop is negative": droop < 0,
        "signed worst is negative": worst < 0,
        "best-fit gain a < 1": a < 1.0,
        "closed-form a == lstsq a": abs(a - a_ls) < 1e-12,
        "gain fit removes most of the error": rms < abs(droop),
    }
    print(f"  droop {droop:6.2f}%  worst {worst:6.2f}%  a {a:.6f} (lstsq {a_ls:.6f})"
          f"  residual rms {rms:5.2f}%  worst {worst_res:5.2f}%")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_1t1r_generic() -> bool:
    """The default cell -- 1T1R on the placeholder .model -- must simulate as shipped."""
    # 1x1 array, no wire R: the FET is then exactly in series with one cell, so the
    # implied Ron is a clean division and can be checked against the level-1 formula.
    g = 100e-6
    x = (0.2,)
    p1 = C.CrossbarParams(weights=((1.0,),), g_max=g, r_row=0.0, r_col=0.0)
    p0 = C.CrossbarParams(weights=((1.0,),), g_max=g, r_row=0.0, r_col=0.0,
                          cell=C.CellParams(passive=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        i1, _ = C.run_mac(p1, x, rundir="/tmp/xbar_1t1r_1")
    i0, _ = C.run_mac(p0, x, rundir="/tmp/xbar_1t1r_0")
    ron = x[0] / i1[0] - x[0] / i0[0]
    # Level-1 linear region: Ron = 1 / (kp * (W/L) * (Vgs - Vth)), kp=120u, Vth=0.5
    ron_est = 1.0 / (120e-6 * (1e-6 / 180e-9) * (1.8 - 0.5))

    # And the shipped default params on a real array, to prove the path is live.
    W, x8 = demo_case(n=8)
    pn, pp = (C.CrossbarParams(weights=W, r_row=1.0, r_col=1.0),
              C.CrossbarParams(weights=W, r_row=1.0, r_col=1.0,
                               cell=C.CellParams(passive=True)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s1, _ = C.run_mac(pn, x8, rundir="/tmp/xbar_1t1r_n")
    s0, _ = C.run_mac(pp, x8, rundir="/tmp/xbar_1t1r_p")

    checks = {
        "default CellParams() is 1T1R + generic": (not C.CellParams().passive
                                                   and C.CellParams().access == "generic"),
        "run_mac warns that the 0T1R reference is invalid here": any(
            issubclass(w.category, UserWarning) and "0T1R ideal" in str(w.message)
            for w in caught),
        "implied Ron within 2x of level-1 estimate": 0.5 < ron / ron_est < 2.0,
        "1T1R currents all positive": bool(np.all(s1 > 0)),
        "series Ron lowers every column vs 0T1R": bool(np.all(s1 < s0)),
    }
    print(f"  1x1 implied Ron {ron:.1f} ohm vs level-1 estimate {ron_est:.1f} ohm")
    print(f"  8x8 1T1R/0T1R column current ratio: {(s1/s0).min():.4f}..{(s1/s0).max():.4f}")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def pdk_present() -> bool:
    """True if a real sky130 model library is reachable from this machine."""
    try:
        C.sky130_root()
    except RuntimeError:
        return False
    return True


def implied_ron(cell: C.CellParams, tag: str) -> float:
    """Access-FET Ron [ohm] from a 1x1 array: (V/I with the FET) - (V/I without)."""
    base = dict(weights=((1.0,),), g_max=100e-6, r_row=0.0, r_col=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i1, _ = C.run_mac(C.CrossbarParams(**base, cell=cell), (0.2,),
                          rundir=f"/tmp/xbar_ron_{tag}")
    i0, _ = C.run_mac(C.CrossbarParams(**base, cell=C.CellParams(passive=True)), (0.2,),
                      rundir="/tmp/xbar_ron_0")
    return 0.2 / i1[0] - 0.2 / i0[0]


def check_sky130_wiring() -> bool:
    """The sky130 path: deck construction always, electricals only if a PDK is here."""
    saved = {v: os.environ.get(v) for v in ("PDK_ROOT", "VOLARE_ROOT")}
    checks = {}
    try:
        # Mask both env vars *and* the ~/.volare fallback, so this exercises the
        # no-PDK-anywhere path even on a machine that has one installed.
        with tempfile.TemporaryDirectory() as empty, \
                patch.object(C.Path, "home", staticmethod(lambda: Path(empty))):
            os.environ["PDK_ROOT"] = empty
            os.environ.pop("VOLARE_ROOT", None)
            try:
                C.sky130_root()
                checks["missing PDK raises"] = False
                msg = ""
            except RuntimeError as e:
                checks["missing PDK raises"] = True
                msg = str(e)
            checks["error names PDK_ROOT"] = "PDK_ROOT" in msg
            checks["error names VOLARE_ROOT"] = "VOLARE_ROOT" in msg
            checks["error names the volare command"] = "volare enable --pdk sky130" in msg
            checks["error names the missing file"] = "sky130.lib.spice" in msg

        # With a PDK tree in place the production path runs: compile + `.lib` attr.
        # The lib body is a stub -- this checks the deck, never sky130 electricals.
        with tempfile.TemporaryDirectory() as root:
            lib = Path(root) / "sky130A" / C.SKY130_LIB_PATH
            lib.parent.mkdir(parents=True)
            lib.write_text(".lib tt\n.endl\n")
            os.environ["PDK_ROOT"] = root
            p = C.CrossbarParams(weights=((0.5, 0.25), (0.75, 1.0)),
                                 cell=C.CellParams(access="sky130"))
            sim = C.mac_sim(p, (0.1, 0.2))
            libs = [a for a in sim.attrs if isinstance(a, hs.Lib)]
            buf = io.StringIO()
            h.netlist(h.to_proto(sim.tb), buf, fmt="spice")  # raises on an uncompiled h.Mos
            netlist = buf.getvalue()
            checks["`.lib ... tt` comes from Install.include"] = (
                len(libs) == 1 and libs[0].section == "tt" and Path(libs[0].path) == lib)
            checks["compile mapped h.Mos -> nfet_01v8"] = (
                "sky130_fd_pr__nfet_01v8" in netlist)
            checks["access FET is a subckt instance"] = "xmacc" in netlist
            # `si_literal` must survive into the deck as sky130-hdl21's um scaling.
            checks["default SI geometry is scaled to um"] = (
                "w='(1e-06 * 1e6)'" in netlist and "l='(1.8e-07 * 1e6)'" in netlist)
    finally:
        for v, was in saved.items():
            os.environ.pop(v, None)
            if was is not None:
                os.environ[v] = was
        # `sky130_install` caches into a process-wide singleton that `Install.__post_init__`
        # refuses to replace. This check aimed it at a stub tree, so clear it or every
        # later check inherits a PDK path that no longer exists.
        import sky130_hdl21

        sky130_hdl21.Install.singleton = None
        sky130_hdl21.install = None

    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_sky130_sim() -> bool:
    """Run the sky130 1T1R for real.  Skips, passing, when no PDK is installed."""
    if not pdk_present():
        print("  SKIP: no sky130 model library found; nothing electrical checked.")
        return True

    W, x = ((1.0, 0.5), (0.25, 0.75)), (0.2, 0.1)
    common = dict(weights=W, r_row=0.0, r_col=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Default SI geometry, no overrides: w_acc=1*µ, l_acc=180*n.
        i_sky, _ = C.run_mac(C.CrossbarParams(
            **common, cell=C.CellParams(access="sky130")), x, rundir="/tmp/xbar_sky_sim")
        i_gen, _ = C.run_mac(C.CrossbarParams(**common), x, rundir="/tmp/xbar_sky_gen")
    i_0t1r, _ = C.run_mac(C.CrossbarParams(
        **common, cell=C.CellParams(passive=True)), x, rundir="/tmp/xbar_sky_ref")

    ron_sky = implied_ron(C.CellParams(access="sky130"), "sky")
    ron_gen = implied_ron(C.CellParams(), "gen")
    checks = {
        "default-geometry sky130 1T1R .op solves": bool(np.all(i_sky > 0)),
        "series Ron lowers every column vs 0T1R": bool(np.all(i_sky < i_0t1r)),
        # A unit slip is a ~1e6x error, so a sane Ron band is the real pin on
        # `si_literal`: a 1 pm device does not solve, and a 1 m one draws nothing.
        "sky130 Ron in a physical band (300..3000 ohm)": 300 < ron_sky < 3000,
        "placeholder Ron is the right order as sky130": 0.5 < ron_gen / ron_sky < 2.0,
    }
    print(f"  0T1R    {i_0t1r}")
    print(f"  generic {i_gen}")
    print(f"  sky130  {i_sky}   (ratio to 0T1R {i_sky / i_0t1r})")
    print(f"  implied Ron at the read bias: sky130 {ron_sky:.1f} ohm, "
          f"placeholder {ron_gen:.1f} ohm")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_1t1r_compression() -> bool:
    """Access-FET Ron compresses the conductance range, and it is weight-structured.

    Two weight matrices, same array, same bias.  With well-mixed random weights each
    column averages over the compression curve and one scalar gain nearly absorbs it.
    With graded columns -- column j uniformly at weight j/(N-1), so each column sits at
    a different point on the curve -- the same scalar gain absorbs almost nothing. The
    series-Ron-corrected reference handles both.
    """
    if not pdk_present():
        print("  SKIP: no sky130 model library found.")
        return True

    N = 16
    cases = {
        "random  ": demo_case(n=N),
        "graded  ": (tuple(tuple(j / (N - 1) for j in range(N)) for _ in range(N)),
                     tuple(0.1 for _ in range(N))),
    }
    out = {}
    for tag, (W, x) in cases.items():
        p = C.CrossbarParams(weights=W, r_row=0.0, r_col=0.0,
                             cell=C.CellParams(access="sky130"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spice, ideal0 = C.run_mac(p, x, rundir=f"/tmp/xbar_cmp_{tag.strip()}")
        ron = implied_ron(C.CellParams(access="sky130"), "sky")
        m0 = C.error_metrics(spice, ideal0)
        m1 = C.error_metrics(spice, C.ideal_1t1r(p, x, ron))
        out[tag] = (m0, m1)
        print(f"  {tag} vs 0T1R ideal      : droop {m0[0]:6.2f}%  a {m0[1]:.4f}  "
              f"residual rms {m0[3]:5.2f}%  worst {m0[4]:5.2f}%")
        print(f"  {tag} vs Ron-corrected   : droop {m1[0]:6.2f}%  a {m1[1]:.4f}  "
              f"residual rms {m1[3]:5.2f}%  worst {m1[4]:5.2f}%")

    # Per-cell compression really is weight-dependent across ~2 decades.
    ron = implied_ron(C.CellParams(access="sky130"), "sky")
    lo, hi = (100 * (1 - 1 / (1 + ron * g)) for g in (1e-6, 100e-6))
    print(f"  per-cell compression at ron={ron:.0f}: {lo:.3f}% at g_min .. {hi:.3f}% at g_max")

    (rand0, rand1), (grad0, grad1) = out["random  "], out["graded  "]
    checks = {
        "1T1R droops vs the 0T1R ideal, both structures": rand0[0] < -1 and grad0[0] < -1,
        "per-cell compression spans >10x across the range": hi / lo > 10,
        "scalar gain nearly absorbs it for random weights": rand0[3] < 0.25 * abs(rand0[0]),
        "scalar gain does NOT absorb it for graded columns": grad0[3] > 0.6 * abs(grad0[0]),
        "Ron-corrected reference fits random weights": rand1[3] < 0.1,
        "Ron-corrected reference fits graded columns": grad1[3] < 0.1,
    }
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def settle_tau(p: C.CrossbarParams, x, tstop, tstep, rundir) -> tuple:
    """(63% settling time of the summed column current, its final value)."""
    res = C.settle_sim(p, x, tstop, tstep).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA,
        rundir=rundir))
    d = res.an[0].data
    t = np.asarray(d["time"])
    i = sum(np.asarray(d[f"i(v.xtop.vvsense{j})"]) for j in range(len(p.weights[0])))
    k = int(np.argmax(i >= 0.632 * i[-1]))
    tau = float(np.interp(0.632 * i[-1], i[k - 1:k + 1], t[k - 1:k + 1]) - 10 * tstep)
    return tau, float(i[-1])


def check_wire_cap_tran() -> bool:
    """c_row / c_col must actually build an RC line, with the right time constant."""
    W, x = demo_case()
    N, r = 32, 5.0
    tstop, tstep = 40e-12, 20e-15

    def case(c, tag):
        p = C.CrossbarParams(weights=W, r_row=r, r_col=r, c_row=c, c_col=c,
                             cell=C.CellParams(passive=True))
        return settle_tau(p, x, tstop, tstep, f"/tmp/xbar_tran_{tag}")

    tau_0, i_0 = case(0.0, "c0")
    tau_a, i_a = case(0.2e-15, "ca")
    tau_b, _ = case(2.0e-15, "cb")
    # Uniform RC line driven from one end: Elmore delay = R_tot * C_tot / 2, with
    # R_tot = r*N and C_tot = c*N on both the row and the column line.
    est_a, est_b = r * 0.2e-15 * N * N / 2, r * 2.0e-15 * N * N / 2

    op, _ = C.run_mac(C.CrossbarParams(weights=W, r_row=r, r_col=r,
                                       cell=C.CellParams(passive=True)), x,
                      rundir="/tmp/xbar_tran_op")
    checks = {
        "c=0 stamps no capacitor (tau ~ input rise)": tau_0 < 2 * tstep,
        "tran endpoint == the .op result": abs(i_a / op.sum() - 1) < 1e-6,
        "tau(0.2 fF) within 3x of R*C/2": 1 / 3 < tau_a / est_a < 3,
        "tau(2 fF) within 3x of R*C/2": 1 / 3 < tau_b / est_b < 3,
        "tau scales linearly in c (10x)": abs(tau_b / tau_a / 10 - 1) < 0.15,
    }
    print(f"  c=0      : tau_63 {tau_0:.3e} s (no RC pole expected)")
    print(f"  c=0.2 fF : tau_63 {tau_a:.3e} s  vs R*C/2 estimate {est_a:.3e} s"
          f"  ({tau_a/est_a:.2f}x)")
    print(f"  c=2   fF : tau_63 {tau_b:.3e} s  vs R*C/2 estimate {est_b:.3e} s"
          f"  ({tau_b/est_b:.2f}x)")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_signed_tile() -> bool:
    """I(outp) - I(outn) must be (g_max - g_min) * W_signed.T @ x with ideal wires."""
    rng = np.random.default_rng(1)
    n = 8
    Ws = tuple(tuple(float(v) for v in row) for row in rng.random((n, n)) * 2 - 1)
    x = tuple(float(v) for v in rng.random(n) * 0.2)

    p0 = C.TileParams(xbar=C.CrossbarParams(
        weights=Ws, r_row=0.0, r_col=0.0, cell=C.CellParams(passive=True)))
    diff0, ref0 = C.run_tile_mac(p0, x, rundir="/tmp/tile_v0")
    rel0 = np.abs(diff0 - ref0).max() / np.abs(ref0).max()

    p5 = C.TileParams(xbar=C.CrossbarParams(
        weights=Ws, r_row=5.0, r_col=5.0, cell=C.CellParams(passive=True)))
    diff5, ref5 = C.run_tile_mac(p5, x, rundir="/tmp/tile_v5")
    rel5 = np.abs(diff5 - ref5).max() / np.abs(ref5).max()

    checks = {
        "signed MAC exact at r=0": rel0 < 1e-9,
        "negative weights give negative currents": bool(np.any(diff0 < 0)),
        "sign of every column matches the reference": bool(
            np.all(np.sign(diff0) == np.sign(ref0))),
        "wire R degrades it but does not break it": rel5 < 0.05,
    }
    print(f"  r=0 ohm/pitch | max rel err vs (g_max-g_min)*W.T@x: {rel0:.2e}")
    print(f"  r=5 ohm/pitch | max rel err vs (g_max-g_min)*W.T@x: {rel5:.2e}")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


CHECKS = [
    ("mna", check_mna),
    ("op regression", check_op_regression),
    ("error metrics", check_error_metrics),
    ("1T1R generic", check_1t1r_generic),
    ("sky130 wiring", check_sky130_wiring),
    ("sky130 sim", check_sky130_sim),
    ("1T1R compression", check_1t1r_compression),
    ("wire cap", check_wire_cap_tran),
    ("signed tile", check_signed_tile),
]


def main() -> int:
    results = {}
    for name, fn in CHECKS:
        print(f"\n== {name} ==")
        results[name] = fn()

    print()
    for name, hit in results.items():
        print(f"{'PASS' if hit else 'FAIL'}  {name}")
    ok = all(results.values())
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
