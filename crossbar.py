"""
Analog compute-in-memory crossbar as Hdl21 generators.

Hierarchy:  Cell  ->  Crossbar  ->  Tile
Crossbar models the array as distributed wire *resistance* (one series R per
cell pitch on every row and column, so IR drop falls out of SPICE) plus
optional distributed wire *capacitance* to vss (one shunt C per pitch, off by
default -- `c_row` / `c_col` = 0 stamps no capacitors at all).  Tile stacks the
signed differential column-pair scheme on top.
The weight matrix is a *parameter*; the SPICE netlist is a compile output.
"""
import os
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Tuple
import hdl21 as h
import hdl21.sim as hs
from hdl21.prefix import µ, n
from vlsirtools.spicetype import SpiceType

import ngspice_compat  # ngspice>=43 rawfile header vs. vlsirtools 7.0.0

ngspice_compat.apply()

Matrix = Tuple[Tuple[float, ...], ...]  # weights in [0, 1], rows x cols


# --------------------------------------------------------------------------
# Access device.  Two routes, selected by `CellParams.access`:
#
#   "generic"  a bare ngspice level-1 `.model` card, injected into the sim by
#              `access_model_attrs`.  *** PLACEHOLDER, NOT SILICON-CALIBRATED ***
#              Its only job is to make the 1T1R topology simulate with no PDK
#              installed, i.e. to exercise plumbing.  Ron / Vt / mobility are
#              round numbers, so no performance number read off it means
#              anything about real silicon.
#   "sky130"   an `h.Mos`, which `sky130_hdl21.compile()` maps onto
#              `sky130_fd_pr__nfet_01v8`, plus the PDK's ngspice model library.
#              Needs an open_pdks/volare install; see `sky130_root`.
#
# `w_acc` / `l_acc` are SI metres on both routes -- see `si_literal`.
# --------------------------------------------------------------------------
GENERIC_NMOS = "xbar_nmos_generic"

GENERIC_NMOS_MODEL = (
    f".model {GENERIC_NMOS} nmos level=1 vto=0.5 kp=120u lambda=0.05 "
    "gamma=0.0 phi=0.7 tox=4.1n"
)


@h.paramclass
class GenericMosParams:
    w = h.Param(dtype=h.Scalar, desc="Channel width")
    l = h.Param(dtype=h.Scalar, desc="Channel length")


GenericNmos = h.ExternalModule(
    name=GENERIC_NMOS,
    desc="Placeholder level-1 NMOS, paired with GENERIC_NMOS_MODEL. Not calibrated.",
    port_list=[h.Port(name="d"), h.Port(name="g"), h.Port(name="s"), h.Port(name="b")],
    paramtype=GenericMosParams,
    spicetype=SpiceType.MOS,
)


# --------------------------------------------------------------------------
# Cell: 1T1R (access NMOS + memristor).  For inference-accuracy simulation the
# memristor is a programmed conductance; for write dynamics swap in a Verilog-A
# compact model wrapped as h.ExternalModule.
# --------------------------------------------------------------------------
@h.paramclass
class CellParams:
    g = h.Param(dtype=float, desc="Programmed conductance [S]", default=10e-6)
    w_acc = h.Param(dtype=h.Scalar, desc="Access device width", default=1 * µ)
    l_acc = h.Param(dtype=h.Scalar, desc="Access device length", default=180 * n)
    passive = h.Param(dtype=bool, desc="0T1R (no access device)", default=False)
    access = h.Param(dtype=str, desc="1T1R access device: 'generic' or 'sky130'",
                     default="generic")


def si_literal(v: h.Scalar) -> h.Scalar:
    """`h.Prefixed` -> `h.Literal`, so sky130-hdl21's SI-to-micron scaling fires.

    `Sky130Walker.scale_param` wraps an `h.Literal` size as `(<text> * 1e6)` but returns
    an `h.Prefixed` untouched -- its own source says "FIXME: where's the scaling?".  So a
    plain `1 * µ` reaches ngspice as a 1 pm device, misses every binned `.model`, and
    fails with nothing but "could not find a valid modelname".  Handing the walker a
    Literal keeps `w_acc` / `l_acc` in SI metres on both access routes.
    """
    return h.Literal(f"{float(v):g}") if isinstance(v, h.Prefixed) else v


@h.generator
def Cell(p: CellParams) -> h.Module:
    m = h.Module()
    m.inp = h.Inout()   # row input line (voltage / pulse applied here)
    m.out = h.Inout()   # column sum line (currents add here)
    m.sel = h.Input()   # row select (access gate)
    m.vss = h.Port()
    if p.passive:
        m.add(h.Res(r=1.0 / p.g)(p=m.inp, n=m.out), name="rmem")
    else:
        m.mid = h.Signal()
        m.add(h.Res(r=1.0 / p.g)(p=m.inp, n=m.mid), name="rmem")
        if p.access == "generic":
            acc = GenericNmos(GenericMosParams(w=p.w_acc, l=p.l_acc))
        elif p.access == "sky130":
            acc = h.Nmos(w=si_literal(p.w_acc), l=si_literal(p.l_acc),
                         family=h.MosFamily.CORE)
        else:
            raise ValueError(f"CellParams.access: want 'generic'|'sky130', got {p.access!r}")
        m.add(acc(d=m.mid, g=m.sel, s=m.out, b=m.vss), name="macc")
    return m


# --------------------------------------------------------------------------
# Crossbar: the weight matrix and the wire parasitics are parameters.
# Every row/column segment gets its own resistor -> IR drop comes out of SPICE.
# Each tap optionally gets a shunt C to vss -> RC settling comes out of SPICE.
# --------------------------------------------------------------------------
@h.paramclass
class CrossbarParams:
    weights = h.Param(dtype=Matrix, desc="Normalized weights, rows x cols")
    g_min = h.Param(dtype=float, desc="Conductance at w=0 [S]", default=1e-6)
    g_max = h.Param(dtype=float, desc="Conductance at w=1 [S]", default=100e-6)
    r_row = h.Param(dtype=float, desc="Row wire R per cell pitch [ohm]", default=1.0)
    r_col = h.Param(dtype=float, desc="Column wire R per cell pitch [ohm]", default=1.0)
    c_row = h.Param(dtype=float, desc="Row wire C per cell pitch to vss [F]", default=0.0)
    c_col = h.Param(dtype=float, desc="Col wire C per cell pitch to vss [F]", default=0.0)
    cell = h.Param(dtype=CellParams, desc="Cell template", default=CellParams())


@h.generator
def Crossbar(p: CrossbarParams) -> h.Module:
    rows, cols = len(p.weights), len(p.weights[0])
    m = h.Module()
    m.inp = h.Inout(width=rows)   # drivers connect at the column-0 end
    m.out = h.Inout(width=cols)   # readout connects at the last-row end
    m.sel = h.Input(width=rows)
    m.vss = h.Port()

    # Distributed row wires: inp[i] -> R -> row[i][0] -> R -> row[i][1] ...
    rows_ = [m.add(h.Signal(width=cols), name=f"row{i}") for i in range(rows)]
    for i, rl in enumerate(rows_):
        m.add(h.Res(r=p.r_row)(p=m.inp[i], n=rl[0]), name=f"rr{i}_0")
        for j in range(1, cols):
            m.add(h.Res(r=p.r_row)(p=rl[j - 1], n=rl[j]), name=f"rr{i}_{j}")

    # Distributed column wires, terminating at out[j]
    cols_ = [m.add(h.Signal(width=rows), name=f"col{j}") for j in range(cols)]
    for j, cl in enumerate(cols_):
        for i in range(1, rows):
            m.add(h.Res(r=p.r_col)(p=cl[i - 1], n=cl[i]), name=f"rc{i}_{j}")
        m.add(h.Res(r=p.r_col)(p=cl[rows - 1], n=m.out[j]), name=f"rc{rows}_{j}")

    # Shunt wire C at every tap.  A zero value stamps nothing, so the default
    # netlist -- and hence the .op regression -- is byte-identical to R-only.
    if p.c_row:
        for i, rl in enumerate(rows_):
            for j in range(cols):
                m.add(h.Cap(c=p.c_row)(p=rl[j], n=m.vss), name=f"cr{i}_{j}")
    if p.c_col:
        for j, cl in enumerate(cols_):
            for i in range(rows):
                m.add(h.Cap(c=p.c_col)(p=cl[i], n=m.vss), name=f"cc{i}_{j}")

    # One cell per intersection, conductance programmed from the weight
    for i in range(rows):
        for j in range(cols):
            g = p.g_min + p.weights[i][j] * (p.g_max - p.g_min)
            cell = Cell(replace(p.cell, g=g))
            m.add(cell(inp=rows_[i][j], out=cols_[j][i], sel=m.sel[i], vss=m.vss),
                  name=f"c{i}_{j}")
    return m


# --------------------------------------------------------------------------
# Tile: signed weights on a unipolar array.  Each signed weight w in [-1, 1]
# becomes a pair of physical columns
#     G+ = g_min + max(+w, 0) * (g_max - g_min)
#     G- = g_min + max(-w, 0) * (g_max - g_min)
# so I(outp[j]) - I(outn[j]) = (g_max - g_min) * sum_i x_i * w[i][j]: the g_min
# offset cancels in the difference, which is the whole point of the scheme.
# The pair is *adjacent* physical columns (2j, 2j+1) so both legs see nearly the
# same wire parasitics and the IR droop is common-mode to first order.
#
# `mux` / `adc_bits` / `in_bits` used to live here.  They are gone: there is no
# ADC or DAC generator yet, so they were knobs with no circuit behind them.
# --------------------------------------------------------------------------
@h.paramclass
class TileParams:
    xbar = h.Param(dtype=CrossbarParams, desc="Array + parasitics; .weights are the *signed* weights in [-1, 1]")
    signed = h.Param(dtype=bool, desc="Differential column pairs (G+ - G-)", default=True)


@h.generator
def Tile(p: TileParams) -> h.Module:
    rows, cols = len(p.xbar.weights), len(p.xbar.weights[0])
    m = h.Module()
    m.inp = h.Inout(width=rows)
    m.sel = h.Input(width=rows)
    m.vss = h.Port()

    if not p.signed:
        m.out = h.Inout(width=cols)
        m.xbar = Crossbar(p.xbar)(inp=m.inp, out=m.out, sel=m.sel, vss=m.vss)
        return m

    m.outp = h.Inout(width=cols)   # G+ leg of each pair
    m.outn = h.Inout(width=cols)   # G- leg of each pair
    split = tuple(
        tuple(v for w in row for v in (max(w, 0.0), max(-w, 0.0)))
        for row in p.xbar.weights
    )
    pairs = [s for j in range(cols) for s in (m.outp[j], m.outn[j])]
    # `Concat` parts land on the target's bits in reverse: verified against
    # ngspice, and pinned by the signed-MAC check in verify_mna.py.
    m.xbar = Crossbar(replace(p.xbar, weights=split))(
        inp=m.inp, out=h.Concat(*reversed(pairs)), sel=m.sel, vss=m.vss)
    return m


# --------------------------------------------------------------------------
# Access-device models for the sim deck.  0T1R needs nothing; 1T1R needs either
# the placeholder `.model` card or the sky130 model library.
# --------------------------------------------------------------------------
SKY130_LIB_PATH = Path("libs.tech/ngspice/sky130.lib.spice")


def sky130_root() -> Path:
    """The `sky130A` directory holding an ngspice model library.

    Searches $PDK_ROOT, then $VOLARE_ROOT, then ~/.volare -- accepting both volare's
    `<root>/sky130A` symlink and the `.../sky130/versions/<hash>/sky130A` behind it.
    """
    roots = [Path(os.environ[v]) for v in ("PDK_ROOT", "VOLARE_ROOT") if os.environ.get(v)]
    roots += [Path.home() / ".volare"]
    for root in roots:
        for pdk_path in [root / "sky130A", *sorted(root.glob("sky130/versions/*/sky130A")),
                         *sorted(root.glob("volare/sky130/versions/*/sky130A"))]:
            if (pdk_path / SKY130_LIB_PATH).is_file():
                return pdk_path
    raise RuntimeError(
        f"sky130 SPICE models not found: no 'sky130A/{SKY130_LIB_PATH}' under any of "
        f"{', '.join(str(r) for r in roots)}. The `sky130-hdl21` package ships device "
        "definitions only, not models. Set $PDK_ROOT (or $VOLARE_ROOT) to an "
        "open_pdks/volare sky130 install, or make one with:  uv tool install volare "
        "&& volare enable --pdk sky130 <version>"
    )


def sky130_install():
    """Cached `sky130_hdl21.Install` for the discovered PDK; `.include()` gives the `.lib`."""
    import sky130_hdl21

    if sky130_hdl21.Install.singleton is None:  # __post_init__ refuses a second one
        sky130_hdl21.install = sky130_hdl21.Install(
            pdk_path=sky130_root(), lib_path=SKY130_LIB_PATH,
            model_ref=SKY130_LIB_PATH.parent)
    return sky130_hdl21.Install.singleton


def access_model_attrs(tb: h.Module, cell: CellParams) -> list:
    """Sim attrs the cell's access device needs.  Compiles `tb` in place for sky130."""
    if cell.passive:
        return []
    if cell.access == "generic":
        return [hs.Literal(GENERIC_NMOS_MODEL)]
    if cell.access != "sky130":
        raise ValueError(f"CellParams.access: want 'generic'|'sky130', got {cell.access!r}")
    import sky130_hdl21

    inst = sky130_install()   # discover the models first: don't mutate `tb` and then fail
    sky130_hdl21.compile(tb)  # h.Mos -> sky130_fd_pr__nfet_01v8, in place
    return [inst.include(h.pdk.Corner.TYP)]


# --------------------------------------------------------------------------
# Testbench: apply an input vector, hold columns at virtual ground with 0 V
# sense sources (an ideal TIA), read the summed currents, compare to G @ x.
# --------------------------------------------------------------------------
def mac_sim(p: CrossbarParams, x: Tuple[float, ...]) -> hs.Sim:
    rows, cols = len(p.weights), len(p.weights[0])
    xbar = Crossbar(p)

    tb = h.Module(name="Tb")
    tb.vss = h.Port()
    tb.inp = h.Signal(width=rows)
    tb.out = h.Signal(width=cols)
    tb.sel = h.Signal(width=rows)
    tb.dut = xbar(inp=tb.inp, out=tb.out, sel=tb.sel, vss=tb.vss)
    for i in range(rows):
        tb.add(h.Vdc(dc=x[i])(p=tb.inp[i], n=tb.vss), name=f"vin{i}")
        tb.add(h.Vdc(dc=1.8)(p=tb.sel[i], n=tb.vss), name=f"vsel{i}")
    for j in range(cols):
        tb.add(h.Vdc(dc=0.0)(p=tb.out[j], n=tb.vss), name=f"vsense{j}")

    models = access_model_attrs(tb, p.cell)
    return hs.Sim(tb=tb, attrs=[*models, hs.Op(), hs.Literal(".save all")])


def settle_sim(p: CrossbarParams, x, tstop: float, tstep: float) -> hs.Sim:
    """`mac_sim` with the row drivers stepped 0 -> x, to see the wire-RC settling."""
    rows, cols = len(p.weights), len(p.weights[0])
    rise = tstep

    tb = h.Module(name="Tb")
    tb.vss = h.Port()
    tb.inp = h.Signal(width=rows)
    tb.out = h.Signal(width=cols)
    tb.sel = h.Signal(width=rows)
    tb.dut = Crossbar(p)(inp=tb.inp, out=tb.out, sel=tb.sel, vss=tb.vss)
    for i in range(rows):
        tb.add(h.Vpulse(v1=0.0, v2=x[i], delay=10 * rise, rise=rise, fall=rise,
                        width=tstop, period=2 * tstop)(p=tb.inp[i], n=tb.vss),
               name=f"vin{i}")
        tb.add(h.Vdc(dc=1.8)(p=tb.sel[i], n=tb.vss), name=f"vsel{i}")
    for j in range(cols):
        tb.add(h.Vdc(dc=0.0)(p=tb.out[j], n=tb.vss), name=f"vsense{j}")

    models = access_model_attrs(tb, p.cell)
    saves = " ".join(f"i(v.xtop.vvsense{j})" for j in range(cols))
    return hs.Sim(tb=tb, attrs=[*models, hs.Tran(tstop=tstop, tstep=tstep),
                                hs.Literal(f".save {saves}")])


def ideal_0t1r(p: CrossbarParams, x):
    """The zero-parasitic 0T1R reference current, G.T @ x.

    Valid *only* for `passive=True` cells with no wire R: it assumes the whole
    read voltage lands across the cell.  With an access FET in series the cell
    current is set by Ron(Vgs, Vds) as well, so this is wrong by construction --
    not noisy -- for 1T1R.  Use `ideal_1t1r` there.
    """
    import numpy as np

    G = p.g_min + np.array(p.weights) * (p.g_max - p.g_min)
    return G.T @ np.array(x)


def ideal_1t1r(p: CrossbarParams, x, ron: float):
    """Series-Ron-corrected reference: G_eff[i,j] = 1 / (1/G[i,j] + ron).

    Still an approximation, not an identity: Ron depends on Vgs and Vds, so one number
    is only right at the bias it was measured at.  It is the right *shape* though.
    Unlike wire IR drop this error is weight-dependent: it compresses the top of the
    conductance range far harder than the bottom -- at the measured ron ~= 820 and the
    default g_min/g_max, 7.6% at w=1 against 0.08% at w=0.  Whether one scalar gain can
    stand in for it therefore depends on the weight structure, not just on ron; see the
    `1T1R compression` check in verify_mna.py.
    """
    import numpy as np

    G = p.g_min + np.array(p.weights) * (p.g_max - p.g_min)
    return (1.0 / (1.0 / G + ron)).T @ np.array(x)


def run_mac(p: CrossbarParams, x, rundir="/tmp/xbar"):
    """Returns (spice column currents, zero-parasitic 0T1R reference) in amps."""
    import numpy as np

    if not p.cell.passive:
        warnings.warn(
            f"1T1R cell (access={p.cell.access!r}): the returned reference is the "
            "zero-parasitic 0T1R ideal, which ignores the access FET's Ron. That is a "
            "large (~8% at g_max) and weight-dependent compression of the conductance "
            "range, not a rounding error. Unlike IR drop it is a per-cell nonlinearity, "
            "so how much of it a scalar gain fit absorbs depends on the weight "
            "distribution -- most of it for i.i.d. weights, little for structured ones. "
            "Use `ideal_1t1r(p, x, ron)` to compare against a 1T1R run.",
            stacklevel=2)

    res = mac_sim(p, x).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA, rundir=rundir))
    data = res.an[0].data
    cols = len(p.weights[0])
    spice = np.array([data[f"i(v.xtop.vvsense{j})"] for j in range(cols)])
    return spice, ideal_0t1r(p, x)


def error_metrics(spice, ideal):
    """Split the column error into calibratable gain droop and what is left.

    IR drop is one-sided, so `abs()` on the raw error hides that most of it is a
    single systematic gain term the digital path can divide out. Returns
    (signed mean relative error %, best-fit gain a, signed worst relative error %,
    rms residual % after removing a, worst residual % after removing a).

    The split is clean for wire IR drop only.  On a 1T1R run the residual also carries
    weight-dependent Ron compression, and how much of that one gain removes depends on
    the weight distribution -- most of it for i.i.d. weights, little for structured ones
    -- so do not read the residual there as "the non-calibratable part" and stop:
    compare against `ideal_1t1r` too, and see how much of the residual the Ron
    correction eats.
    """
    import numpy as np

    rel = 100 * (spice - ideal) / ideal
    a = float(spice @ ideal / (ideal @ ideal))     # argmin ||spice - a*ideal||
    resid = 100 * (spice - a * ideal) / (a * ideal)
    return (rel.mean(), a, rel[np.abs(rel).argmax()],
            float(np.sqrt((resid ** 2).mean())), float(np.abs(resid).max()))


def tile_sim(p: TileParams, x: Tuple[float, ...]) -> hs.Sim:
    """Both legs of every column pair held at virtual ground and sensed separately."""
    rows, cols = len(p.xbar.weights), len(p.xbar.weights[0])

    tb = h.Module(name="Tb")
    tb.vss = h.Port()
    tb.inp = h.Signal(width=rows)
    tb.outp = h.Signal(width=cols)
    tb.outn = h.Signal(width=cols)
    tb.sel = h.Signal(width=rows)
    tb.dut = Tile(p)(inp=tb.inp, outp=tb.outp, outn=tb.outn, sel=tb.sel, vss=tb.vss)
    for i in range(rows):
        tb.add(h.Vdc(dc=x[i])(p=tb.inp[i], n=tb.vss), name=f"vin{i}")
        tb.add(h.Vdc(dc=1.8)(p=tb.sel[i], n=tb.vss), name=f"vsel{i}")
    for j in range(cols):
        tb.add(h.Vdc(dc=0.0)(p=tb.outp[j], n=tb.vss), name=f"vsensep{j}")
        tb.add(h.Vdc(dc=0.0)(p=tb.outn[j], n=tb.vss), name=f"vsensen{j}")

    models = access_model_attrs(tb, p.xbar.cell)
    return hs.Sim(tb=tb, attrs=[*models, hs.Op(), hs.Literal(".save all")])


def run_tile_mac(p: TileParams, x, rundir="/tmp/tile"):
    """Returns (differential column currents I+ - I-, (g_max-g_min) * W_signed.T @ x)."""
    import numpy as np

    res = tile_sim(p, x).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA, rundir=rundir))
    data = res.an[0].data
    cols = len(p.xbar.weights[0])
    diff = np.array([data[f"i(v.xtop.vvsensep{j})"] - data[f"i(v.xtop.vvsensen{j})"]
                     for j in range(cols)])
    W = np.array(p.xbar.weights)
    return diff, (p.xbar.g_max - p.xbar.g_min) * (W.T @ np.array(x))


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    N = 32
    W = tuple(tuple(float(v) for v in row) for row in rng.random((N, N)))
    x = tuple(float(v) for v in rng.random(N) * 0.2)  # 0..200 mV read voltages

    for r in (0.0, 1.0, 5.0):
        p = CrossbarParams(weights=W, r_row=r, r_col=r, cell=CellParams(passive=True))
        spice, ideal = run_mac(p, x, rundir=f"/tmp/xbar_r{int(r)}")
        droop, a, worst, rms, worst_res = error_metrics(spice, ideal)
        print(f"{N}x{N}, r_wire={r:>3} ohm/pitch: mean err {droop:6.2f}%  "
              f"worst column {worst:6.2f}%  |  after gain fit (a={a:.4f}): "
              f"residual rms {rms:5.2f}%  worst {worst_res:5.2f}%")
