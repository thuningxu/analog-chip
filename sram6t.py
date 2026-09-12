"""6T SRAM bitcell in sky130: the three special SRAM devices, wrapped, and its margins.

Everything else in this project stores its weights in *conductance*.  This module stores a
bit in a **latch**, which is a different problem with a different acceptance criterion: a
crossbar cell is judged by how linearly it converts a voltage into a current, a bitcell is
judged by whether it holds its value while you look at it.  So the deliverable here is not
a netlist, it is a set of measured margins -- static noise margin above all -- and
`verify_sram6t.py` asserts them.

Three things set the shape of this module.

  * **There is no sizing exercise.**  sky130 characterizes its bitcell devices at exactly
    the widths its own bitcell draws: one W per device, no other bin.  `model_bins` parses
    the `wmin`/`wmax`/`lmin`/`lmax` limits straight out of each device's model file and
    `check_bins` refuses a geometry outside them, because a MOSFET outside its bin is not a
    characterized device -- ngspice will not even find a model for it.  The beta ratio and
    the write ratio are therefore *outputs* of the foundry's choice, not knobs:
    `DEVICE_REFERENCE` pins the three drive currents they come from.

  * **`sky130_hdl21` does not have these devices.**  Its `xtors` are the standard
    `nfet_01v8` / `pfet_01v8` families; the `special_*fet_*` cells are absent.  So they are
    wrapped as `h.ExternalModule` with `spicetype=SpiceType.SUBCKT` -- the same move
    `cell_layout` makes for the poly resistor and `crossbar` for its placeholder NMOS.  A
    consequence worth stating: nothing in the path calls `sky130_hdl21.compile`, so the
    SI-to-micron scaling that `crossbar.si_literal` exists to trigger never fires and the
    geometry has to be written in microns by hand.  See `UNIT_NOTE` and `um`.

  * **The deck's DC sweep is unusable, so the butterfly curve is a slow ramp.**
    `vlsirtools` 7.0.0 emits `.dc param start=... stop=... step=...` for ngspice, which is
    not ngspice syntax and drops the swept source's name, and then looks the result up
    under `Plotname: DC Analysis` where ngspice writes `DC transfer characteristic`.  Two
    independent breaks, so rather than patch two internals this module ramps a source
    slowly under `hs.Tran` and checks quasi-staticity by re-running an order of magnitude
    slower -- `vtc_ramp_error` is that check, and it is a stronger statement than trusting
    a DC solver would have been.

Run it:  uv run sram6t.py
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import hdl21 as h
import hdl21.sim as hs
import numpy as np
from vlsirtools.spicetype import SpiceType

import crossbar as C   # sky130_root(), SKY130_LIB_PATH, and the ngspice rawfile shim

VDD = 1.8
CORNERS = ("tt", "ss", "ff")

# --------------------------------------------------------------------------
# Units.  This is the fourth SI-versus-micron boundary in this project; see
# docs/LAYOUT.md 5.1 for the first three.  It is also the first one where the deck
# convention is *not* what the obvious grep says.
# --------------------------------------------------------------------------
UNIT_NOTE = (
    "Geometry in this deck is MICRONS: `w=0.21 l=0.15` reaches the BSIM3 card as W = 2.1e-7 "
    "m, L = 1.5e-7 m.  The `.option scale=1.0u` responsible is NOT in sky130.lib.spice -- "
    "grepping that file finds the option only inside `.lib mc` -- it comes in one level "
    "down, from `corners/<corner>.spice` including `../all.spice` whose second line sets "
    "it.  `all.spice` is also the only file in the tt chain that defines the special_* "
    "subckts at all.  So a Prefixed `0.21 * um` netlists as 2.1e-07, is scaled again to "
    "2.1e-13 m, misses the bin, and dies with nothing but `could not find a valid "
    "modelname` -- exactly hazard #1 in docs/LAYOUT.md 5.1, one layer deeper. "
    "`unit_check` asserts the convention in both directions rather than trusting it."
)

# Drain current at |Vgs| = |Vds| = VDD, in amps, for each device at its characterized
# geometry.  Reference values in the sense of `layout_oracle.SHEET_RES_REFERENCE`: cheap to
# check, and the only thing standing between a silent unit slip and a set of margins that
# look plausible and are wrong.  A 1e-6 geometry error does not perturb these, it removes
# the model entirely -- but a *corner* slip or a model revision would perturb them.
DEVICE_REFERENCE = {"sky130_fd_pr__special_nfet_latch": 91.406e-6,
                    "sky130_fd_pr__special_nfet_pass": 70.566e-6,
                    "sky130_fd_pr__special_pfet_latch": 20.815e-6}


def um(v: float) -> h.Literal:
    """A drawn dimension in microns, as the `h.Literal` this deck wants.

    Deliberately not `v * µ`.  `h.Prefixed` would netlist as SI metres, and because nothing
    here goes through `sky130_hdl21.compile` there is no walker to scale it up again -- so
    the deck's own `.option scale=1.0u` would take the metres as microns and shrink the
    device by 1e6.  See `UNIT_NOTE`.
    """
    return h.Literal(f"{v:g}")


def sky130_lib() -> Path:
    return C.sky130_root() / C.SKY130_LIB_PATH


def scale_provenance(corner: str = "tt") -> tuple:
    """`(value, "a:line -> b:line ...")` for the `.option scale` a corner pulls in.

    Breadth-first over each file's own `.include` lines starting at
    `corners/<corner>.spice`, so the claim in `UNIT_NOTE` is a parse and the *path* to the
    option is part of the answer -- which is the point, since grepping the file the deck
    names does not find it.  Raises if the chain stops setting the option, because that
    would silently reinterpret every dimension in this module by 1e6.
    """
    root = sky130_lib().parent

    def where(path, n):
        try:
            return f"{path.relative_to(root)}:{n}"
        except ValueError:   # volare's sky130A is a symlink; keep the full path
            return f"{path}:{n}"

    start = root / f"corners/{corner}.spice"
    chain, seen = [(start, "")], []
    while chain:
        path, trail = chain.pop(0)
        if not path.is_file() or path in seen:
            continue
        seen.append(path)
        for n, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.split("*")[0].strip()
            m = re.match(r"^\.option[s]?\s+scale\s*=\s*(\S+)", line, re.I)
            if m:
                txt = m.group(1)
                mult = {"u": 1e-6, "n": 1e-9, "m": 1e-3}.get(txt[-1].lower())
                val = float(txt[:-1]) * mult if mult else float(txt)
                return val, f"{trail}{where(path, n)}"
            m = re.match(r'^\.include\s+"?([^"\s]+)"?', line, re.I)
            if m:
                # normpath, not resolve: resolve() walks volare's sky130A symlink and the
                # reported provenance stops being the path the deck actually names.
                nxt = Path(os.path.normpath(path.parent / m.group(1)))
                chain.append((nxt, f"{trail}{where(path, n)} -> "))
    raise RuntimeError(
        f"no `.option scale` anywhere in the include chain of corners/{corner}.spice under "
        f"{root}. This module writes geometry in microns because that chain sets "
        f"scale=1.0u; without it every dimension here is off by 1e6. {UNIT_NOTE}")


# --------------------------------------------------------------------------
# The three devices, and the single bin each of them has.
# --------------------------------------------------------------------------
MODEL_DIR = "libs.ref/sky130_fd_pr/spice"

# The bin limits sit on one continuation line of the `.model` card, in metres because the
# same card says `binunit = 2.0`.  Parsed, not typed: the whole "no sizing exercise" claim
# rests on these four numbers.
BIN_RE = re.compile(r"lmin\s*=\s*(\S+)\s+lmax\s*=\s*(\S+)\s+wmin\s*=\s*(\S+)\s+wmax\s*=\s*(\S+)")


@dataclass(frozen=True)
class ModelBin:
    """One `.model` bin of a device, in microns, with the line it was parsed from."""
    l_min: float
    l_max: float
    w_min: float
    w_max: float
    line: int

    def holds(self, w: float, l: float) -> bool:
        return self.w_min <= w <= self.w_max and self.l_min <= l <= self.l_max

    def __str__(self):
        return (f"W {self.w_min:g}..{self.w_max:g}, L {self.l_min:g}..{self.l_max:g} um "
                f"(line {self.line})")


def model_path(device: str) -> Path:
    p = C.sky130_root() / MODEL_DIR / f"{device}.pm3.spice"
    if not p.is_file():
        raise RuntimeError(f"no model file for {device} at {p}")
    return p


def model_bins(device: str) -> list:
    """Every bin the device's own model file declares, converted to microns.

    `binunit = 2.0` on the same card means the file's limits are metres; this returns
    microns because that is what the deck is written in, so a caller compares like with
    like instead of carrying two conventions.
    """
    path = model_path(device)
    text = path.read_text()
    if "binunit = 2.0" not in text and "binunit=2.0" not in text:
        raise RuntimeError(f"{path}: no `binunit = 2.0`, so the bin limits below are not "
                           "metres and this parse would silently mis-scale them")
    out = []
    for n, raw in enumerate(text.splitlines(), start=1):
        m = BIN_RE.search(raw)
        if m:
            lmin, lmax, wmin, wmax = (float(g) * 1e6 for g in m.groups())
            out.append(ModelBin(l_min=lmin, l_max=lmax, w_min=wmin, w_max=wmax, line=n))
    if not out:
        raise RuntimeError(f"{path}: no `lmin/lmax/wmin/wmax` bin line to parse")
    return out


@h.paramclass
class SramMosParams:
    w = h.Param(dtype=h.Scalar, desc="Channel width, MICRONS in the deck -- see UNIT_NOTE")
    l = h.Param(dtype=h.Scalar, desc="Channel length, MICRONS in the deck")


def _wrap(name: str, desc: str) -> h.ExternalModule:
    """One `special_*fet_*` cell as an hdl21 `ExternalModule`.

    `spicetype=SpiceType.SUBCKT`, not `MOS`: the PDK ships these as `.subckt d g s b`
    wrappers around a binned `.model`, so they netlist with an `x` prefix.  Port order
    follows the subckt's own.
    """
    return h.ExternalModule(
        name=name, desc=desc,
        port_list=[h.Port(name="d"), h.Port(name="g"), h.Port(name="s"), h.Port(name="b")],
        paramtype=SramMosParams, spicetype=SpiceType.SUBCKT)


NFET_LATCH = _wrap("sky130_fd_pr__special_nfet_latch",
                   "SRAM latch pull-down NMOS; one bin, W = 0.21 um")
NFET_PASS = _wrap("sky130_fd_pr__special_nfet_pass",
                  "SRAM access NMOS; one bin, W = 0.14 um")
PFET_LATCH = _wrap("sky130_fd_pr__special_pfet_latch",
                   "SRAM latch pull-up PMOS; one bin, W = 0.14 um")


@dataclass(frozen=True)
class SramDevice:
    """One bitcell device: its role, its characterized geometry, and its wrapper."""
    role: str
    name: str
    w: float          # um
    l: float          # um
    module: h.ExternalModule
    pmos: bool

    @property
    def bins(self) -> list:
        return model_bins(self.name)

    def __str__(self):
        return f"{self.name} ({self.role}, W/L = {self.w:g}/{self.l:g} um)"


DEVICES = (
    SramDevice("pull-down", NFET_LATCH.name, 0.21, 0.15, NFET_LATCH, False),
    SramDevice("access", NFET_PASS.name, 0.14, 0.15, NFET_PASS, False),
    SramDevice("pull-up", PFET_LATCH.name, 0.14, 0.15, PFET_LATCH, True),
)


def check_bins(devices=DEVICES) -> dict:
    """{device: the one bin holding its geometry}; raises if a size leaves its bin.

    "Leaves its bin" is not a style violation, it is an uncharacterized device: ngspice
    answers a W outside `wmin..wmax` with `could not find a valid modelname` and nothing
    else, which is why this runs before any simulation rather than after one fails.
    """
    out = {}
    for d in devices:
        hits = [b for b in d.bins if b.holds(d.w, d.l)]
        if len(hits) != 1:
            raise ValueError(
                f"{d.name} at W/L = {d.w:g}/{d.l:g} um lands in {len(hits)} of its "
                f"{len(d.bins)} model bins ({[str(b) for b in d.bins]}); exactly one is "
                f"required, since a size in no bin has no model and a size in two is "
                f"ambiguous. Model file: {model_path(d.name)}")
        out[d.name] = hits[0]
    return out


# --------------------------------------------------------------------------
# The cell.  Built out of three add-a-piece helpers rather than one flat generator, so an
# 8T/10T variant is `storage_core` + `add_access` + a new read port and not a rewrite:
# the latch and the write port are already separable here.
# --------------------------------------------------------------------------
@h.paramclass
class Sram6TParams:
    w_pd = h.Param(dtype=float, desc="Pull-down NMOS width [um]", default=0.21)
    l_pd = h.Param(dtype=float, desc="Pull-down NMOS length [um]", default=0.15)
    w_pu = h.Param(dtype=float, desc="Pull-up PMOS width [um]", default=0.14)
    l_pu = h.Param(dtype=float, desc="Pull-up PMOS length [um]", default=0.15)
    w_acc = h.Param(dtype=float, desc="Access NMOS width [um]", default=0.14)
    l_acc = h.Param(dtype=float, desc="Access NMOS length [um]", default=0.15)


def _sizes(p: Sram6TParams) -> tuple:
    """`(pull-down, pull-up, access)` param objects, bin-checked against the model files."""
    devs = (SramDevice("pull-down", NFET_LATCH.name, p.w_pd, p.l_pd, NFET_LATCH, False),
            SramDevice("access", NFET_PASS.name, p.w_acc, p.l_acc, NFET_PASS, False),
            SramDevice("pull-up", PFET_LATCH.name, p.w_pu, p.l_pu, PFET_LATCH, True))
    check_bins(devs)
    return (SramMosParams(w=um(p.w_pd), l=um(p.l_pd)),
            SramMosParams(w=um(p.w_pu), l=um(p.l_pu)),
            SramMosParams(w=um(p.w_acc), l=um(p.l_acc)))


def add_inverter(m: h.Module, p: Sram6TParams, inp, out, vdd, vss, suffix: str) -> None:
    """One latch leg: pull-down NMOS and pull-up PMOS sharing a gate and a drain.

    Bodies go to the rails, which is what the drawn cell does too -- the NMOS to the pwell
    tap and the PMOS to the nwell tap -- so the netlist and the layout agree on four
    terminals per device rather than three plus an assumption.
    """
    pd, pu, _ = _sizes(p)
    m.add(NFET_LATCH(pd)(d=out, g=inp, s=vss, b=vss), name=f"mn{suffix}")
    m.add(PFET_LATCH(pu)(d=out, g=inp, s=vdd, b=vdd), name=f"mp{suffix}")


def add_access(m: h.Module, p: Sram6TParams, node, bitline, wl, vss, suffix: str) -> None:
    """One access NMOS from a storage node to a bitline, gated by the wordline.

    Drain on the bitline and source on the storage node by name only: the device is
    symmetric and which terminal is which reverses between a read and a write.
    """
    _, _, acc = _sizes(p)
    m.add(NFET_PASS(acc)(d=bitline, g=wl, s=node, b=vss), name=f"ma{suffix}")


def storage_core(m: h.Module, p: Sram6TParams) -> tuple:
    """`q`, `qb` and the cross-coupled pair.  The part every nT cell has in common.

    Returns the two storage nodes.  An 8T cell calls this, then `add_access` twice for the
    write port, then adds its own read stack off `qb` -- none of which touches this.
    """
    m.q, m.qb = h.Signal(), h.Signal()
    add_inverter(m, p, m.qb, m.q, m.vdd, m.vss, "q")
    add_inverter(m, p, m.q, m.qb, m.vdd, m.vss, "qb")
    return m.q, m.qb


def build_sram6t(p: Sram6TParams = Sram6TParams(), name: str = "sram_6t") -> h.Module:
    """The six devices in one flat module with a caller-chosen name.

    Flat and name-controlled because LVS needs both: the deck pairs the layout's top cell
    with the schematic's *by name* (`layout_oracle.NFET_CELL` says so), and a hierarchical
    schematic would need a matching layout hierarchy the drawn cell does not have.
    """
    m = h.Module(name=name)
    m.bl, m.blb = h.Inout(), h.Inout()
    m.wl = h.Input()
    m.vdd, m.vss = h.Port(), h.Port()
    q, qb = storage_core(m, p)
    add_access(m, p, q, m.bl, m.wl, m.vss, "q")
    add_access(m, p, qb, m.blb, m.wl, m.vss, "qb")
    return m


@h.generator
def Sram6T(p: Sram6TParams) -> h.Module:
    """Generator front end for `build_sram6t`, for use inside a testbench."""
    return build_sram6t(p, name="Sram6T")


def build_half_cell(p: Sram6TParams = Sram6TParams(),
                    name: str = "sram_6t_half") -> h.Module:
    """Half the cell with the feedback broken: one inverter plus its access device.

    This is the butterfly-curve device under test.  It is built from the *same*
    `add_inverter` and `add_access` the whole cell is built from, so the VTC measured here
    is the VTC of the cell's own leg and not of a look-alike.
    """
    m = h.Module(name=name)
    m.inp, m.out, m.bl = h.Inout(), h.Inout(), h.Inout()
    m.wl = h.Input()
    m.vdd, m.vss = h.Port(), h.Port()
    add_inverter(m, p, m.inp, m.out, m.vdd, m.vss, "h")
    add_access(m, p, m.out, m.bl, m.wl, m.vss, "h")
    return m


# --------------------------------------------------------------------------
# Simulation plumbing.
# --------------------------------------------------------------------------
def sim_attrs(corner: str, *analyses) -> list:
    if corner not in ("tt", "ss", "ff", "sf", "fs", "ll", "hh"):
        raise ValueError(f"unknown sky130.lib.spice section {corner!r}")
    return [hs.Lib(path=sky130_lib(), section=corner), *analyses, hs.Literal(".save all")]


def run(tb: h.Module, attrs: list, rundir) -> dict:
    res = hs.Sim(tb=tb, attrs=attrs).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA,
        rundir=str(rundir)))
    return res.an[0].data


def device_current(dev: SramDevice, corner: str = "tt", rundir="/tmp/sram6t/dev",
                   w=None, l=None) -> float:
    """|Id| at |Vgs| = |Vds| = VDD, one `.op`.  The drive strength every ratio comes from.

    `w` / `l` override the geometry and are passed through untouched -- `unit_check` uses
    that to hand the same device an SI-metre size and watch it fail.
    """
    par = SramMosParams(w=um(dev.w) if w is None else w, l=um(dev.l) if l is None else l)
    tb = h.Module(name="Tb")
    tb.VSS = h.Port()
    tb.d, tb.g, tb.s = h.Signal(), h.Signal(), h.Signal()
    if dev.pmos:   # source and body at VDD, gate and drain at 0
        tb.vs = h.Vdc(dc=VDD)(p=tb.s, n=tb.VSS)
        tb.vd = h.Vdc(dc=0.0)(p=tb.d, n=tb.VSS)
        tb.vg = h.Vdc(dc=0.0)(p=tb.g, n=tb.VSS)
        tb.m = dev.module(par)(d=tb.d, g=tb.g, s=tb.s, b=tb.s)
    else:
        tb.vs = h.Vdc(dc=0.0)(p=tb.s, n=tb.VSS)
        tb.vd = h.Vdc(dc=VDD)(p=tb.d, n=tb.VSS)
        tb.vg = h.Vdc(dc=VDD)(p=tb.g, n=tb.VSS)
        tb.m = dev.module(par)(d=tb.d, g=tb.g, s=tb.s, b=tb.VSS)
    data = run(tb, sim_attrs(corner, hs.Op()), rundir)
    return abs(float(data["i(v.xtop.vvd)"]))


def unit_check(rundir="/tmp/sram6t/units") -> dict:
    """Assert the micron convention end-to-end, in both directions.

    docs/LAYOUT.md 5.1's practical rule is to assert a known dimension rather than trust a
    convention, because every failure at this boundary is silent.  Two halves:

      positive  the characterized geometry, written in microns, reproduces
                `DEVICE_REFERENCE` to 0.1%.
      negative  the *same* device with the *same* number expressed as SI metres
                (`h.Prefixed`) does not simulate at all, and ngspice's complaint is the
                modelname one -- i.e. the bin miss, not something else.

    Without the negative half the positive one is weak: a scale factor of 1 and a scale
    factor of 1e-6 both produce *some* current if the bin is wide enough.  These bins are
    0.5% wide, so they do not, and the negative control proves it.
    """
    from hdl21.prefix import µ

    out = {}
    for dev in DEVICES:
        i = device_current(dev, rundir=Path(rundir) / dev.role.replace("-", ""))
        want = DEVICE_REFERENCE[dev.name]
        out[dev.name] = {"i": i, "want": want, "rel": abs(i - want) / want}
    dev = DEVICES[0]
    try:
        device_current(dev, rundir=Path(rundir) / "si", w=dev.w * µ, l=dev.l * µ)
        out["si_negative_control"] = {"failed": False, "modelname": False}
    except Exception as e:
        out["si_negative_control"] = {
            "failed": True,
            "modelname": "could not find a valid modelname" in str(e)}
    return out


# --------------------------------------------------------------------------
# The butterfly curve and SNM.
# --------------------------------------------------------------------------
# Ramp time for the quasi-static VTC, and the slower one `vtc_ramp_error` compares against.
VTC_RAMP = 100e-9
VTC_POINTS = 6001


def vtc(p: Sram6TParams, corner: str, wl_v: float, v_bl: float = VDD,
        tramp: float = VTC_RAMP, points: int = VTC_POINTS, rundir="/tmp/sram6t/vtc") -> tuple:
    """`(vin, vout)` of one half cell, swept by a slow ramp.  See the module docstring.

    `wl_v = VDD, v_bl = VDD` is the read condition: the access device is on and pulling the
    low node up out of the rail, which is the whole reason read SNM is the worse number.
    `wl_v = 0` is hold.
    """
    settle = tramp / 20
    tb = h.Module(name="Tb")
    tb.VSS = h.Port()
    tb.vdd, tb.inp, tb.out, tb.bl, tb.wl = (h.Signal() for _ in range(5))
    tb.vvdd = h.Vdc(dc=VDD)(p=tb.vdd, n=tb.VSS)
    tb.vbl = h.Vdc(dc=v_bl)(p=tb.bl, n=tb.VSS)
    tb.vwl = h.Vdc(dc=wl_v)(p=tb.wl, n=tb.VSS)
    tb.vin = h.Vpulse(v1=0.0, v2=VDD, delay=settle, rise=tramp, fall=tramp,
                      width=tramp, period=10 * tramp)(p=tb.inp, n=tb.VSS)
    tb.dut = build_half_cell(p)(inp=tb.inp, out=tb.out, bl=tb.bl, wl=tb.wl,
                                vdd=tb.vdd, vss=tb.VSS)
    tstop = settle + tramp
    data = run(tb, sim_attrs(corner, hs.Tran(tstop=tstop, tstep=tstop / points)), rundir)
    t = np.asarray(data["time"])
    keep = t >= settle
    return np.asarray(data["v(xtop.inp)"])[keep], np.asarray(data["v(xtop.out)"])[keep]


def on_grid(vin, vout, n: int = 361) -> tuple:
    """Resample a ramp-swept VTC onto a monotone `vin` grid, for comparing two of them."""
    o = np.argsort(vin)
    grid = np.linspace(0.0, VDD, n)
    return grid, np.interp(grid, vin[o], vout[o])


@dataclass(frozen=True)
class RampError:
    """How much slowing the VTC ramp down by `factor` moves the answer."""
    factor: float
    d_vtc: float         # V, max vertical |VTC(fast) - VTC(slow)|
    d_snm: float         # V, and the difference it makes to the SNM
    snm_fast: float
    snm_slow: float

    def __str__(self):
        return (f"x{self.factor:g} slower: SNM moves {self.d_snm * 1e6:.1f} uV "
                f"({self.snm_fast * 1e3:.2f} -> {self.snm_slow * 1e3:.2f} mV); raw vertical "
                f"VTC gap {self.d_vtc * 1e3:.2f} mV")


def vtc_ramp_error(p: Sram6TParams, corner: str, wl_v: float, factor: float = 10.0,
                   rundir="/tmp/sram6t/ramp") -> RampError:
    """Re-run the VTC `factor` times slower and report what moved.

    A ramp stands in for a DC sweep, and the honest way to size that substitution's error is
    to make the ramp slower and see how much the answer moves; the dynamic term is
    displacement current through the device capacitances and falls as 1/tramp.

    The acceptance number is the change in SNM, not the pointwise VTC gap, because **no
    pointwise gap is well conditioned on a VTC**.  Compared vertically, the high-gain region
    turns a sub-mV difference in where the two runs' adaptive timesteps landed into tens of
    mV of apparent error; compared horizontally, the two flat regions do the same thing in
    reverse, and worse, since there the curve is nearly horizontal.  SNM is both the quantity
    this module exists to report and a functional of the whole curve, so its movement is the
    error bar that means something.  The raw vertical gap is reported anyway, to be looked at
    rather than quoted.
    """
    fast_raw = vtc(p, corner, wl_v, tramp=VTC_RAMP, rundir=Path(rundir) / "fast")
    slow_raw = vtc(p, corner, wl_v, tramp=factor * VTC_RAMP, rundir=Path(rundir) / "slow")
    _, fast = on_grid(*fast_raw)
    _, slow = on_grid(*slow_raw)
    a, b = snm(*fast_raw), snm(*slow_raw)
    return RampError(factor=factor, d_vtc=float(np.abs(fast - slow).max()),
                     d_snm=abs(a.snm - b.snm), snm_fast=a.snm, snm_slow=b.snm)


@dataclass(frozen=True)
class Snm:
    """One lobe of the butterfly, and the square that measures it."""
    snm: float           # V, the square's side
    x0: float            # V, the square's low corner in q
    y0: float            # V, ... and in qb
    v_low: float         # V, the VTC's low level -- read disturb lifts it off the rail
    v_high: float        # V, and its high level, which a weak pull-up lets droop
    v_trip: float        # V, where the VTC crosses vout = vin
    gain: float          # max |dvout/dvin|

    def __str__(self):
        return (f"SNM {self.snm * 1e3:.1f} mV (square at q = {self.x0:.3f}, "
                f"qb = {self.y0:.3f} V); VTC {self.v_low * 1e3:.1f} mV .. "
                f"{self.v_high:.4f} V, trip {self.v_trip:.3f} V, max gain {self.gain:.1f}")


def snm(vin, vout, n: int = 4001) -> Snm:
    """Side of the largest axis-aligned square inside a lobe of the butterfly curve.

    The butterfly is the measured VTC `qb = f(q)` together with its mirror `q = f(qb)`.  In
    the upper-left lobe `f` is the upper boundary and `f^-1` the lower one, and both fall
    with q, so an axis-aligned square `[x0, x0+s] x [y0, y0+s]` fits exactly when

        y0 + s <= f(x0 + s)        (the top-right corner, where `f` is lowest)
        y0     >= f^-1(x0)         (the bottom-left corner, where `f^-1` is highest)

    Pushing both to equality makes `s` the root of `s = f(x0+s) - f^-1(x0)`, decreasing in
    `s`, so a bisection per `x0` and a max over `x0` is the answer with no rotation
    bookkeeping.  Two consequences of the derivation worth keeping in mind: the square's
    diagonal runs at +45 degrees between the two curves, which is the same square
    Seevinck's rotate-by-45-and-take-the-maximum-distance construction finds; and the
    containment argument is exact for these monotone boundaries, so `square_fits` can check
    it rather than assume it.

    The cell's two legs are identical, so the mirror is an exact reflection about q = qb and
    the two lobes are congruent: this returns the one number, and `verify_sram6t` asserts
    the congruence instead of reporting the same measurement twice.
    """
    o = np.argsort(vin)
    x, y = np.asarray(vin)[o], np.asarray(vout)[o]
    yi = np.argsort(y)

    def f(t):
        return np.interp(t, x, y)

    def finv(t):
        return np.interp(t, y[yi], x[yi])

    best, corner = 0.0, (0.0, 0.0)
    for x0 in np.linspace(x.min(), x.max(), n):
        lo, hi = 0.0, min(f(x0) - finv(x0), x.max() - x0)
        if hi <= 0:
            continue
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if f(x0 + mid) - finv(x0) - mid >= 0:
                lo = mid
            else:
                hi = mid
        if lo > best:
            best, corner = lo, (float(x0), float(finv(x0)))
    slope = np.abs(np.gradient(y, x))
    return Snm(snm=float(best), x0=corner[0], y0=corner[1], v_low=float(y.min()),
               v_high=float(y.max()),
               v_trip=float(np.interp(0.0, (y - x)[::-1], x[::-1])),
               gain=float(np.nanmax(slope[np.isfinite(slope)])))


def square_fits(vin, vout, side: float, x0: float, y0: float, n: int = 101) -> bool:
    """Is the square of side `side` at `(x0, y0)` inside the lobe?  Checked, not assumed.

    Samples the square's boundary and asks that every point lie between the two curves.  An
    optimizer that returns a number nothing checks is a number.
    """
    o = np.argsort(vin)
    x, y = np.asarray(vin)[o], np.asarray(vout)[o]
    yi = np.argsort(y)
    xs = np.linspace(x0, x0 + side, n)
    upper, lower = np.interp(xs, x, y), np.interp(xs, y[yi], x[yi])
    tol = 1e-6
    return bool(np.all(y0 + side <= upper + tol) and np.all(y0 >= lower - tol))


def read_snm(p: Sram6TParams, corner: str, rundir="/tmp/sram6t/snm") -> Snm:
    return snm(*vtc(p, corner, wl_v=VDD, v_bl=VDD, rundir=Path(rundir) / f"read_{corner}"))


def hold_snm(p: Sram6TParams, corner: str, rundir="/tmp/sram6t/snm") -> Snm:
    return snm(*vtc(p, corner, wl_v=0.0, v_bl=VDD, rundir=Path(rundir) / f"hold_{corner}"))


# --------------------------------------------------------------------------
# Write, hold and read-upset transients.  One three-phase testbench does all three, because
# they are the same circuit at three bitline biases:
#
#   phase 1  wl high, bl = 0,   blb = VDD   -> write q = 0 unconditionally
#   phase 2  wl low,  bl = VDD, blb = VDD   -> hold, with both bitlines precharged
#   phase 3  wl high, bl = VDD, blb = v3    -> v3 = 0 writes, v3 = VDD is a read
#
# Phase 2 is the hold test and phase 3 at `v3 = VDD` is the read-disturb test, so nothing
# here needs `.ic`: every state is reached by driving the cell the way a real array would.
# --------------------------------------------------------------------------
T_PHASE = 2e-9


def cell_tran(p: Sram6TParams, v3, corner: str = "tt", t_phase: float = T_PHASE,
              points: int = 3000, rundir="/tmp/sram6t/tran") -> dict:
    """The three-phase sequence above, with one independent cell per entry of `v3`.

    `v3` is a sequence of phase-3 `blb` voltages, and each gets its own copy of the cell and
    its own `blb` source; `vdd`, `wl` and `bl` are shared because every copy sees the same
    waveform on them.  The copies do not interact -- they share only forced nodes -- so a
    28-cell deck is 28 independent experiments, and the point of doing it that way is that
    ngspice spends ~11 s parsing this model library and ~1 ms solving the circuit.  A
    bisection would pay that 11 s a dozen times over for one number.
    """
    v3 = [float(v) for v in np.atleast_1d(v3)]
    tr, big = t_phase / 20, 100 * t_phase
    tb = h.Module(name="Tb")
    tb.VSS = h.Port()
    tb.vdd, tb.bl, tb.wl = (h.Signal() for _ in range(3))
    tb.vvdd = h.Vdc(dc=VDD)(p=tb.vdd, n=tb.VSS)
    tb.vwl = h.Vpulse(v1=VDD, v2=0.0, delay=t_phase, rise=tr, fall=tr, width=t_phase,
                      period=big)(p=tb.wl, n=tb.VSS)
    tb.vbl = h.Vpulse(v1=0.0, v2=VDD, delay=t_phase, rise=tr, fall=tr, width=big,
                      period=2 * big)(p=tb.bl, n=tb.VSS)
    for k, v in enumerate(v3):
        blb = tb.add(h.Signal(), name=f"blb{k}")
        tb.add(h.Vpulse(v1=VDD, v2=v, delay=2 * t_phase, rise=tr, fall=tr, width=big,
                        period=2 * big)(p=blb, n=tb.VSS), name=f"vblb{k}")
        tb.add(Sram6T(p)(bl=tb.bl, blb=blb, wl=tb.wl, vdd=tb.vdd, vss=tb.VSS),
               name=f"dut{k}")
    tstop = 3 * t_phase
    return run(tb, sim_attrs(corner, hs.Tran(tstop=tstop, tstep=tstop / points)), rundir)


@dataclass(frozen=True)
class CellTran:
    """What the three-phase run says about hold, write and read disturb."""
    v3: float
    corner: str
    t_phase: float
    q_hold: float        # V, q at the end of phase 2
    qb_hold: float
    q_hold_drift: float  # V, worst excursion of q across phase 2
    q_end: float         # V, q at the end of phase 3
    qb_end: float
    q_peak3: float       # V, worst excursion of q during phase 3 -- the read disturb bump
    flipped: bool

    def __str__(self):
        return (f"blb = {self.v3:.4f} V, {self.corner}: hold q/qb = {self.q_hold:.4f}/"
                f"{self.qb_hold:.4f}, phase-3 q peak {self.q_peak3:.4f}, end q/qb = "
                f"{self.q_end:.4f}/{self.qb_end:.4f} -> {'FLIP' if self.flipped else 'held'}")


def cell_phases(p: Sram6TParams, v3, corner: str = "tt", t_phase: float = T_PHASE,
                rundir="/tmp/sram6t/tran") -> list:
    """`cell_tran` reduced to the numbers the checks are about, one entry per `v3`."""
    v3 = [float(v) for v in np.atleast_1d(v3)]
    data = cell_tran(p, v3, corner, t_phase, rundir=rundir)
    t = np.asarray(data["time"])
    # Sample each phase after its wordline edge has settled, so nothing below is measuring a
    # transition.  The edges are one twentieth of a phase; a tenth of a phase is comfortable.
    ph2 = (t >= 1.1 * t_phase) & (t <= 2.0 * t_phase)
    ph3 = (t >= 2.1 * t_phase) & (t <= 3.0 * t_phase)
    out = []
    for k, v in enumerate(v3):
        q = np.asarray(data[f"v(xtop.xdut{k}.q)"])
        qb = np.asarray(data[f"v(xtop.xdut{k}.qb)"])
        out.append(CellTran(
            v3=v, corner=corner, t_phase=t_phase,
            q_hold=float(q[ph2][-1]), qb_hold=float(qb[ph2][-1]),
            q_hold_drift=float(np.abs(q[ph2] - q[ph2][0]).max()),
            q_end=float(q[-1]), qb_end=float(qb[-1]), q_peak3=float(q[ph3].max()),
            flipped=bool(q[-1] > VDD / 2 and qb[-1] < VDD / 2)))
    return out


@dataclass(frozen=True)
class WriteMargin:
    """The highest bitline voltage that still flips the cell, and the run that found it."""
    corner: str
    v_write: float       # V, highest blb that writes
    v_fail: float        # V, lowest blb tried that did not
    margin: float        # V, VDD - v_write, the conventional "write margin"
    resolution: float    # V, the surviving bracket width
    t_phase: float
    stages: int
    monotone: bool       # every tried voltage below `v_write` wrote and every one above did not
    at_vdd: "CellTran"   # the blb = VDD point, which is the read-disturb experiment

    def __str__(self):
        return (f"{self.corner}: writes at blb <= {self.v_write:.4f} V, fails at "
                f"{self.v_fail:.4f} V -> write margin {self.margin * 1e3:.1f} mV "
                f"(bracket {self.resolution * 1e3:.3f} mV, {self.stages} stages, "
                f"{'monotone' if self.monotone else 'NOT MONOTONE'})")


def write_margin(p: Sram6TParams, corner: str = "tt", t_phase: float = T_PHASE,
                 n: int = 25, stages: int = 3, rundir="/tmp/sram6t/wm") -> WriteMargin:
    """Grid-refine the phase-3 `blb` voltage down to the flip boundary.

    A flip is a discontinuous function of the bitline voltage -- there is no partial write --
    so the measurement is a boundary hunt.  It is done as `stages` refinements of an
    `n`-point grid rather than a bisection because `cell_tran` puts the whole grid in one
    deck: `stages` simulations for a resolution of `VDD / (n - 1)**stages` instead of
    `log2(VDD / tol)` of them.

    Stage 1 spans [0, VDD] inclusive, so `blb = 0` must write and `blb = VDD` must not; both
    are asserted, because a cell that cannot be written at 0 V or that flips with both
    bitlines high is broken in a way a boundary hunt would happily report a number for.
    `monotone` additionally asserts that the whole grid splits cleanly at the boundary, which
    is what makes the refinement legitimate.
    """
    a, b, monotone, at_vdd, res = 0.0, VDD, True, None, VDD
    for s in range(stages):
        grid = np.linspace(a, b, n)
        rows = cell_phases(p, grid, corner, t_phase, Path(rundir) / f"{corner}_{s}")
        if s == 0:
            if not rows[0].flipped:
                raise AssertionError(f"{corner}: the cell does not write even with blb at "
                                     f"0 V: {rows[0]}")
            if rows[-1].flipped:
                raise AssertionError(f"{corner}: the cell flips with both bitlines at VDD, "
                                     f"i.e. it fails read stability outright: {rows[-1]}")
            at_vdd = rows[-1]
        flips = [r.flipped for r in rows]
        k = max(i for i, f in enumerate(flips) if f)
        monotone = monotone and all(flips[:k + 1]) and not any(flips[k + 1:])
        a, b = float(grid[k]), float(grid[k + 1])
        res = b - a
    return WriteMargin(corner=corner, v_write=a, v_fail=b, margin=VDD - a, resolution=res,
                       t_phase=t_phase, stages=stages, monotone=monotone, at_vdd=at_vdd)


def hold_tran(p: Sram6TParams, corner: str = "tt", t_write: float = T_PHASE,
              t_hold: float = 200e-9, points: int = 4000,
              rundir="/tmp/sram6t/hold") -> dict:
    """Write q = 0, drop the wordline, then sit there for `t_hold` with bl = blb = VDD.

    Two orders of magnitude longer than the write phase, so this is a statement about
    retention against subthreshold leakage through the off access devices and not just
    about the write settling.
    """
    tr = t_write / 20
    tstop = t_write + t_hold
    tb = h.Module(name="Tb")
    tb.VSS = h.Port()
    tb.vdd, tb.bl, tb.blb, tb.wl = (h.Signal() for _ in range(4))
    tb.vvdd = h.Vdc(dc=VDD)(p=tb.vdd, n=tb.VSS)
    tb.vwl = h.Vpulse(v1=VDD, v2=0.0, delay=t_write, rise=tr, fall=tr, width=10 * tstop,
                      period=20 * tstop)(p=tb.wl, n=tb.VSS)
    tb.vbl = h.Vpulse(v1=0.0, v2=VDD, delay=t_write, rise=tr, fall=tr, width=10 * tstop,
                      period=20 * tstop)(p=tb.bl, n=tb.VSS)
    tb.vblb = h.Vdc(dc=VDD)(p=tb.blb, n=tb.VSS)
    tb.dut = Sram6T(p)(bl=tb.bl, blb=tb.blb, wl=tb.wl, vdd=tb.vdd, vss=tb.VSS)
    return run(tb, sim_attrs(corner, hs.Tran(tstop=tstop, tstep=tstop / points)), rundir)


@dataclass(frozen=True)
class Hold:
    corner: str
    t_hold: float
    q_start: float
    q_end: float
    qb_start: float
    qb_end: float
    q_max: float         # V, the worst q gets over the whole hold window
    qb_min: float        # V, ... and the worst qb gets

    @property
    def drift(self) -> float:
        return max(abs(self.q_end - self.q_start), abs(self.qb_end - self.qb_start))

    def __str__(self):
        return (f"{self.corner}, {self.t_hold * 1e9:g} ns: q {self.q_start * 1e3:.3f} -> "
                f"{self.q_end * 1e3:.3f} mV (max {self.q_max * 1e3:.3f}), qb "
                f"{self.qb_start:.5f} -> {self.qb_end:.5f} V (min {self.qb_min:.5f}); "
                f"worst drift {self.drift * 1e6:.1f} uV")


def hold(p: Sram6TParams, corner: str = "tt", t_write: float = T_PHASE,
         t_hold: float = 200e-9, rundir="/tmp/sram6t/hold") -> Hold:
    data = hold_tran(p, corner, t_write, t_hold, rundir=Path(rundir) / corner)
    t = np.asarray(data["time"])
    q, qb = np.asarray(data["v(xtop.xdut.q)"]), np.asarray(data["v(xtop.xdut.qb)"])
    # Start half a write phase after the wordline edge: closer than that and `drift` is
    # measuring the edge's own capacitive kick on the storage nodes rather than retention.
    keep = t >= 1.5 * t_write
    q, qb = q[keep], qb[keep]
    return Hold(corner=corner, t_hold=t_hold, q_start=float(q[0]), q_end=float(q[-1]),
                qb_start=float(qb[0]), qb_end=float(qb[-1]), q_max=float(q.max()),
                qb_min=float(qb.min()))


def read_upset(p: Sram6TParams, corner: str = "tt", t_phase: float = T_PHASE,
               rundir="/tmp/sram6t/read") -> CellTran:
    """Phase 3 with both bitlines at VDD: the read the beta ratio exists to survive.

    The access device on the low node is a current source into it from a bitline at VDD; the
    pull-down has to sink that without letting the node rise past the other inverter's trip
    point.  `q_peak3` is how far up it actually gets.

    `write_margin` already runs this point as its upper bracket, so a caller doing both
    should take `WriteMargin.at_vdd` rather than pay for a second simulation.
    """
    return cell_phases(p, VDD, corner, t_phase, Path(rundir) / corner)[0]


# --------------------------------------------------------------------------
# One pass over everything, so nothing gets simulated twice.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CornerResult:
    corner: str
    read: Snm
    hold: Snm
    write: WriteMargin

    @property
    def upset(self) -> CellTran:
        """The read-disturb run.  It is the write sweep's `blb = VDD` point, not a rerun."""
        return self.write.at_vdd


@dataclass(frozen=True)
class Margins:
    """Every number Stage A measures, in one object.

    Built once by `measure` and read by both `main` and `verify_sram6t`, because at ~11 s of
    library parsing per ngspice invocation a check suite that re-simulates per check would
    cost half an hour to say the same thing.
    """
    scale: tuple
    bins: dict
    units: dict
    ramp: RampError
    corners: dict        # corner -> CornerResult, the required set
    hold: Hold
    skew: dict           # corner -> Snm, read SNM at the two *skewed* corners
    currents: dict       # corner -> {role: |Id| at |Vgs| = |Vds| = VDD}

    @property
    def i_pd(self) -> float:
        return self.units[DEVICES[0].name]["i"]

    @property
    def i_acc(self) -> float:
        return self.units[DEVICES[1].name]["i"]

    @property
    def i_pu(self) -> float:
        return self.units[DEVICES[2].name]["i"]

    @property
    def beta(self) -> float:
        return self.i_pd / self.i_acc

    @property
    def write_ratio(self) -> float:
        return self.i_acc / self.i_pu

    @property
    def worst_read_corner(self) -> str:
        both = {**{c: r.read for c, r in self.corners.items()}, **self.skew}
        return min(both, key=lambda c: both[c].snm)

    def read_snm(self, corner: str) -> Snm:
        return self.corners[corner].read if corner in self.corners else self.skew[corner]


# ff/ss are magnitude corners, not skewed ones: they move NMOS and PMOS the same way.  The
# brief asks for read SNM at tt/ss/ff and separately says the worst read SNM lives at a
# skewed corner -- which those three cannot show, since none of them is skewed.  So sf and fs
# are measured too, for read SNM only, and reported as an addition rather than folded in.
SKEW_CORNERS = ("sf", "fs")


def measure(p: Sram6TParams = Sram6TParams(), corners=CORNERS, skew=SKEW_CORNERS,
            rundir=Path("/tmp/sram6t")) -> Margins:
    """Run the whole Stage-A measurement set once."""
    rundir = Path(rundir)
    units = unit_check(rundir / "units")
    # The corner names say `sf`/`fs` but not which letter is which device, and the SNM table
    # is unreadable without knowing.  Measured rather than inferred from the name.
    currents = {"tt": {d.role: units[d.name]["i"] for d in DEVICES}}
    for c in [c for c in (*corners, *skew) if c != "tt"]:
        currents[c] = {d.role: device_current(d, corner=c,
                                              rundir=rundir / "idrive" / f"{c}_{d.role}")
                       for d in DEVICES}
    return Margins(
        scale=scale_provenance("tt"),
        bins=check_bins(),
        units=units,
        ramp=vtc_ramp_error(p, "tt", VDD, rundir=rundir / "ramp"),
        corners={c: CornerResult(corner=c, read=read_snm(p, c, rundir / "snm"),
                                 hold=hold_snm(p, c, rundir / "snm"),
                                 write=write_margin(p, c, rundir=rundir / "wm"))
                 for c in corners},
        hold=hold(p, "tt", rundir=rundir / "hold"),
        skew={c: read_snm(p, c, rundir / "snm") for c in skew},
        currents=currents)


def main(rundir: Path = Path("/tmp/sram6t")):
    import shutil
    import time

    t_start = time.perf_counter()
    rundir = Path(rundir)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    p = Sram6TParams()
    mg = measure(p, rundir=rundir)

    print("=" * 78)
    print("STAGE A  6T SRAM bitcell in sky130: the devices, the cell, and its margins")
    print("=" * 78)
    print(f"library  {sky130_lib()}")
    print(f"scale    {mg.scale[0]:g} (i.e. geometry in um), from {mg.scale[1]}")

    print("\n" + "-" * 78)
    print("the three devices, and the bins they are pinned to")
    print("-" * 78)
    for d in DEVICES:
        others = [b for b in d.bins if b != mg.bins[d.name]]
        print(f"  {d.role:<10s} {d.name}")
        print(f"             W/L = {d.w:g}/{d.l:g} um in bin {mg.bins[d.name]}"
              + (f"; the file's other bin: {', '.join(str(b) for b in others)}"
                 if others else "; the file declares no other bin"))
    print(f"  bin limits parsed from {model_path(DEVICES[0].name).parent}/"
          "*.pm3.spice, `binunit = 2.0` so they are metres")
    print("  W is single-valued for all three, so there is no sizing exercise: a geometry")
    print("  outside a bin is not a slower device, it is an absent model.  L is not: the two")
    print("  NMOS bins accept 0.075..0.1505 um and the PMOS has a second short-channel bin")
    print("  at 0.0245..0.0805 um.  0.15 um is the length the foundry's own bitcell draws.")

    print("\n" + "-" * 78)
    print("units, asserted end to end in both directions")
    print("-" * 78)
    for d in DEVICES:
        r = mg.units[d.name]
        print(f"  {d.role:<10s} |Id| at |Vgs|=|Vds|={VDD} V: {r['i'] * 1e6:8.3f} uA "
              f"(reference {r['want'] * 1e6:.3f}, {100 * r['rel']:.3f}% off)")
    neg = mg.units["si_negative_control"]
    print(f"  negative control: the pull-down written as SI metres "
          f"({'refused' if neg['failed'] else 'SIMULATED -- unexpected'}"
          f"{', modelname error' if neg['modelname'] else ''})")
    print(f"  beta = pull-down / access = {mg.beta:.3f}  (read stability)")
    print(f"  access / pull-up          = {mg.write_ratio:.3f}  (write margin)")

    print("\n" + "-" * 78)
    print("the cell")
    print("-" * 78)
    m = build_sram6t(p)
    print(f"  {m.name}: ports {', '.join(s.name for s in m.ports.values())}; "
          f"{len(m.instances)} devices")
    for name, inst in m.instances.items():
        print(f"    {name:<6s} {inst.of.module.name:<38s} "
              + ", ".join(f"{k}={v.name}" for k, v in inst.conns.items()))

    print("\n" + "-" * 78)
    print("butterfly curves and static noise margin, tt")
    print("-" * 78)
    print(f"  the ramp is standing in for a DC sweep; {mg.ramp}")
    print("  the vertical gap is the misleading one -- it is a sub-mV horizontal sampling")
    print(f"  difference multiplied by a gain of {mg.corners['tt'].read.gain:.0f}. What the "
          "measurement is for moves by uV.")
    tt = mg.corners["tt"]
    print(f"  hold  (wl low,  bl = blb = {VDD} V)  {tt.hold}")
    print(f"  read  (wl high, bl = blb = {VDD} V)  {tt.read}")
    print(f"  read SNM is {100 * (1 - tt.read.snm / tt.hold.snm):.0f}% below hold SNM, which "
          "is the access device lifting the low\n"
          f"  node off the rail: {tt.hold.v_low * 1e3:.1f} mV holding, "
          f"{tt.read.v_low * 1e3:.1f} mV reading.")

    print("\n" + "-" * 78)
    print("hold: write a 0, drop the wordline, wait")
    print("-" * 78)
    print(f"  {mg.hold}")

    print("\n" + "-" * 78)
    print("read stability: wordline high, both bitlines at VDD, value must survive")
    print("-" * 78)
    ru = tt.upset
    print(f"  {ru}")
    print(f"  the stored 0 rises to {ru.q_peak3 * 1e3:.1f} mV during the read and comes "
          f"back; the other\n  inverter trips at {tt.hold.v_trip:.3f} V, so the margin "
          f"against upset is {(tt.hold.v_trip - ru.q_peak3) * 1e3:.0f} mV.")
    print(f"  that peak is the read VTC's low level to "
          f"{abs(ru.q_peak3 - tt.read.v_low) * 1e6:.1f} uV -- the transient and the DC "
          "butterfly\n  are measuring the same thing two ways, which is the cheapest "
          "cross-check available here.")

    print("\n" + "-" * 78)
    print("write margin: highest blb that still flips the cell")
    print("-" * 78)
    print(f"  {tt.write}")

    print("\n" + "-" * 78)
    print("corners.  tt/ss/ff are the asked-for set; sf/fs are added because neither ss nor")
    print("ff is a *skewed* corner -- they move NMOS and PMOS together -- and the skewed")
    print("ones are where read SNM is expected to be worst.  What the letters mean is")
    print("measured below, not read off the name.")
    print("-" * 78)
    print(f"  {'corner':<8s}{'I pd':>8s}{'I acc':>8s}{'I pu':>8s}{'beta':>7s}{'acc/pu':>8s}"
          f"{'read SNM':>10s}{'hold SNM':>10s}{'read low':>10s}{'trip':>8s}{'gain':>7s}"
          f"{'write V':>9s}{'WM':>8s}")
    print(f"  {'':<8s}{'[uA]':>8s}{'[uA]':>8s}{'[uA]':>8s}{'':>7s}{'':>8s}"
          f"{'[mV]':>10s}{'[mV]':>10s}{'[mV]':>10s}{'[V]':>8s}{'':>7s}{'[V]':>9s}{'[mV]':>8s}")
    for c in (*mg.corners, *mg.skew):
        i = mg.currents[c]
        pd, ac, pu = i["pull-down"], i["access"], i["pull-up"]
        s = mg.read_snm(c)
        head = (f"  {c:<8s}{pd * 1e6:>8.1f}{ac * 1e6:>8.1f}{pu * 1e6:>8.1f}"
                f"{pd / ac:>7.3f}{ac / pu:>8.3f}{s.snm * 1e3:>10.1f}")
        if c in mg.corners:
            r = mg.corners[c]
            print(head + f"{r.hold.snm * 1e3:>10.1f}{s.v_low * 1e3:>10.1f}"
                         f"{s.v_trip:>8.3f}{s.gain:>7.1f}{r.write.v_write:>9.4f}"
                         f"{r.write.margin * 1e3:>8.1f}")
        else:
            print(head + f"{'':>10s}{s.v_low * 1e3:>10.1f}{s.v_trip:>8.3f}"
                         f"{s.gain:>7.1f}{'':>9s}{'':>8s}   read SNM only")
    worst = mg.worst_read_corner
    print(f"  worst read SNM at {worst}: {mg.read_snm(worst).snm * 1e3:.1f} mV, "
          f"{100 * (1 - mg.read_snm(worst).snm / tt.read.snm):.0f}% below tt")
    print(f"  beta hardly moves with corner ({min(mg.currents[c]['pull-down'] / mg.currents[c]['access'] for c in mg.currents):.3f}"
          f"..{max(mg.currents[c]['pull-down'] / mg.currents[c]['access'] for c in mg.currents):.3f}) "
          "because both devices in it are NMOS, so the read low\n  level barely moves either. "
          "What the corners actually move is the pull-up, and the\n  weakest one "
          f"({worst}, acc/pu = {mg.currents[worst]['access'] / mg.currents[worst]['pull-up']:.2f}) "
          "is the worst read SNM.")

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s, artifacts in {rundir}")


if __name__ == "__main__":
    main()
