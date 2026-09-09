"""Crossbar interconnect skeleton, and the parasitics the project had been assuming.

docs/REPORT.md opens by saying `r_row` / `r_col` / `c_row` / `c_col` are assumed values
swept over a plausible range, and that pinning them to real geometry is the missing step.
This module takes that step: it draws the N x N interconnect -- row rails on metal1,
column rails on metal2, one via tap per cell -- runs the sky130A DRC deck on it, derives
R and C per cell pitch from the drawn dimensions and the parsed tech constants, and re-runs
the published int8 accuracy study with those numbers in place of the assumed ones.

No memory device is drawn, for the blunt reason that **sky130 has no resistive-memory
device**: `libs.ref` holds only the standard-cell, IO, primitive and SRAM libraries.  That
is not a gap in the skeleton, it is the scope: `r_row` / `r_col` / `c_row` / `c_col` *are*
interconnect parameters, and the interconnect is what a crossbar's layout mostly is.  The
cell itself enters only through `g_min`, which turns out to dominate the pitch -- see
`gmin_serpentine`.

Three things this also settles, all of which the project had to leave open:

  * **Uniform per-pitch R is not a discovery, it is a tautology** for uniform rails, and
    `per_pitch` says so by construction.  The interesting deviations are elsewhere.
  * **Via resistance at the cell tap.** `crossbar.Crossbar` has no such term.  Quantified
    in `tap_resistance`, with a verdict.
  * **Rail-to-rail coupling capacitance.** `crossbar.Crossbar` has shunt C to `vss` and
    nothing else.  `coupling_report` puts numbers on what that leaves out, and the answer
    is that the C model is structurally incomplete rather than merely imprecise.

Run it:  uv run array_layout.py            (skeleton + parasitics + analyses, no ngspice)
         uv run array_layout.py --accuracy (adds the int8 re-run; needs ngspice)
"""
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import klayout.db as kdb
import numpy as np

import crossbar as C
import fp_matmul as F
import int8_matmul as Q
import layout_oracle as O

# The published accuracy configuration, copied from scripts/int8_accuracy.py so the numbers
# below drop straight into the docs/MATMUL.md 3.2 tables instead of being a new experiment.
TILE = (16, 16)          # rows x cols; 16 x 32 physical with the signed column pairs
SHAPE = (16, 32, 8)      # A (M,N) @ B (N,K)
SEED = 0
ASSUMED_R = (0.0, 0.25, 1.0, 5.0)   # the r_wire values docs/MATMUL.md reports

# In-array access-FET Ron and the g_max compression it causes, both measured, both from
# docs/REPORT.md 5.2.  `tap_resistance` asks what adding contact resistance does to them.
RON_IN_ARRAY = 819.0
COMPRESSION_G_MAX = 7.569


# --------------------------------------------------------------------------
# The skeleton.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Skeleton:
    """An N x N crossbar interconnect skeleton, all dimensions in microns.

    Row rails run in x on metal1; column rails run in y on metal2 and simply cross over
    them, which is what puts a crossbar's two rail families on two layers in the first
    place.  Each cell's tap is one metal1->metal2 via off the row rail onto a small metal2
    island, offset in x so it clears the column rail.  The gap between that island and the
    column rail is where the memory device would sit; nothing is drawn in it.

    One via per cell, on the row side only, is a deliberate choice and not the only one: a
    1T1R cell needs a further mcon/licon stack down to the access transistor.  Both routes
    are priced in `tap_resistance` from the tech file's contact resistances.
    """
    n: int = 16
    pitch: float = 1.0
    w_row: float = 0.32      # metal1; >= 0.32 lets the via land on the rail with no pad
    w_col: float = 0.14      # metal2; min width, nothing lands on it
    via: float = 0.15        # via.1a is an exact width, not a minimum
    pad: float = 0.32        # metal2 island around the via
    gap: float = 0.14        # metal2 island to column rail

    row_layer = "met1"
    col_layer = "met2"
    cut_layer = "via"

    def check(self, deck: Path = None):
        """Refuse a parameter set that cannot be DRC clean, naming the rule.

        The DRC run is still the authority -- this only turns an unhelpful violation list
        into a message that says which knob to turn.  Every threshold is read out of the
        deck by `layout_oracle.drc_rule_value`.
        """
        r = lambda name: O.drc_rule_value(name, deck)
        for what, got, want, rule in (
            ("metal1 row rail width", self.w_row, r("m1.1"), "m1.1"),
            ("metal2 col rail width", self.w_col, r("m2.1"), "m2.1"),
            ("metal1 row rail spacing", self.pitch - self.w_row, r("m1.2"), "m1.2"),
            ("via width", self.via, r("via.1a"), "via.1a"),
            ("metal1 enclosure of via, 2 opposite edges",
             self.w_row - self.via, 2 * r("via1.5a"), "via1.5a"),
            ("metal2 enclosure of via, 2 opposite edges",
             self.pad - self.via, 2 * r("m2.5"), "m2.5"),
            ("metal2 island area", self.pad ** 2, r("m2.6"), "m2.6"),
            ("metal2 island to column rail", self.gap, r("m2.2"), "m2.2"),
            ("metal2 island to next column rail",
             self.pitch - self.w_col - self.gap - self.pad, r("m2.2"), "m2.2"),
            ("metal2 island to island in y", self.pitch - self.pad, r("m2.2"), "m2.2"),
        ):
            if got < want - 1e-9:
                raise ValueError(f"{what} is {got:.4g} um, rule {rule} wants "
                                 f">= {want:.4g} um")

    @property
    def span(self) -> float:
        return self.n * self.pitch

    def tap_center(self, i: int, j: int) -> tuple:
        """Centre of cell (i, j)'s via: on the row rail's centreline, clear of the col rail."""
        return (j * self.pitch + self.w_col + self.gap + self.pad / 2,
                i * self.pitch + self.w_row / 2)

    def __str__(self):
        return (f"{self.n}x{self.n} @ {self.pitch:g} um pitch, "
                f"met1 rows {self.w_row:g} um / met2 cols {self.w_col:g} um")


def array_gds(path: Path, sk: Skeleton, layers: O.Layers = None,
              cellname: str = "XBAR_SKELETON") -> Path:
    """Draw the skeleton.  `Skeleton.check` runs first so failures name a rule."""
    sk.check()
    layers = layers or O.Layers()
    ly, top = O.new_layout(cellname)
    m1 = ly.layer(*layers[sk.row_layer])
    m2 = ly.layer(*layers[sk.col_layer])
    cut = ly.layer(*layers[sk.cut_layer])

    for i in range(sk.n):
        y = i * sk.pitch
        top.shapes(m1).insert(O.rect(0.0, y, sk.span, y + sk.w_row))
    for j in range(sk.n):
        x = j * sk.pitch
        top.shapes(m2).insert(O.rect(x, 0.0, x + sk.w_col, sk.span))
    h, p = sk.via / 2, sk.pad / 2
    for i in range(sk.n):
        for j in range(sk.n):
            cx, cy = sk.tap_center(i, j)
            top.shapes(cut).insert(O.rect(cx - h, cy - h, cx + h, cy + h))
            top.shapes(m2).insert(O.rect(cx - p, cy - p, cx + p, cy + p))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


# --------------------------------------------------------------------------
# Per-pitch R and C, by finite difference on rail length.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PerPitch:
    """What one more cell pitch of rail costs: exactly what `r_row` / `c_row` mean."""
    layer: str
    width: float
    pitch: float
    r: float           # ohm
    c: float           # F, to substrate
    c_area: float
    c_fringe: float
    squares: float

    def __str__(self):
        return (f"{self.layer} {self.width:g} um wide, {self.pitch:g} um pitch: "
                f"{self.squares:.3f} sq -> r = {self.r:.4g} ohm, "
                f"c = {self.c / 1e-15:.4g} fF ({self.c_area / 1e-15:.4g} area + "
                f"{self.c_fringe / 1e-15:.4g} fringe)")


def per_pitch(tech: O.TechConstants, layer: str, width: float, pitch: float,
              rundir: Path, n: int = 16) -> PerPitch:
    """Measure two rails one pitch apart in length and difference them.

    The difference is the point.  A single rail's perimeter carries two endcaps worth
    `2 * width`, which at a 1 um pitch is a 30% error on the per-pitch fringe term if you
    just divide by N.  Differencing cancels them exactly and leaves the marginal cost of
    one more cell, which is the quantity `crossbar.Crossbar` stamps one of per pitch.

    Both measurements go through GDS write, read back and `layout_oracle.measure`, i.e. the
    path the Stage-1 hand check pins.
    """
    rundir = Path(rundir)
    out = []
    for k, m in enumerate((n, n + 1)):
        gds = O.wire_gds(rundir / f"rail_{layer}_{width:g}_{pitch:g}_{m}.gds",
                         width, m * pitch, layer=layer)
        out.append(O.parasitics(O.measure(gds, layer), tech))
    a, b = out
    return PerPitch(layer=layer, width=width, pitch=pitch, r=b.r - a.r, c=b.c - a.c,
                    c_area=b.c_area - a.c_area, c_fringe=b.c_fringe - a.c_fringe,
                    squares=b.squares - a.squares)


def derived_params(tech: O.TechConstants, sk: Skeleton, rundir: Path) -> tuple:
    """(row PerPitch, col PerPitch, CrossbarParams) for one skeleton.

    The returned `CrossbarParams` carries a placeholder weight matrix, exactly as
    `fp_matmul.IDEAL_XBAR` does: `fp_matmul.matmul` overwrites `.weights` per block.
    """
    row = per_pitch(tech, sk.row_layer, sk.w_row, sk.pitch, rundir, n=sk.n)
    col = per_pitch(tech, sk.col_layer, sk.w_col, sk.pitch, rundir, n=sk.n)
    xbar = replace(F.IDEAL_XBAR, r_row=row.r, r_col=col.r, c_row=row.c, c_col=col.c)
    return row, col, xbar


# --------------------------------------------------------------------------
# The two terms the circuit model does not have.
# --------------------------------------------------------------------------
def tap_resistance(tech: O.TechConstants, sk: Skeleton, row: PerPitch) -> dict:
    """Series resistance at each cell tap, which `crossbar.Crossbar` does not model.

    Two routes, both priced from the tech file's `contact` lines: the 0T1R skeleton drawn
    here (one metal1->metal2 via) and a 1T1R cell, which must also reach down through
    metal1->li1 (`mcon`) and li1->diffusion (`nsc`) to the access transistor.

    The comparison that decides whether it matters is *not* against the rail R per pitch --
    it is against the cell impedance, because a tap resistance sits in series with one cell
    and is shared with nothing.  docs/DESIGN.md measured that ~93% of the droop at
    5 ohm/pitch is shared-segment loading; a per-cell series R contributes none of that
    mechanism.  Its whole effect is the weight-dependent compression `crossbar.ideal_1t1r`
    already models, with `ron` raised by this amount -- so that is what gets reported: the
    g_max compression each route produces, against the 7.569% the access FET alone causes.

    The two routes land on opposite sides of "negligible", which is why both are priced.
    """
    r_cell_max = 1.0 / F.IDEAL_XBAR.g_min      # 1 Mohm at g_min
    r_cell_min = 1.0 / F.IDEAL_XBAR.g_max      # 10 kohm at g_max
    routes = {
        # 0T1R: the row rail taps up to the cell through the one via that is drawn.
        "0T1R as drawn, 1 via": (tech.r_contact("via"), 0.0),
        # 1T1R: the cell must also reach the access FET's drain, m1 -> li1 -> diffusion.
        "1T1R, + mcon + li1-to-diff": (tech.r_contact("via") + tech.r_contact("mcon")
                                       + tech.contact_res["nsc"].value / 1e3,
                                       RON_IN_ARRAY),
    }
    out = {}
    for label, (r_tap, ron) in routes.items():
        # compression at g_max = 1 - G_eff/G, G_eff = 1/(1/G + ron + r_tap)
        comp = 100 * (1 - r_cell_min / (r_cell_min + ron + r_tap))
        base = 100 * (1 - r_cell_min / (r_cell_min + ron))
        out[label] = {"r_tap": r_tap, "ron": ron, "compression": comp,
                      "compression_without_tap": base, "delta": comp - base,
                      "x_rail": r_tap / row.r}
    return {"routes": out, "r_rail_per_pitch": row.r,
            "r_cell_min": r_cell_min, "r_cell_max": r_cell_max}


def coupling_report(tech: O.TechConstants, sk: Skeleton, row: PerPitch,
                    col: PerPitch) -> dict:
    """Coupling C per cell pitch, against the shunt-to-substrate C the model does have.

    Three terms, all from the tech file, none of them in `crossbar.Crossbar`:

      row-row     two adjacent metal1 rails, `defaultsidewall allm1`, over `pitch` of
                  facing edge on each side -- so 2x for an interior rail.
      col-col     the same on metal2.
      row-col     `defaultoverlap allm2 metal2 allm1 metal1` over the `w_row * w_col`
                  crossing, plus `defaultsideoverlap` for the column rail's two edges
                  running across the row rail, `2 * w_row` of edge per crossing.  This one
                  is *input line to output line*: in a crossbar it is a direct feedthrough
                  from a driven row to a sensed column, which is a different kind of error
                  from a shunt load.  The edge term is the larger of the two and its
                  `2 * w_row` edge length is this module's accounting, not magic's.

    Two honest caveats on the sidewall numbers, both stated in docs/LAYOUT.md:

      * The tech file gives one sidewall constant per layer and no reference spacing.
        magic scales coupling with separation inside a `sidehalo` (sky130A.tech:5027) and
        this does not.  Taking the constant at the drawn spacing is a model choice; it is
        the conservative one, and it gets less conservative as the pitch opens up.
      * Adding the full perimeter fringe term *and* the full sidewall term double-counts
        the same physical edge.  A close neighbour shields the fringe field that would
        otherwise reach the substrate -- magic has `fringeshieldhalo` for exactly this and
        this does not.  So `c_total` here is an upper bound, and the *ratio* is the robust
        part of the finding.
    """
    s_row = tech.c_side(sk.row_layer) * sk.pitch
    s_col = tech.c_side(sk.col_layer) * sk.pitch
    plate = tech.c_overlap(sk.col_layer, sk.row_layer) * sk.w_row * sk.w_col
    fringe = tech.c_sideoverlap(sk.col_layer, sk.row_layer) * 2 * sk.w_row
    return {
        "row_gnd": row.c, "col_gnd": col.c,
        "row_row": 2 * s_row, "col_col": 2 * s_col,
        "row_col_plate": plate, "row_col_fringe": fringe, "row_col": plate + fringe,
        "row_ratio": 2 * s_row / row.c, "col_ratio": 2 * s_col / col.c,
        "cross_ratio": (plate + fringe) / row.c,
        "spacing_row": sk.pitch - sk.w_row, "spacing_col": sk.pitch - sk.w_col,
    }


# --------------------------------------------------------------------------
# The pitch / resistance loop: g_min sets the cell area, the cell area sets the pitch,
# the pitch sets the rail R.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Serpentine:
    r_target: float
    r_sheet: float
    squares: float
    width: float
    space: float
    length: float
    side: float
    stripes: float

    def __str__(self):
        return (f"{self.r_target / 1e3:.0f} kohm from {self.r_sheet:.1f} ohm/sq needs "
                f"{self.squares:.0f} squares = {self.length:.0f} um of {self.width:g} um "
                f"line; serpentined at {self.width + self.space:g} um stripe pitch that is "
                f"a {self.side:.1f} x {self.side:.1f} um block ({self.stripes:.0f} stripes)")


def gmin_serpentine(tech: O.TechConstants, g_min: float = None,
                    device: str = "res_xhigh_po") -> Serpentine:
    """How big a square block of `xhrpoly` it takes to reach `1 / g_min`.

    This is the constraint that actually sets the cell pitch, and it is brutal.  `g_min`
    defaults to `crossbar.CrossbarParams`' 1 uS, i.e. 1 Mohm, and sky130's highest-sheet
    poly resistor is 319.8 ohm/square -- so ~3100 squares of it.

    Width is the narrowest *binned* device the PDK ships a model for, not the DRC minimum:
    `poly.3` allows 0.33 um but there is no 0.33 um model, and a resistor you cannot
    simulate is not a design.  Stripe spacing is `poly.9`, poly-resistor to poly.
    """
    r_target = 1.0 / (g_min or F.IDEAL_XBAR.g_min)
    r_sheet = tech.sheet_res["xhrpoly"].value / 1e3
    squares = r_target / r_sheet
    width = O.pdk_resistor_widths(device)[0]
    space = O.drc_rule_value("poly.9")
    length = squares * width
    # A square block of side S serpentined at stripe pitch (w + s) holds S/(w+s) stripes of
    # length S, so S = sqrt(length * (w + s)).
    side = float(np.sqrt(length * (width + space)))
    return Serpentine(r_target=r_target, r_sheet=r_sheet, squares=squares, width=width,
                      space=space, length=length, side=side,
                      stripes=side / (width + space))


def settling_estimate(sk: Skeleton, pp: PerPitch) -> dict:
    """`R_tot * C_tot / 2 = r * c * N^2 / 2`, the Elmore estimate docs/REPORT.md 6 uses.

    Not a new model: REPORT.md measured 63% settling of the summed column current against
    exactly this expression on a 32x32 array and found it low by a consistent 1.32-1.34x,
    the usual gap between a one-pole Elmore delay and a distributed line.  So the corrected
    figure below applies that measured 1.3x rather than pretending the estimate is tight.

    This is the only place the derived C can show up at all: `crossbar.mac_sim` runs a `.op`
    and a `.op` has no capacitors in it.
    """
    tau = pp.r * pp.c * sk.n ** 2 / 2
    return {"tau_elmore": tau, "tau_63": 1.3 * tau}


def routing_layer_penalty(tech: O.TechConstants, pitch: float,
                          layers=("met1", "met2", "li1")) -> dict:
    """R per pitch on each candidate rail layer, at that layer's own minimum width.

    The point of the comparison is local interconnect: at 12.8 ohm/square against metal1's
    0.125 it is 102x worse per square, and it is also narrower, so the per-pitch figure is
    worse still.  docs/MATMUL.md measures 74.2% int8 match at 5 ohm/pitch even *with*
    per-channel calibration, so anything in the tens of ohms is not a design.
    """
    rules = {"met1": "m1.1", "met2": "m2.1", "li1": "li.1"}
    out = {}
    for lay in layers:
        w = O.drc_rule_value(rules[lay])
        rs = tech.r_sheet(lay)
        out[lay] = {"w_min": w, "r_sheet": rs, "r_per_pitch": rs * pitch / w}
    return out


# --------------------------------------------------------------------------
# The payoff: the published accuracy study, re-run on derived parasitics.
# --------------------------------------------------------------------------
def operands(shape=SHAPE, seed=SEED):
    """i.i.d. standard-normal A and B, the same pair scripts/int8_accuracy.py uses."""
    M, N, K = shape
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N)), rng.standard_normal((N, K))


def int8_row(A_q, B_q, xbar, tag: str, rundir: str) -> dict:
    """One row of the docs/MATMUL.md 3.2 table: raw and calibrated, both granularities."""
    row = {"tag": tag}
    t0 = time.perf_counter()
    for per_channel in (False, True):
        Aq = A_q[per_channel]
        res = Q.run_q(Aq, B_q, xbar=xbar, rows=TILE[0], cols=TILE[1],
                      rundir=f"{rundir}_{per_channel:d}")
        raw = Q.metrics(res)
        cal = Q.metrics(Q.calibrate(
            res, Q.fit_gain(res.c_analog, res.c_ref, per_channel=per_channel)))
        key = "pc" if per_channel else "pt"
        row[f"{key}_raw"] = 100 * raw.match
        row[f"{key}_cal"] = 100 * cal.match
        row[f"{key}_raw_lsb"] = raw.max_lsb
        row[f"{key}_cal_lsb"] = cal.max_lsb
        row[f"{key}_bits"] = cal.bits_rms
        row["bits_needed"] = raw.bits_needed
    row["seconds"] = time.perf_counter() - t0
    return row


def accuracy_rerun(derived: list, rundir: str = "/tmp/xbar_layout_acc") -> list:
    """The assumed `r_wire` sweep and the derived-parasitic points, one table.

    `A` is quantized once per granularity and reused across every parasitic setting, so
    every row compares against the same integer reference -- the reason
    `int8_matmul.run_q` is split out from `matmul` in the first place.
    """
    A, B = operands()
    A_q = {False: Q.quantize(A)[0], True: Q.quantize(A, axis=1)[0]}
    B_q, _ = Q.quantize(B)

    rows = []
    for r in ASSUMED_R:
        rows.append(int8_row(A_q, B_q, replace(F.IDEAL_XBAR, r_row=r, r_col=r),
                             f"assumed r_wire = {r:g}", f"{rundir}/assumed_{r:g}"))
    for name, sk, xbar in derived:
        rows.append(int8_row(A_q, B_q, xbar,
                             f"derived {name} (r {xbar.r_row:.3g}/{xbar.r_col:.3g})",
                             f"{rundir}/derived_{name}"))
    return rows


def fp_rerun(derived: list, rundir: str = "/tmp/xbar_layout_fp") -> list:
    """`fp_matmul` effective bits on the same operands, assumed vs derived."""
    A, B = operands()
    ref = A @ B
    out = []
    for tag, xbar, d in ([(f"assumed r_wire = {r:g}", replace(F.IDEAL_XBAR, r_row=r,
                                                              r_col=r), None)
                          for r in ASSUMED_R]
                         + [(f"derived {n}", x, n) for n, _, x in derived]):
        t0 = time.perf_counter()
        got = F.matmul(A, B, xbar=xbar, rows=TILE[0], cols=TILE[1],
                       rundir=f"{rundir}/{tag.replace(' ', '_').replace('=', '')}")
        mx, rms = F.rel_errors(got, ref)
        out.append({"tag": tag, "max": mx, "rms": rms, "bits_rms": F.effective_bits(rms),
                    "seconds": time.perf_counter() - t0})
    return out


# --------------------------------------------------------------------------
SKELETONS = {
    # The densest this skeleton supports: pitch set by the metal2 island clearing the
    # column rail on both sides.  Interconnect-limited, and not reachable in practice
    # because no memory device fits in 0.74 um -- see `gmin_serpentine`.
    "dense": Skeleton(n=16, pitch=0.74, w_row=0.32, w_col=0.14),
    # A pitch with room for a cell, on rails wide enough to be worth drawing wide.
    "relaxed": Skeleton(n=16, pitch=2.0, w_row=0.5, w_col=0.5),
}


def main(accuracy: bool = False, rundir: Path = Path("/tmp/xbar_layout")):
    t_start = time.perf_counter()
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    tech = O.TechConstants.parse()
    layers = O.Layers()

    print("=" * 78)
    print("STAGE 2  crossbar interconnect skeleton -> geometry-derived parasitics")
    print("=" * 78)
    print(f"tech  {tech.path}  ({O.CORNER_NOTE.split(':')[0]} corner)")
    print(f"rails {SKELETONS['dense'].row_layer} rows / "
          f"{SKELETONS['dense'].col_layer} columns, "
          f"{SKELETONS['dense'].cut_layer} tap per cell")

    # The g_min serpentine first: it is what makes the third skeleton's pitch what it is.
    print("\n" + "-" * 78)
    print("the pitch / resistance loop: g_min sets the cell size")
    print("-" * 78)
    serp = gmin_serpentine(tech)
    print(f"  {serp}")
    print(f"  provenance: {O.drc_rule_value('poly.9')} um stripe space is rule poly.9; "
          f"{serp.width:g} um width is the narrowest binned res_xhigh_po model")
    print(tech.provenance("xhrpoly"))
    pitch_gmin = float(np.ceil(serp.side))
    print(f"  -> a cell that holds one g_min resistor is ~{serp.side:.1f} um on a side, so "
          f"the pitch cannot be below that")

    skeletons = dict(SKELETONS)
    skeletons["g_min-driven"] = Skeleton(n=16, pitch=pitch_gmin, w_row=1.0, w_col=1.0)

    print("\n" + "-" * 78)
    print("skeletons: draw, DRC, derive")
    print("-" * 78)
    derived, measured = [], {}
    for name, sk in skeletons.items():
        gds = array_gds(rundir / f"skeleton_{name}.gds", sk, layers)
        drc = O.run_drc(gds, rundir)
        row, col, xbar = derived_params(tech, sk, rundir)
        m1 = O.measure(gds, sk.row_layer, layers)
        m2 = O.measure(gds, sk.col_layer, layers, require_boxes=False)
        print(f"\n  {name}: {sk}")
        print(f"    {sk.span:g} x {sk.span:g} um, {m1.n_polys} met1 + {m2.n_polys} met2 "
              f"polygons + {sk.n ** 2} vias; met1 {m1.area:.2f} um^2, "
              f"met2 {m2.area:.2f} um^2")
        print(f"    {drc}")
        print(f"    row {row}")
        print(f"    col {col}")
        print(f"    -> r_row = {row.r:.4g}, r_col = {col.r:.4g} ohm/pitch;  "
              f"c_row = {row.c:.4g}, c_col = {col.c:.4g} F/pitch")
        s = settling_estimate(sk, row)
        print(f"    row-line settling r*c*N^2/2 = {s['tau_elmore']:.3g} s, "
              f"x1.3 measured correction -> tau_63 ~ {s['tau_63']:.3g} s")
        if not drc.clean:
            raise AssertionError(f"{name} skeleton is not DRC clean: {drc.by_rule}")
        derived.append((name, sk, xbar))
        measured[name] = (row, col)

    print("\n" + "-" * 78)
    print("uniformity, and the two terms the circuit model does not have")
    print("-" * 78)
    name, sk, xbar = derived[0]
    row, col = measured[name]
    print(f"  (all figures for the '{name}' skeleton)")
    print("\n  rail R per pitch is uniform *by construction*: a uniform-width rail has "
          "pitch/w\n  squares in every cell, so there is nothing to discover there. The "
          "deviations are:")

    tap = tap_resistance(tech, sk, row)
    print(f"\n  1. via / contact resistance at the cell tap -- not in crossbar.Crossbar")
    print(tech.provenance("m2c", "mcon", "nsc"))
    print(f"     cell impedance for scale: {tap['r_cell_min'] / 1e3:.0f} kohm at g_max, "
          f"{tap['r_cell_max'] / 1e3:.0f} kohm at g_min")
    for label, d in tap["routes"].items():
        print(f"     {label:28s} {d['r_tap']:7.2f} ohm ({d['x_rail']:5.1f}x the rail R per "
              f"pitch) -> g_max compression {d['compression_without_tap']:.3f}% -> "
              f"{d['compression']:.3f}%  ({d['delta']:+.3f} pt)")
    print("     VERDICT: it is not a wire-IR-drop-like effect at all. A tap R is in series "
          "with\n     one cell and shared with nothing, so it produces none of the "
          "shared-segment loading\n     DESIGN.md measured as ~93% of the droop; its entire "
          "effect is the weight-dependent\n     compression `crossbar.ideal_1t1r` already "
          "models, with `ron` raised.")
    print(f"     For the 0T1R skeleton as drawn that is "
          f"{tap['routes']['0T1R as drawn, 1 via']['delta']:.3f} points of compression at "
          f"g_max: genuinely\n     negligible. For 1T1R it is "
          f"{tap['routes']['1T1R, + mcon + li1-to-diff']['delta']:.2f} points on top of the "
          f"FET's own {COMPRESSION_G_MAX}%, a ~22%\n     increase in the effective Ron -- "
          "small but *not* negligible, and it is missing from\n     both "
          "`crossbar.Crossbar` and the `ideal_1t1r` reference as they stand.")

    cp = coupling_report(tech, sk, row, col)
    print(f"\n  2. rail-to-rail coupling C -- crossbar.Crossbar has shunt C to vss only")
    print(tech.provenance("allm1", "allm2", ("allm2", "allm1")))
    print(f"     row shunt to substrate      {cp['row_gnd'] / 1e-18:8.1f} aF/pitch")
    print(f"     row to the two neighbours   {cp['row_row'] / 1e-18:8.1f} aF/pitch  "
          f"({cp['row_ratio']:.2f}x the shunt, at {cp['spacing_row']:g} um spacing)")
    print(f"     col shunt to substrate      {cp['col_gnd'] / 1e-18:8.1f} aF/pitch")
    print(f"     col to the two neighbours   {cp['col_col'] / 1e-18:8.1f} aF/pitch  "
          f"({cp['col_ratio']:.2f}x the shunt, at {cp['spacing_col']:g} um spacing)")
    print(f"     row-to-column crossing      {cp['row_col'] / 1e-18:8.1f} aF/cell   "
          f"({cp['row_col_plate'] / 1e-18:.1f} plate + "
          f"{cp['row_col_fringe'] / 1e-18:.1f} edge fringe, "
          f"{cp['cross_ratio']:.2f}x the row shunt)")
    print("     VERDICT: the C model is structurally incomplete, not just imprecise. "
          "Coupling to\n     the neighbouring rails is the same order as the modelled "
          "shunt to substrate, and\n     the row-to-column term has no counterpart in the "
          "model at all -- it is a direct\n     input-to-output feedthrough, which is a "
          "different error from a shunt load. None\n     of this touches the DC .op MAC "
          "accuracy, which is what the study measures; it\n     changes settling and it "
          "would put crosstalk on a transient read.")

    print("\n" + "-" * 78)
    print(f"routing layer choice, at the '{name}' pitch of {sk.pitch:g} um")
    print("-" * 78)
    pen = routing_layer_penalty(tech, sk.pitch)
    base = pen["met1"]["r_per_pitch"]
    for lay, d in pen.items():
        print(f"  {lay:5s} {d['r_sheet']:8.3f} ohm/sq at min width {d['w_min']:g} um "
              f"-> {d['r_per_pitch']:8.3f} ohm/pitch  ({d['r_per_pitch'] / base:6.1f}x met1)")
    print(f"  local interconnect is {pen['li1']['r_sheet'] / pen['met1']['r_sheet']:.1f}x "
          "metal1 per square and narrower on top of that. docs/MATMUL.md measures\n  74.2% "
          "int8 match at 5 ohm/pitch with per-channel calibration, so a li1 rail is not a "
          "design.")

    if accuracy:
        print("\n" + "-" * 78)
        print("the payoff: docs/MATMUL.md 3.2, re-run on derived parasitics")
        print(f"A {SHAPE[0]}x{SHAPE[1]} @ B {SHAPE[1]}x{SHAPE[2]}, {TILE[0]}x{TILE[1]} "
              "tile, 0T1R, unipolar two-pass, batched")
        print("-" * 78)
        rows = accuracy_rerun(derived)
        hdr = ("configuration", "pt raw", "pt+gain", "pc raw", "pc+gain", "LSB raw",
               "LSB cal", "bits(rms)")
        print(f"  {hdr[0]:<38s}{hdr[1]:>8s}{hdr[2]:>9s}{hdr[3]:>8s}{hdr[4]:>9s}"
              f"{hdr[5]:>9s}{hdr[6]:>9s}{hdr[7]:>11s}")
        for r in rows:
            print(f"  {r['tag']:<38s}{r['pt_raw']:>7.2f}%{r['pt_cal']:>8.2f}%"
                  f"{r['pc_raw']:>7.2f}%{r['pc_cal']:>8.2f}%"
                  f"{r['pc_raw_lsb']:>9d}{r['pc_cal_lsb']:>9d}{r['pc_bits']:>11.2f}")
        print(f"  (exact int32 for this pair would need {rows[0]['bits_needed']:.1f} bits; "
              f"{sum(r['seconds'] for r in rows):.1f} s in ngspice)")

        print("\n  float64 path, same operands:")
        for r in fp_rerun(derived):
            print(f"    {r['tag']:<24s} rms rel {r['rms']:.3e}  max rel {r['max']:.3e}  "
                  f"{r['bits_rms']:5.2f} bits  {r['seconds']:.2f} s")
        print("\n  c_row / c_col are set on the derived rows but a `.op` solve ignores "
              "capacitors,\n  so they cannot move these numbers. What they change is "
              "settling; see crossbar.settle_sim.")

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s, artifacts in {rundir}")


if __name__ == "__main__":
    main(accuracy="--accuracy" in sys.argv)
