"""Device-level 0T1R crossbar cell: one real sky130 poly resistor per cell, drawn.

`array_layout.py` draws the interconnect and leaves an empty gap where the memory device
would sit, because sky130 has no resistive-memory element.  This module fills the gap the
other way round: it gives up programmability and draws a **mask-programmed resistor** --
`sky130_fd_pr__res_xhigh_po_0p35`, a real PDK device with a real model and a real
LVS-recognizable marker stack -- so that the cell pitch stops being a parameter and becomes
an output of the geometry.

Three things follow from that, and they are the reason this module exists:

  * **The pitch is computed from the device.**  `plan_cell` takes a target resistance,
    picks the stripe count that makes the resistor block squarest, and reports the cell
    bounding box.  `r_row` / `r_col` then come out of `array_layout.per_pitch` at that
    pitch, so the wire parasitics are downstream of the cell, not assumed alongside it.
  * **DRC runs with FEOL enabled.**  A cell with a poly resistor has front-end layers, so
    the stock deck's `FEOL = false` would make "DRC clean" close to meaningless.
    `feol_deck` writes a patched copy of the deck with `FEOL = true` -- 381 rule categories
    instead of 145 -- and `bad_cell_gds` plants a poly-only violation to prove the harness
    still fails when it should.
  * **LVS compares the length.**  For a fixed-width poly resistor the deck uses
    `BResistorFixedWidth`, which disables `R` and `W` and enables `L` with a 0.5% relative
    tolerance (`sky130.lvs:465`, `:2254`).  `L` is the parameter that carries the
    resistance, so unlike the MOS `W`/`L`-only match in docs/LAYOUT.md 5 this one is not
    hollow -- but it is the chain's *total* length, because the extractor combines series
    devices.  `lvs_probes` measures all of that by breaking one thing at a time instead of
    trusting the deck source, and `LVS_COMPARED` states the conclusion.

What is given up: **the weights freeze at tapeout.**  A mask-programmed resistor array is
an inference demonstrator, not a programmable accelerator -- there is no write operation,
so there is nothing to program.  See `DEVICE_NOTE`.

Run it:  uv run cell_layout.py               (cell + DRC + LVS + array + g_min sweep)
         uv run cell_layout.py --accuracy    (adds the int8 re-run; needs ngspice)
"""
import re
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import klayout.db as kdb
import numpy as np

import array_layout as A
import crossbar as C
import fp_matmul as F
import layout_oracle as O

# --------------------------------------------------------------------------
# The device decision.
# --------------------------------------------------------------------------
DEVICE_NOTE = (
    "0T1R with a mask-programmed poly resistor: no access FET.  The access device in an "
    "RRAM array isolates a cell for *programming* and blocks sneak paths during partial "
    "access.  A mask-programmed array does neither -- there is no write, and in the MAC "
    "every row is driven by a source and every column is held at virtual ground, so no "
    "node floats and there is no sneak path.  The cost is that the weights are fixed at "
    "tapeout: this is an inference-only demonstrator, not a programmable accelerator."
)

# `res_xhigh_po`, not `res_high_po`.  Both families ship the same five binned widths, but
# they sit on different marker layers with a 6.25x difference in sheet resistance, and the
# magic tech file names them confusingly: its `xhrpoly` type-set (319.8 ohm/square) is
# `res_high_po` (`sky130A.tech:6036`), and `uhrpoly` (2000 ohm/square, :5117) is
# `res_xhigh_po` (:6136).  docs/LAYOUT.md 8 pairs 319.8 ohm/square with the *name*
# `res_xhigh_po`, which is the two halves of two different devices.
#
# Two reasons to take `uhrpoly`: 6.25x fewer squares for the same resistance, and its
# constant is self-consistent.  `res_xhigh_po__base.model.spice` has `rsheet = 2000.0` with
# `rbody = l*rsheet/w`, exactly matching the tech file.  `res_high_po`'s own 0.35 um model
# has `rsheet = 1112.41` ohm per um of length, i.e. 389.3 ohm/square effective -- 22% above
# the tech file's 319.8, which is the wide-line asymptote its 5.73 um bin converges to.
DEVICE_FAMILY = "res_xhigh_po"
DEVICE_TYPESET = "uhrpoly"        # the magic tech-file type-set carrying its sheet R
DEVICE_MARKER = "urpm"            # the marker layer the LVS deck keys the 2 kohm bin on

# `sky130_hdl21.ress` keys for the binned `res_xhigh_po` devices, by drawn width in um.
HDL21_RES = {0.35: "PM_PREC_0p35", 0.69: "PM_PREC_0p69", 1.41: "PM_PREC_1p41",
             2.85: "PM_PREC_2p85", 5.73: "PM_PREC_5p73"}

# `sky130_fd_pr__res_xhigh_po__base.model.spice` states its end resistance as a polynomial in
# the drawn width.  Matching the exact form rather than evaluating the text keeps this a parse
# instead of an `eval`, and makes a model revision fail loudly instead of quietly.
RCON_FORM = re.compile(r"^([-\d.eE+]+)/\(w\*w\)\+([-\d.eE+]+)/w\+([-\d.eE+]+)$")

# Front-end and marker layers this module draws.  Keys are the *DRC deck's* own variable
# names so `_check_layers` can compare them against the deck's `polygons(l, d)` calls --
# the map file is a LEF/DEF routing map and has no rows for any of these.
CELL_LAYERS = {"poly": (66, 20), "poly_rs": (66, 13), "urpm": (79, 20),
               "psdm": (94, 20), "npc": (95, 20), "licon": (66, 44)}

# met2 text, for the column terminal's net name.  `layout_oracle.LABEL_LAYERS` has met1 but
# not met2; `connect(met2_con, met2_label)` is sky130.lvs:1852.
MET2_LABEL = (69, 5)

# The manufacturing grid every dimension is snapped to.  `sky130A.lydrc` checks it as
# `poly.ongrid(0.005)`; `_check_grid` greps that back so the number is not just typed here.
GRID = 0.005

# One grid step of slack on every enclosure and clearance.  Drawing exactly at a rule limit
# is legal, but grid snapping a computed dimension can land it one database unit under, and
# the failure looks like a mysterious enclosure violation rather than a rounding bug.
SLACK = 0.010

# Rail widths, held fixed across every array this module draws so that the comparison
# against docs/LAYOUT.md 9's `g_min-driven` row isolates one variable: the pitch.  That row
# used 1.0 um rails at an assumed 31 um pitch.
W_ROW = 1.0
W_COL = 1.0

# The `g_min` sweep.  Sampled finely between 10 and 50 uS because that is where the pitch
# stops being resistor-limited and the int8 fidelity turns over.
GMIN_SWEEP = (1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 30e-6, 40e-6, 50e-6)

# Every DRC dimension the cell generator uses, by rule name.  Nothing below is typed in.
CELL_RULES = ("poly.3", "poly.9", "licon.1", "licon.2", "licon.8", "licon.15",
              "li.1", "li.3", "li.5", "li.6", "ct.1", "ct.2", "m1.1", "m1.2", "m1.4",
              "m1.5", "m1.6", "via.1a", "via1.5a", "m2.1", "m2.2", "m2.5", "m2.6",
              "npc.1", "npc.2", "n/psdm.1", "rpm.2", "rpm.3")

CELL_NAME = "xbar_cell"     # must equal the LVS reference netlist's subckt name
ARRAY_NAME = "xbar_array"
ROW_NET, COL_NET = "row", "col"
LVS_SUB = O.LVS_SUBSTRATE   # "b", the deck's global substrate net, via -rd lvs_sub=

# The deck needs this to compare a series chain at all -- see `LVS_EXTRA` and `lvs_probes`.
LVS_EXTRA = {"schematic_simplify": "true"}

# What the LVS deck says it compares for this device.  `lvs_probes` measures it instead of
# taking the source's word for it, because docs/LAYOUT.md 5 records a match that was much
# weaker than the phrase "LVS MATCH" implies and the same trap is available here.
LVS_COMPARED = (
    "sky130.lvs:2254 extracts it as `BResistorFixedWidth`, defined at :465 with "
    "enable_parameter('R', false), ('W', false), ('L', true), and :2255 puts a 0.5% "
    "relative tolerance on L.  So the deck's claim is: length compared, resistance and "
    "width not.  What the probes below add is that the extractor **combines the series "
    "chain into one device before comparing**, so the quantity checked is the *total* "
    "resistor length -- 16 x 10.94 um and 8 x 21.88 um would compare equal -- and that the "
    "device bin is checked through the extracted device class name."
)


# --------------------------------------------------------------------------
# The device, and the rules.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ResDevice:
    """One binned sky130 poly precision resistor."""
    name: str          # sky130_fd_pr__res_xhigh_po_0p35
    hdl21_key: str     # sky130_hdl21.ress key
    width: float       # um, the fixed drawn width -- length is the only free parameter
    r_sheet: float     # ohm/square, from the magic tech file, == the model's `rsheet`
    r_end: float       # ohm, the model's own end resistance for both contact heads
    marker: str        # the marker layer the LVS deck keys on

    def __str__(self):
        return (f"{self.name} ({self.width:g} um wide, {self.r_sheet:g} ohm/square, "
                f"{self.r_end:.1f} ohm end resistance)")


def base_model(family: str = DEVICE_FAMILY) -> tuple:
    """`(rsheet, rcon(w))` parsed out of the device family's own SPICE model.

    The tech file and the model are two independent statements of the same physics, which is
    what makes them checkable against each other -- so `rsheet` here is compared against the
    tech file's `uhrpoly` rather than either being trusted alone.

    `rcon` is the model's accounting of the two contact heads, and it is **not** the same
    number magic gives: magic says `contact pc,xpc 152000` per cut, i.e. 304 ohm for a
    two-ended stripe, while the model's polynomial gives 587.8 ohm at 0.35 um -- 1.9x more.
    Two PDK statements of one physical thing.  This module uses the model's, because the
    model is what would actually simulate, and reports magic's alongside it.
    """
    path = (C.sky130_root() / "libs.ref/sky130_fd_pr/spice"
            / f"sky130_fd_pr__{family}__base.model.spice")
    text = path.read_text()
    rsheet = re.search(r"rsheet\s*=\s*([\d.eE+-]+)", text)
    rcon = re.search(r"rcon\s*=\s*\{([^}]*)\}", text)
    if not rsheet or not rcon:
        raise RuntimeError(f"{path}: no `rsheet` and `rcon` to parse")
    m = RCON_FORM.match(rcon.group(1).replace(" ", ""))
    if not m:
        raise RuntimeError(f"{path}: `rcon` is {rcon.group(1)!r}, which is not the "
                           f"A/(w*w)+B/w+C form {RCON_FORM.pattern!r} expects")
    a, b, c = (float(g) for g in m.groups())
    return float(rsheet.group(1)), lambda w: a / (w * w) + b / w + c


def res_devices(tech: O.TechConstants) -> list:
    """The binned `res_xhigh_po` devices the PDK ships a model for, narrowest first.

    Widths come off the model filenames via `layout_oracle.pdk_resistor_widths`, the sheet
    resistance off the tech file, the end resistance off the model.  Two cross-checks: every
    width must have a `sky130_hdl21` module behind it (otherwise there is no netlist for LVS),
    and the tech file's sheet resistance must equal the model's.
    """
    rs = tech.sheet_res[DEVICE_TYPESET].value / 1e3
    r_model, rcon = base_model()
    if rs != r_model:
        raise ValueError(
            f"{tech.path}: sheet resistance for {DEVICE_TYPESET!r} is {rs} ohm/square but "
            f"sky130_fd_pr__{DEVICE_FAMILY}__base.model.spice says {r_model}. One of the two "
            f"is not the device this module thinks it is. {O.CORNER_NOTE}")
    out = []
    for w in O.pdk_resistor_widths(DEVICE_FAMILY):
        key = HDL21_RES.get(w)
        if key is None:
            raise ValueError(f"the PDK ships a {w} um {DEVICE_FAMILY} model but HDL21_RES "
                             "has no sky130_hdl21 module for it, so it has no LVS netlist")
        out.append(ResDevice(name=f"sky130_fd_pr__{DEVICE_FAMILY}_"
                                  f"{str(w).replace('.', 'p')}",
                             hdl21_key=key, width=w, r_sheet=rs, r_end=rcon(w),
                             marker=DEVICE_MARKER))
    return out


def cell_rules(deck: Path = None) -> dict:
    """{rule name: dimension in um}, parsed off the deck line that emits each rule."""
    return {r: O.drc_rule_value(r, deck) for r in CELL_RULES}


def _check_layers(deck: Path = None) -> dict:
    """Cross-check `CELL_LAYERS` against the DRC deck's own layer definitions."""
    found = O._feol_layers_from_drc(deck)
    bad = {k: (v, found.get(k)) for k, v in CELL_LAYERS.items() if found.get(k) != v}
    if bad:
        raise ValueError(f"cell layer numbers disagree with the DRC deck: {bad}")
    return found


def _check_grid(deck: Path = None) -> float:
    """Grep the deck's own off-grid threshold back, so `GRID` is not just an assertion."""
    text = Path(deck or O.default_drc_deck()).read_text()
    got = sorted({float(v) for v in re.findall(r"\.ongrid\((\d*\.?\d+)\)", text)})
    if got != [GRID]:
        raise ValueError(f"the deck's ongrid thresholds are {got}, this module snaps to "
                         f"{GRID}")
    return GRID


def grid(v: float) -> float:
    """Snap a computed dimension to the manufacturing grid."""
    return round(round(v / GRID) * GRID, 6)


def grid_up(v: float) -> float:
    return round(np.ceil(round(v / GRID, 6)) * GRID, 6)


# --------------------------------------------------------------------------
# The cell plan: target resistance in, geometry out.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CellSpec:
    """A drawable 0T1R cell.  Every dimension is derived, none is chosen by hand.

    The resistor is `stripes` straight poly bars, each `body` um of marked resistor between
    two contact heads, wired in series by li1 straps that alternate top and bottom.  It is
    **not** a continuous poly serpentine, and the difference is not cosmetic:

      * a poly corner is worth somewhere between 0.5 and 0.6 squares depending on whose
        conformal map you believe, and a *precision* resistor whose value is set by a
        corner count is not precise.  Straight bars have an exact square count.
      * the LVS deck extracts each bar as its own `res_xhigh_po` device with its own `L`,
        so a K-stripe cell is K length comparisons rather than one square-counted guess.
      * it costs ~3% in pitch.  The head overhead is `2 * head` per stripe against a stripe
        `body` tens of times longer, so a strapped block is barely larger than a serpentined
        one -- `main` prints both.

    The price is real and is reported rather than hidden: one contact-head resistance per
    stripe, plus the li1 straps, in series with the body.  At 1 Mohm it is half a percent; at
    20 kohm, which is where the `g_min` sweep ends up, it is 6%.
    """
    dev: ResDevice
    rules: dict
    r_target: float
    stripes: int
    body: float          # um, one stripe's marked resistor length -- the LVS `L`
    head: float          # um, poly beyond the marker at each end of a stripe
    squares: float       # stripes * body / width, exact for rectangles
    r_body: float        # ohm, r_sheet * squares
    r_end: float         # ohm, stripes x the model's own contact-head resistance
    r_contact: float     # ohm, the same heads priced from magic's `xpc`, for comparison only
    r_strap: float       # ohm, the li1 straps
    block: tuple         # (w, h) um of the poly block
    origin: tuple        # (x, y) um of the poly block inside the cell
    pitch: float
    pitch_limit: str     # which constraint set the pitch -- the whole g_min story turns on it
    w_row: float
    w_col: float
    serp_side: float     # um, side of the equivalent continuous serpentine block

    @property
    def r_cell(self) -> float:
        """The whole cell's resistance, body plus the parasitics of getting into it."""
        return self.r_body + self.r_end + self.r_strap

    @property
    def area(self) -> float:
        return self.pitch ** 2

    def __str__(self):
        return (f"{self.r_target / 1e3:.0f} kohm target -> {self.stripes} x {self.body:g} "
                f"um of {self.dev.width:g} um {DEVICE_FAMILY} = {self.squares:.1f} squares "
                f"= {self.r_body / 1e3:.2f} kohm body; block "
                f"{self.block[0]:.2f} x {self.block[1]:.2f} um, cell pitch "
                f"{self.pitch:g} um")


def tap_stack(rules: dict, w_row: float) -> tuple:
    """`(y, height)` of the column terminal's met1 island, above the row rail.

    One source of truth, because `plan_cell` needs it to place the poly block and `draw_cell`
    needs it to draw the island: if the two computed it separately they could drift apart and
    the only symptom would be a via landing off its metal.  The height is set by whichever of
    the two cuts that land on the island needs more enclosure -- the mcon (`m1.5`) or the via
    (`via1.5a`).
    """
    height = grid_up(max(rules["ct.1"] + 2 * rules["m1.5"],
                         rules["via.1a"] + 2 * rules["via1.5a"]) + SLACK)
    return grid_up(w_row + rules["m1.2"] + SLACK), height


def plan_cell(tech: O.TechConstants, r_target: float, dev: ResDevice = None,
              rules: dict = None, w_row: float = W_ROW, w_col: float = W_COL) -> CellSpec:
    """Size and place one cell for a target resistance.

    The stripe count is the one that makes the poly block squarest, because a square block
    minimizes the pitch -- and the pitch, not the area, is what sets the rail resistance.
    Solving `stripes * p - s == body + 2 * head` with `stripes * body = total` gives a
    quadratic in `stripes`; it is rounded to an even number so both terminals come out at
    the bottom of the block, one under the row rail and one under the column rail.
    """
    rules = rules or cell_rules()
    dev = dev or res_devices(tech)[0]
    w, sp = dev.width, rules["poly.9"]
    p = w + sp                                   # stripe pitch
    head = grid_up(2 * rules["licon.8"] + rules["licon.1"] + SLACK)
    total = r_target / dev.r_sheet * w           # um of marked resistor needed

    a, b, c = p, -(sp + 2 * head), -total
    k = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    stripes = max(2, 2 * int(round(k / 2)))
    body = grid(total / stripes)
    if body < rules["licon.1"]:
        raise ValueError(f"{r_target:g} ohm needs a {body:g} um stripe body, which cannot "
                         f"hold a {rules['licon.1']:g} um contact; raise r_target or use a "
                         "narrower device")
    squares = stripes * body / w
    block = (grid(stripes * p - sp), grid(body + 2 * head))

    # Vertical stack, bottom up: row rail, then the column terminal's met1 island, then the
    # poly block.  Horizontally the block clears the column rail.
    y_island, island_h = tap_stack(rules, w_row)
    origin = (grid_up(w_col + SLACK), grid_up(y_island + island_h + SLACK))

    # The pitch.  Two families of constraint, and which one binds is the whole point of the
    # g_min sweep: while the resistor is big the block dominates and the pitch tracks
    # sqrt(1/g_min), but once it is small the rails and the terminal stack take over and the
    # pitch stops shrinking -- at which point raising g_min buys no rail R and only costs
    # signal.  `pitch_limit` names the binder so that turnover is visible, not inferred.
    #   fit      the block and its markers inside the cell box, above the rail/tap stack
    #   gap      adjacent cells' marker rectangles must not crowd each other, and `urpm`
    #            wants the widest gap of the three (rpm.2, 0.84 um)
    mark, gap = rules["rpm.3"], rules["rpm.2"]
    bounds = {"tap stack + block (y)": origin[1] + block[1] + mark + SLACK,
              "col rail + block (x)": origin[0] + block[0] + mark + SLACK,
              "urpm gap (x)": block[0] + 2 * mark + gap,
              "urpm gap (y)": block[1] + 2 * mark + gap}
    pitch_limit = max(bounds, key=bounds.get)
    pitch = grid_up(bounds[pitch_limit])

    # One strap per junction, `p` long between licon centres at li1's own sheet resistance.
    r_strap = (stripes - 1) * tech.r_sheet("li1") * p / (rules["licon.1"] + 2 * rules["li.5"])
    return CellSpec(dev=dev, rules=rules, r_target=r_target, stripes=stripes, body=body,
                    head=head, squares=squares, r_body=dev.r_sheet * squares,
                    r_end=stripes * dev.r_end,
                    r_contact=2 * stripes * tech.contact_res["xpc"].value / 1e3,
                    r_strap=r_strap, block=block, origin=origin,
                    pitch=pitch, pitch_limit=pitch_limit, w_row=w_row, w_col=w_col,
                    serp_side=float(np.sqrt(total * p)))


# --------------------------------------------------------------------------
# Drawing.  The unit cell holds everything except the rails, which span the whole array
# and therefore live in the top cell; DRC runs `deep` so the hierarchy costs nothing.
# --------------------------------------------------------------------------
def _cell_layers(ly, layers: O.Layers) -> dict:
    lay = {k: ly.layer(*v) for k, v in CELL_LAYERS.items()}
    for name in ("li1", "mcon", "met1", "via", "met2"):
        lay[name] = ly.layer(*layers[name])
    return lay


def draw_cell(ly, cell, sp: CellSpec, layers: O.Layers, stripe_w: float = None):
    """One cell's resistor, contacts and terminal taps, at the cell origin.

    `stripe_w` overrides the drawn poly width; `bad_cell_gds` uses it to plant a `poly.3`
    violation and nothing else touches it.
    """
    lay = _cell_layers(ly, layers)
    R = sp.rules
    w = sp.dev.width if stripe_w is None else stripe_w
    p = sp.dev.width + R["poly.9"]
    bx, by = sp.origin
    bh = sp.block[1]
    lic, li_pad = R["licon.1"], R["licon.1"] + 2 * R["li.5"]
    x_c = [bx + j * p + sp.dev.width / 2 for j in range(sp.stripes)]   # stripe centrelines
    y_c = (by + sp.head / 2, by + bh - sp.head / 2)                    # head centres

    for j in range(sp.stripes):
        x0 = bx + j * p + (sp.dev.width - w) / 2
        cell.shapes(lay["poly"]).insert(O.rect(x0, by, x0 + w, by + bh))
        for yc in y_c:
            cell.shapes(lay["licon"]).insert(
                O.rect(x_c[j] - lic / 2, yc - lic / 2, x_c[j] + lic / 2, yc + lic / 2))

    # The resistor body: poly inside the 66/13 marker.  Everything outside it is `poly_con`,
    # which is what the LVS deck takes as the terminals -- so the marker edge, not the
    # contact, is what sets the extracted `L`.
    mark = R["rpm.3"]
    cell.shapes(lay["poly_rs"]).insert(
        O.rect(bx - mark, by + sp.head, bx + sp.block[0] + mark, by + bh - sp.head))
    for name in ("urpm", "psdm", "npc"):
        cell.shapes(lay[name]).insert(
            O.rect(bx - mark, by - mark, bx + sp.block[0] + mark, by + bh + mark))

    # Series straps: top for pairs (0,1), (2,3), ...; bottom for (1,2), (3,4), ... .  With
    # an even stripe count that leaves stripe 0 and stripe -1 free at the bottom, which are
    # the two terminals.
    for j in range(0, sp.stripes - 1):
        yc = y_c[1] if j % 2 == 0 else y_c[0]
        cell.shapes(lay["li1"]).insert(
            O.rect(x_c[j] - li_pad / 2, yc - li_pad / 2,
                   x_c[j + 1] + li_pad / 2, yc + li_pad / 2))

    # Row terminal: the last stripe's bottom head, straight down onto the met1 row rail.
    y_mcon_row = sp.w_row / 2
    cell.shapes(lay["li1"]).insert(
        O.rect(x_c[-1] - li_pad / 2, y_mcon_row - li_pad / 2,
               x_c[-1] + li_pad / 2, y_c[0] + li_pad / 2))
    cell.shapes(lay["mcon"]).insert(
        O.rect(x_c[-1] - R["ct.1"] / 2, y_mcon_row - R["ct.1"] / 2,
               x_c[-1] + R["ct.1"] / 2, y_mcon_row + R["ct.1"] / 2))

    # Column terminal: the first stripe's bottom head, down to a met1 island that runs left
    # under the met2 column rail and takes a via up to it.  The island clears the row rail
    # by m1.2; the row terminal's li1 crosses over it on a different layer with no mcon
    # between them, which is what keeps the two nets apart.
    y0, island_h = tap_stack(R, sp.w_row)
    y_mcon_col = y0 + island_h / 2
    cell.shapes(lay["li1"]).insert(
        O.rect(x_c[0] - li_pad / 2, y_mcon_col - li_pad / 2,
               x_c[0] + li_pad / 2, y_c[0] + li_pad / 2))
    cell.shapes(lay["mcon"]).insert(
        O.rect(x_c[0] - R["ct.1"] / 2, y_mcon_col - R["ct.1"] / 2,
               x_c[0] + R["ct.1"] / 2, y_mcon_col + R["ct.1"] / 2))
    cell.shapes(lay["met1"]).insert(
        O.rect(0.0, y0, x_c[0] + li_pad / 2, y0 + island_h))
    cell.shapes(lay["via"]).insert(
        O.rect(sp.w_col / 2 - R["via.1a"] / 2, y_mcon_col - R["via.1a"] / 2,
               sp.w_col / 2 + R["via.1a"] / 2, y_mcon_col + R["via.1a"] / 2))


def cell_gds(path: Path, sp: CellSpec, layers: O.Layers = None, labels: bool = False,
             stripe_w: float = None, cellname: str = CELL_NAME) -> Path:
    """One cell with a one-pitch stub of each rail: what cell-level DRC and LVS run on."""
    layers = layers or O.Layers()
    _check_layers()
    ly, top = O.new_layout(cellname)
    lay = _cell_layers(ly, layers)
    top.shapes(lay["met1"]).insert(O.rect(0.0, 0.0, sp.pitch, sp.w_row))
    top.shapes(lay["met2"]).insert(O.rect(0.0, 0.0, sp.w_col, sp.pitch))
    draw_cell(ly, top, sp, layers, stripe_w=stripe_w)
    if labels:
        top.shapes(ly.layer(*O.LABEL_LAYERS["met1_label"])).insert(
            kdb.Text(ROW_NET, O.um(sp.pitch - sp.w_row), O.um(sp.w_row / 2)))
        top.shapes(ly.layer(*MET2_LABEL)).insert(
            kdb.Text(COL_NET, O.um(sp.w_col / 2), O.um(sp.pitch - sp.w_col)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


def narrow_stripe(sp: CellSpec) -> float:
    """The widest illegal stripe width that keeps every vertex on grid.

    `draw_cell` centres a narrowed stripe on the same centreline, so the width has to come
    down in steps of `2 * GRID` or the stripe edges land between grid points and the
    off-grid rules fire instead of the one being tested.
    """
    step = 2 * GRID
    k = int(np.ceil((sp.dev.width - (sp.rules["poly.3"] - GRID)) / step))
    return grid(sp.dev.width - k * step)


def bad_cell_gds(path: Path, sp: CellSpec, layers: O.Layers = None) -> Path:
    """The real cell with its resistor stripes narrowed below `poly.3`, and nothing else.

    A FEOL-enabled harness that has only ever seen clean layouts is not a tested harness,
    and `layout_oracle.bad_wire_gds` proves nothing about the FEOL half of the deck -- it
    is metal only.  This plants a violation on a *front-end* layer of the real structure,
    so the same layout is clean with `FEOL = false` and dirty with `FEOL = true`.
    """
    return cell_gds(path, sp, layers, stripe_w=narrow_stripe(sp), cellname="BADCELL")


def array_gds(path: Path, sp: CellSpec, n: int, layers: O.Layers = None,
              labels: bool = False, cellname: str = ARRAY_NAME) -> Path:
    """N x N: full-length rails in the top cell, one instance array of the unit cell.

    Instancing rather than flattening is not just tidiness -- an n = 16 array of a 16-stripe
    cell is ~50 000 polygons flat, and the DRC deck runs `deep`, so the hierarchy is what
    keeps a FEOL run on the array in the same time class as a run on the cell.
    """
    layers = layers or O.Layers()
    _check_layers()
    ly, top = O.new_layout(cellname)
    unit = ly.create_cell(CELL_NAME)
    draw_cell(ly, unit, sp, layers)
    lay = _cell_layers(ly, layers)
    span = n * sp.pitch
    m1_txt = ly.layer(*O.LABEL_LAYERS["met1_label"])
    m2_txt = ly.layer(*MET2_LABEL)
    for i in range(n):
        y = i * sp.pitch
        top.shapes(lay["met1"]).insert(O.rect(0.0, y, span, y + sp.w_row))
        if labels:
            top.shapes(m1_txt).insert(
                kdb.Text(f"{ROW_NET}{i}", O.um(span - sp.w_row), O.um(y + sp.w_row / 2)))
    for j in range(n):
        x = j * sp.pitch
        top.shapes(lay["met2"]).insert(O.rect(x, 0.0, x + sp.w_col, span))
        if labels:
            top.shapes(m2_txt).insert(
                kdb.Text(f"{COL_NET}{j}", O.um(x + sp.w_col / 2), O.um(span - sp.w_col)))
    top.insert(kdb.CellInstArray(unit.cell_index(), kdb.Trans(),
                                 kdb.Vector(O.um(sp.pitch), 0), kdb.Vector(0, O.um(sp.pitch)),
                                 n, n))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


# --------------------------------------------------------------------------
# DRC with FEOL on.
# --------------------------------------------------------------------------
def verify_cell(gds: Path, sp: CellSpec) -> dict:
    """Read the written GDS back and re-derive the resistance from it.

    docs/LAYOUT.md 5.1 records three separate wrong results from the SI-versus-micron
    boundary, and a resistor adds two more places to get it wrong: a drawn length in
    database units, and a model `l` the LVS deck reads as metres.  So the drawn body is
    measured back out of the file -- as the LVS deck sees it, `poly` AND the `poly_rs`
    marker -- and every derived number is checked against the plan rather than trusted.

    Note that magic and KLayout do not even agree on what `l` means for this device: magic's
    extraction rule is `l=l+0.16` (`sky130A.tech:6058`), KLayout's is the marked length
    itself.  Neither is wrong; both cannot be assumed.
    """
    ly = kdb.Layout()
    ly.read(str(gds))
    top = ly.top_cell()
    body = (kdb.Region(top.begin_shapes_rec(ly.layer(*CELL_LAYERS["poly"])))
            & kdb.Region(top.begin_shapes_rec(ly.layer(*CELL_LAYERS["poly_rs"]))))
    body.merge()
    widths = sorted({round(min(p.bbox().width(), p.bbox().height()) * ly.dbu, 6)
                     for p in body.each()})
    lengths = sorted({round(max(p.bbox().width(), p.bbox().height()) * ly.dbu, 6)
                      for p in body.each()})
    got = {"n_bodies": body.size(), "widths": widths, "lengths": lengths,
           "area": body.area() * ly.dbu ** 2}
    got["squares"] = got["area"] / sp.dev.width ** 2
    got["r_body"] = got["squares"] * sp.dev.r_sheet
    got["l_total"] = sum(max(p.bbox().width(), p.bbox().height()) * ly.dbu
                         for p in body.each())
    for name, a, b in (("body count", got["n_bodies"], sp.stripes),
                       ("drawn width", widths, [sp.dev.width]),
                       ("body length", lengths, [sp.body]),
                       ("squares", got["squares"], sp.squares),
                       ("body R", got["r_body"], sp.r_body)):
        if isinstance(a, list) or isinstance(a, int):
            if a != b:
                raise AssertionError(f"{name} measured back out of {gds.name} is {a!r}, "
                                     f"the plan says {b!r}")
        elif abs(a - b) > 1e-9 * max(abs(a), abs(b), 1e-30):
            raise AssertionError(f"{name} from GDS {a!r} != plan {b!r}")
    return got


def feol_deck(path: Path, src: Path = None) -> Path:
    """A copy of the stock deck with `FEOL = false` flipped to true.

    The flag is a plain Ruby assignment (`sky130A.lydrc:46`), not a `-rd` variable, so there
    is no way to enable the front-end rules from the command line.  Patching a *copy* keeps
    the PDK read-only.
    """
    src = Path(src or O.default_drc_deck())
    text, n = re.subn(r"(?m)^FEOL(\s*)=(\s*)false", r"FEOL\1=\2true", src.read_text())
    if n != 1:
        raise ValueError(f"{src}: expected exactly one `FEOL = false` line, found {n}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# LVS.
# --------------------------------------------------------------------------
def hdl21_cell_netlist(path: Path, sp: CellSpec, cellname: str = CELL_NAME) -> Path:
    """The cell's schematic: `stripes` real sky130 resistors in series, through hdl21.

    The device is `sky130_hdl21.ress[...]`, the PDK's own `ExternalModule`, instantiated
    directly rather than through `h.PhysicalResistor` -- because `Sky130Walker`'s resistor
    path throws the caller's length away and substitutes `default_prec_res_L`
    (`pdk_logic.py:254`), which would netlist every cell as a 0.35 um resistor.

    Instance names must start with `r`.  The LVS deck's reader delegate turns an `X` call
    into a primitive by taking the *first character of the instance name* as the element
    letter (`sky130.lvs:404`), so `xr0` becomes an `R` and `xres0` would become... also an
    `R`, but `xcell0` would become a capacitor.
    """
    import hdl21 as h
    import sky130_hdl21

    res = sky130_hdl21.ress[sp.dev.hdl21_key]
    m = h.Module(name=cellname)
    m.add(h.Port(), name=ROW_NET)
    m.add(h.Port(), name=COL_NET)
    body = m.add(h.Port(), name=LVS_SUB)
    prev = m.get(COL_NET)
    for k in range(sp.stripes):
        nxt = m.get(ROW_NET) if k == sp.stripes - 1 else m.add(h.Signal(), name=f"n{k}")
        m.add(res(sky130_hdl21.Sky130PrecResParams(l=sp.body))(p=prev, n=nxt, b=body),
              name=f"r{k}")
        prev = nxt
    C.sky130_install()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        h.netlist(h.to_proto([m]), f, fmt="spice")
    return path


def hdl21_array_netlist(path: Path, sp: CellSpec, n: int,
                        cellname: str = ARRAY_NAME) -> Path:
    """The N x N array's schematic: one `xbar_cell` subckt, n^2 instances, 2n rail nets.

    docs/LAYOUT.md 10 lists "no crossbar layout has been LVS'd" as a limitation, on the
    grounds that a crossbar with no device in it has nothing to match against.  With a real
    device in the cell that reason is gone, so this builds the array reference too.  The
    rails carry no devices -- metal only extracts as a resistor if it is marked with
    `m1_res`, which nothing here draws -- so electrically the array is n^2 cells between 2n
    nets, which is exactly what `crossbar.Crossbar` is minus the per-pitch rail resistors.
    """
    import hdl21 as h
    import sky130_hdl21

    res = sky130_hdl21.ress[sp.dev.hdl21_key]
    cell = h.Module(name=CELL_NAME)
    cell.add(h.Port(), name=ROW_NET)
    cell.add(h.Port(), name=COL_NET)
    cbody = cell.add(h.Port(), name=LVS_SUB)
    prev = cell.get(COL_NET)
    for k in range(sp.stripes):
        nxt = cell.get(ROW_NET) if k == sp.stripes - 1 else cell.add(h.Signal(), name=f"n{k}")
        cell.add(res(sky130_hdl21.Sky130PrecResParams(l=sp.body))(p=prev, n=nxt, b=cbody),
                 name=f"r{k}")
        prev = nxt

    top = h.Module(name=cellname)
    rows = [top.add(h.Port(), name=f"{ROW_NET}{i}") for i in range(n)]
    cols = [top.add(h.Port(), name=f"{COL_NET}{j}") for j in range(n)]
    body = top.add(h.Port(), name=LVS_SUB)
    for i in range(n):
        for j in range(n):
            top.add(cell(**{ROW_NET: rows[i], COL_NET: cols[j], LVS_SUB: body}),
                    name=f"c{i}_{j}")
    C.sky130_install()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        h.netlist(h.to_proto([cell, top]), f, fmt="spice")
    return path


def _scale_l(text: str, factor: float) -> str:
    """Multiply every adapted `l='(x)*1e-6'` by `factor`, for the tolerance probe."""
    return re.sub(r"l='\(([\d.eE+-]+)\)\*1e-6'",
                  lambda m: f"l='({float(m.group(1)) * factor:.8f})*1e-6'", text)


def lvs_probes(gds: Path, netlist: Path, rundir: Path) -> list:
    """Measure what LVS compares, by breaking one thing at a time.

    `netlist` is the adapted (metres) reference that already matches.  Each probe changes
    exactly one thing and re-runs, so "L is compared" and "R is not" become observations
    rather than a reading of the deck source.  Returns [(label, expectation, matched), ...].
    """
    rundir = Path(rundir)
    base = Path(netlist).read_text()
    probes = [("reference, unchanged", True, base),
              ("L +0.4% (inside the 0.5% tolerance)", True, _scale_l(base, 1.004)),
              ("L +0.6% (outside it)", False, _scale_l(base, 1.006)),
              ("L +10%", False, _scale_l(base, 1.10)),
              ("device bin 0p35 -> 0p69", False, base.replace("_0p35", "_0p69"))]
    out = []
    for i, (label, want, text) in enumerate(probes):
        p = rundir / f"probe{i}.spice"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        r = O.run_lvs(gds, p, rundir / f"probe{i}", extra=LVS_EXTRA)
        out.append((label, want, r.ok))
    return out


# --------------------------------------------------------------------------
# Parasitics at the real pitch, and the g_min / area / IR-drop trade.
# --------------------------------------------------------------------------
def derived_params(tech: O.TechConstants, sp: CellSpec, rundir: Path, n: int) -> tuple:
    """(row PerPitch, col PerPitch, CrossbarParams) for a cell spec, at its own pitch.

    Straight reuse of `array_layout.per_pitch`: two rails one pitch apart in length,
    measured through the GDS round trip and differenced.  Nothing about the finite-
    difference method changes because the pitch now comes from a device.
    """
    row = A.per_pitch(tech, A.Skeleton.row_layer, sp.w_row, sp.pitch, rundir, n=n)
    col = A.per_pitch(tech, A.Skeleton.col_layer, sp.w_col, sp.pitch, rundir, n=n)
    g_min = 1.0 / sp.r_cell
    xbar = replace(F.IDEAL_XBAR, r_row=row.r, r_col=col.r, c_row=row.c, c_col=col.c,
                   g_min=g_min)
    return row, col, xbar


@dataclass(frozen=True)
class GminPoint:
    """One point of the `g_min` sweep: cell geometry and the rail R it implies."""
    g_min: float
    spec: CellSpec
    r_row: float
    r_col: float
    c_row: float
    c_col: float
    i_common: float      # A, the per-column common-mode floor N * g_min * V
    i_signal: float      # A, full-scale differential signal N * (g_max - g_min) * V
    v_drop: float        # V, worst-case IR drop of the common-mode current along a column

    @property
    def delta(self) -> float:
        return F.IDEAL_XBAR.g_max - self.g_min


def gmin_sweep(tech: O.TechConstants, g_mins, rundir: Path, n: int = 16,
               rules: dict = None, dev: ResDevice = None) -> list:
    """Cell area, pitch and rail R against `g_min`, with the two currents that fight.

    The differential `Tile` computes `G+ - G- = (g_max - g_min) * w`, so `g_min` cancels
    exactly in the signal and raising it costs only the `delta` it eats -- 1 uS to 10 uS is
    9% of the range.  What it buys is area: the largest resistor is `1 / g_min`, so 10x the
    conductance is 10x fewer squares and ~3.2x less cell side, which is ~3.2x less rail R
    per pitch.  What it costs, beyond `delta`, is the common-mode current every column
    carries whether or not any weight is on: `N * g_min * V` per column, and that current
    drops voltage along the same rail.  `v_drop` is the sum over the column's segments,
    `sum_i i * I * r_col` with `I` the per-cell common-mode current -- the standard
    `N(N+1)/2` ladder, which is what makes it grow faster than the pitch shrinks.
    """
    out = []
    for g in g_mins:
        sp = plan_cell(tech, 1.0 / g, dev=dev, rules=rules)
        row, col, _ = derived_params(tech, sp, rundir, n)
        i_cell = g * F.V_MAX
        i_common = n * i_cell
        # Ladder sum: the segment nearest the readout carries all n cells' current.
        v_drop = col.r * i_cell * n * (n + 1) / 2
        out.append(GminPoint(g_min=g, spec=sp, r_row=row.r, r_col=col.r, c_row=row.c,
                             c_col=col.c, i_common=i_common,
                             i_signal=n * (F.IDEAL_XBAR.g_max - g) * F.V_MAX,
                             v_drop=v_drop))
    return out


# --------------------------------------------------------------------------
def main(accuracy: bool = False, rundir: Path = Path("/tmp/xbar_cell"), n: int = 16):
    t_start = time.perf_counter()
    rundir = Path(rundir)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)

    tech = O.TechConstants.parse()
    layers = O.Layers()
    _check_layers()
    _check_grid()
    rules = cell_rules()
    devs = res_devices(tech)
    dev = devs[0]

    print("=" * 78)
    print("STAGE 3  a real device-level 0T1R cell: draw it, DRC it with FEOL on, tile it")
    print("=" * 78)
    print(f"tech  {tech.path}")
    print(f"grid  {GRID} um, from the deck's own poly.ongrid()")

    print("\n" + "-" * 78)
    print("the device decision")
    print("-" * 78)
    for line in _wrap(DEVICE_NOTE, 76):
        print(f"  {line}")
    print(f"\n  binned {DEVICE_FAMILY} widths the PDK ships a model for: "
          + ", ".join(f"{d.width:g}" for d in devs) + " um")
    print(f"  chosen: {dev}")
    print(tech.provenance(DEVICE_TYPESET, "xhrpoly", "xpc"))
    print(f"  the tech file's {DEVICE_TYPESET} = {dev.r_sheet:g} ohm/square is the same "
          f"number\n  {DEVICE_FAMILY}__base.model.spice uses as `rsheet`, so the geometry "
          "and the\n  simulation model agree by construction rather than by luck.")
    print("  length is the only free parameter: width is pinned to a bin by the LVS deck's\n"
          "  own recognition layers (sky130.lvs:1412), so the cell is a strapped stack of\n"
          "  fixed-width bars.")

    print("\n" + "-" * 78)
    print(f"{len(rules)} DRC dimensions, every one parsed off the line that emits the rule")
    print("-" * 78)
    items = sorted(rules.items())
    for i in range(0, len(items), 4):
        print("  " + "".join(f"{k:>10s} {v:<7g}" for k, v in items[i:i + 4]))

    print("\n" + "-" * 78)
    print("the cell, sized for the project's default g_min = "
          f"{F.IDEAL_XBAR.g_min * 1e6:g} uS")
    print("-" * 78)
    sp = plan_cell(tech, 1.0 / F.IDEAL_XBAR.g_min, dev=dev, rules=rules)
    print(f"  {sp}")
    print(f"  {sp.stripes} stripes at {dev.width + rules['poly.9']:g} um stripe pitch "
          f"({dev.width:g} drawn + {rules['poly.9']:g} poly.9 space), "
          f"{sp.head:g} um contact head per end")
    print(f"  resistance, drawn:  {'body':<22s}{sp.r_body / 1e3:9.2f} kohm  "
          f"({sp.squares:.1f} squares x {dev.r_sheet:g} ohm/square)")
    print(f"                      {f'{sp.stripes} contact heads':<22s}"
          f"{sp.r_end / 1e3:9.2f} kohm  "
          f"({100 * sp.r_end / sp.r_cell:.2f}% of the cell, at {dev.r_end:.1f} ohm/stripe "
          "from the model)")
    print(f"                      {'li1 straps':<22s}{sp.r_strap / 1e3:9.2f} kohm  "
          f"({100 * sp.r_strap / sp.r_cell:.2f}%)")
    print(f"                      {'total':<22s}{sp.r_cell / 1e3:9.2f} kohm  "
          f"-> g = {1e6 / sp.r_cell:.4f} uS against the {F.IDEAL_XBAR.g_min * 1e6:g} uS "
          "asked for")
    print(f"  magic prices the same {2 * sp.stripes} cuts at "
          f"{tech.contact_res['xpc'].value / 1e3:g} ohm each = {sp.r_contact / 1e3:.2f} kohm, "
          f"{sp.r_end / sp.r_contact:.2f}x less than the\n  model's own end resistance. Two "
          "PDK statements of one thing; the model's is used, being\n  the one that would "
          "simulate.")
    print(f"  cell bounding box: {sp.pitch:g} x {sp.pitch:g} um "
          f"({sp.area:.1f} um^2), block {sp.block[0]:.2f} x {sp.block[1]:.2f} um at "
          f"({sp.origin[0]:g}, {sp.origin[1]:g})")
    print(f"  a continuous poly serpentine of the same line would be "
          f"{sp.serp_side:.2f} um square,\n  so the strapped block costs "
          f"{100 * (max(sp.block) / sp.serp_side - 1):+.1f}% in side length and buys an "
          "exact square count.")
    print(f"  docs/LAYOUT.md 8 put this cell at 30.1 um square using res_high_po's "
          f"319.8 ohm/square;\n  on the right device it is "
          f"{max(sp.block):.1f} um square, a "
          f"{(30.1 / max(sp.block)) ** 2:.1f}x area saving.")

    print("\n" + "-" * 78)
    print("DRC, with FEOL enabled -- and the proof that enabling it changed something")
    print("-" * 78)
    deck = feol_deck(rundir / "sky130A_feol.lydrc")
    print(f"  patched deck: {deck}  (FEOL = false -> true; the PDK copy is untouched)")
    gds = cell_gds(rundir / "cell.gds", sp, layers)
    m = verify_cell(gds, sp)
    print(f"  measured back out of {gds.name}: {m['n_bodies']} marked bodies, "
          f"widths {m['widths']}, lengths {m['lengths']} um\n"
          f"  -> {m['l_total']:g} um of marked resistor, {m['squares']:.4f} squares, "
          f"{m['r_body'] / 1e3:.2f} kohm; matches the plan exactly")

    off = O.run_drc(gds, rundir, deck=None)
    on = O.run_drc(gds, rundir / "feol", deck=deck)
    print(f"  cell, FEOL off  {off.n_rules_checked:4d} rules  {off}")
    print(f"  cell, FEOL on   {on.n_rules_checked:4d} rules  {on}")
    bad = bad_cell_gds(rundir / "badcell.gds", sp, layers)
    bad_off = O.run_drc(bad, rundir, deck=None)
    bad_on = O.run_drc(bad, rundir / "feol", deck=deck)
    print(f"  planted violation: stripes narrowed to {narrow_stripe(sp):g} um, "
          f"poly.3 wants {rules['poly.3']:g}")
    print(f"    FEOL off      {bad_off}")
    print(f"    FEOL on       {bad_on}")
    if not on.clean:
        raise AssertionError(f"the cell is not DRC clean with FEOL on: {on.by_rule}")
    if bad_off.by_rule:
        raise AssertionError("the planted front-end violation should be invisible with "
                             f"FEOL off, got {bad_off.by_rule}")
    if bad_on.by_rule.get("poly.3") != sp.stripes:
        raise AssertionError(f"FEOL on should flag poly.3 once per stripe "
                             f"({sp.stripes}), got {bad_on.by_rule}")
    print(f"  harness proven on the front end: {on.n_rules_checked - off.n_rules_checked} "
          "extra rule categories, a clean\n  cell, and one narrowing that is invisible with "
          "FEOL off and trips poly.3 once per\n  stripe with FEOL on (plus licon.8a, the "
          "poly enclosure of the contacts the same\n  narrowing also breaks -- also a "
          "front-end rule, also invisible with FEOL off).")

    print("\n" + "-" * 78)
    print("LVS: the drawn cell against an hdl21 netlist of the real sky130 device")
    print("-" * 78)
    lvs_gds = cell_gds(rundir / "cell_lvs.gds", sp, layers, labels=True)
    net = hdl21_cell_netlist(rundir / "cell_ref.spice", sp)
    print(f"  layout   {lvs_gds}   netlist  {net}")
    print("  " + "\n  ".join(l for l in net.read_text().splitlines()[6:12] if l.strip()))
    si = O.lvs_netlist_units(net, rundir / "cell_ref_si.spice")
    for k, (tag, n_, extra) in enumerate((("as hdl21 emits it", net, LVS_EXTRA),
                                          ("um->m adapter, no schematic_simplify", si, None),
                                          ("um->m adapter + schematic_simplify", si,
                                           LVS_EXTRA))):
        r = O.run_lvs(lvs_gds, n_, rundir / f"lvs{k}", extra=extra)
        print(f"  {tag:38s} {r}")
        fixed = r
    if not fixed.ok:
        print("  --- deck output tail ---")
        print("  " + "\n  ".join(fixed.stdout_tail.strip().splitlines()[-14:]))
        raise AssertionError("the cell should LVS-match its hdl21 netlist")
    print(f"  extracted netlist: {fixed.extracted}")
    print("  " + "\n  ".join(fixed.extracted.read_text().strip().splitlines()[2:]))
    print(f"  the {sp.stripes} drawn stripes extract as ONE device: KLayout's resistor class "
          "combines a\n  series chain during extraction, so R and L above are the whole "
          "cell's.")
    print("\n  what LVS compares, measured one broken thing at a time:")
    for label, want, got in lvs_probes(lvs_gds, si, rundir / "probes"):
        verdict = "MATCH" if got else "NO MATCH"
        print(f"    {label:38s} {verdict:9s}{'' if got == want else '  <-- UNEXPECTED'}")
        if got != want:
            raise AssertionError(f"LVS probe {label!r}: expected "
                                 f"{'MATCH' if want else 'NO MATCH'}")
    for line in _wrap(LVS_COMPARED, 74):
        print(f"    {line}")

    print("\n" + "-" * 78)
    print(f"the array: {n} x {n} of the real cell, at the real pitch")
    print("-" * 78)
    arr = array_gds(rundir / "array.gds", sp, n, layers, labels=True)
    a_on = O.run_drc(arr, rundir / "feol", deck=deck)
    m1 = O.measure(arr, A.Skeleton.row_layer, layers, require_boxes=False)
    poly = O.measure(arr, "poly", layers=_PolyLayers(), require_boxes=False)
    print(f"  {arr}  {n * sp.pitch:.1f} x {n * sp.pitch:.1f} um, "
          f"{n * n} cells, {n * n * sp.stripes} resistor stripes")
    print(f"  met1 {m1.area:.1f} um^2, poly {poly.area:.1f} um^2 "
          f"({100 * poly.area / (n * sp.pitch) ** 2:.1f}% of the die in resistor)")
    print(f"  FEOL on   {a_on.n_rules_checked} rules  {a_on}")
    if not a_on.clean:
        raise AssertionError(f"the array is not DRC clean with FEOL on: {a_on.by_rule}")

    a_net = O.lvs_netlist_units(hdl21_array_netlist(rundir / "array_ref.spice", sp, n),
                                rundir / "array_ref_si.spice")
    a_lvs = O.run_lvs(arr, a_net, rundir / "array_lvs", extra=LVS_EXTRA)
    print(f"  LVS       {a_lvs}")
    if a_lvs.ok:
        print(f"  {n * n} cells x {sp.stripes} stripes = {n * n * sp.stripes} drawn devices "
              f"against an hdl21 netlist of the same,\n  matched hierarchically on the "
              f"{CELL_NAME} subcircuit. docs/LAYOUT.md 10.8 said no crossbar\n  layout had "
              "been LVS'd because there was no device to match; there is one now.")
    else:
        print("  --- deck output tail ---")
        print("  " + "\n  ".join(a_lvs.stdout_tail.strip().splitlines()[-12:]))

    row, col, xbar = derived_params(tech, sp, rundir, n)
    print(f"  row {row}")
    print(f"  col {col}")
    print(f"  -> r_row = {row.r:.4g}, r_col = {col.r:.4g} ohm/pitch;  "
          f"c_row = {row.c:.4g}, c_col = {col.c:.4g} F/pitch")
    s = A.settling_estimate(A.Skeleton(n=n, pitch=sp.pitch), row)
    print(f"  row-line settling r*c*N^2/2 = {s['tau_elmore']:.3g} s, "
          f"x1.3 -> tau_63 ~ {s['tau_63']:.3g} s")
    print(f"\n  against the numbers this replaces:")
    print(f"    project default assumption      r_row = r_col = 1.0    ohm/pitch")
    print(f"    docs/LAYOUT.md 9 'g_min-driven' r_row = r_col = 3.875  ohm/pitch "
          "(31 um pitch, wrong device)")
    print(f"    derived here                    r_row = {row.r:.4g}, r_col = {col.r:.4g} "
          f"ohm/pitch ({sp.pitch:g} um pitch)")

    print("\n" + "-" * 78)
    print("g_min against cell area and rail IR drop: the two pressures, measured")
    print("-" * 78)
    pts = gmin_sweep(tech, GMIN_SWEEP, rundir, n=n, rules=rules, dev=dev)
    print(f"  N = {n}, V = {F.V_MAX} V, g_max = {F.IDEAL_XBAR.g_max * 1e6:g} uS, "
          f"{sp.w_row:g} um rails")
    hdr = ("g_min", "1/g_min", "squares", "stripes", "pitch", "area", "r_wire", "delta",
           "I_cm", "IR drop", "pitch set by")
    print(f"  {hdr[0]:>7s}{hdr[1]:>9s}{hdr[2]:>9s}{hdr[3]:>8s}{hdr[4]:>8s}{hdr[5]:>9s}"
          f"{hdr[6]:>8s}{hdr[7]:>7s}{hdr[8]:>8s}{hdr[9]:>9s}   {hdr[10]}")
    print(f"  {'[uS]':>7s}{'[kohm]':>9s}{'':>9s}{'':>8s}{'[um]':>8s}{'[um^2]':>9s}"
          f"{'[ohm]':>8s}{'[uS]':>7s}{'[uA]':>8s}{'[mV]':>9s}")
    for p in pts:
        print(f"  {p.g_min * 1e6:>7.0f}{p.spec.r_target / 1e3:>9.0f}"
              f"{p.spec.squares:>9.1f}{p.spec.stripes:>8d}{p.spec.pitch:>8.2f}"
              f"{p.spec.area:>9.0f}{p.r_row:>8.3f}{p.delta * 1e6:>7.0f}"
              f"{p.i_common * 1e6:>8.1f}{p.v_drop * 1e3:>9.3f}   {p.spec.pitch_limit}")
    print(f"  delta = g_max - g_min is the whole signal: g_min cancels in the differential\n"
          f"  pair, so 1 -> {GMIN_SWEEP[-1] * 1e6:g} uS costs "
          f"{100 * (1 - pts[-1].delta / pts[0].delta):.0f}% of the range and buys "
          f"{pts[0].spec.area / pts[-1].spec.area:.0f}x the cell area back.")
    floor = min(p.spec.pitch for p in pts)
    near = next(p for p in pts if p.spec.pitch <= 1.15 * floor)
    print(f"  but the pitch has a floor near {floor:g} um that the resistor has nothing to "
          f"do with: in x the\n  urpm-to-urpm gap plus two enclosures, in y the row rail, "
          "the column tap's met1 island\n  and its clearances. "
          f"By g_min = {near.g_min * 1e6:g} uS the pitch is already within 15% of it, so "
          "further\n  g_min buys almost no rail R and still costs delta -- which is where "
          "the int8 table\n  below turns over.")
    print(f"  the IR-drop column is the {n}-row ladder sum and scales as N^2: at N = 32 it "
          f"is 4x\n  larger ({4 * pts[0].v_drop * 1e3:.3f} mV at 1 uS, "
          f"{4 * pts[-1].v_drop * 1e3:.3f} mV at {GMIN_SWEEP[-1] * 1e6:g} uS), so a taller "
          "tile moves the optimum down.")

    if accuracy:
        print("\n" + "-" * 78)
        print("the payoff: docs/MATMUL.md 3.2, re-run on the device-derived parasitics")
        print(f"A {A.SHAPE[0]}x{A.SHAPE[1]} @ B {A.SHAPE[1]}x{A.SHAPE[2]}, "
              f"{A.TILE[0]}x{A.TILE[1]} tile, 0T1R, unipolar two-pass, batched")
        print("-" * 78)
        import int8_matmul as Q

        Am, Bm = A.operands()
        A_q = {False: Q.quantize(Am)[0], True: Q.quantize(Am, axis=1)[0]}
        B_q, _ = Q.quantize(Bm)
        rows = []
        for r in A.ASSUMED_R:
            rows.append(A.int8_row(A_q, B_q, replace(F.IDEAL_XBAR, r_row=r, r_col=r),
                                   f"assumed r_wire = {r:g}",
                                   f"/tmp/xbar_cell_acc/assumed_{r:g}"))
        rows.append(A.int8_row(A_q, B_q, replace(F.IDEAL_XBAR, r_row=3.875, r_col=3.875),
                               "LAYOUT.md 9 g_min-driven 3.875",
                               "/tmp/xbar_cell_acc/skeleton"))
        for p in pts:
            xb = replace(F.IDEAL_XBAR, r_row=p.r_row, r_col=p.r_col, c_row=p.c_row,
                         c_col=p.c_col, g_min=1.0 / p.spec.r_cell)
            rows.append(A.int8_row(
                A_q, B_q, xb,
                f"drawn g_min {p.g_min * 1e6:g} uS, r {p.r_row:.3g}",
                f"/tmp/xbar_cell_acc/g{p.g_min * 1e6:g}"))
        hdr = ("configuration", "pt raw", "pt+gain", "pc raw", "pc+gain", "LSB raw",
               "LSB cal", "bits(rms)")
        print(f"  {hdr[0]:<38s}{hdr[1]:>8s}{hdr[2]:>9s}{hdr[3]:>8s}{hdr[4]:>9s}"
              f"{hdr[5]:>9s}{hdr[6]:>9s}{hdr[7]:>11s}")
        for r in rows:
            print(f"  {r['tag']:<38s}{r['pt_raw']:>7.2f}%{r['pt_cal']:>8.2f}%"
                  f"{r['pc_raw']:>7.2f}%{r['pc_cal']:>8.2f}%"
                  f"{r['pc_raw_lsb']:>9d}{r['pc_cal_lsb']:>9d}{r['pc_bits']:>11.2f}")
        best = max(rows[len(A.ASSUMED_R) + 1:], key=lambda r: (r["pc_cal"], r["pc_bits"]))
        print(f"  ({sum(r['seconds'] for r in rows):.1f} s in ngspice)")
        print(f"  best drawn point: {best['tag']}")

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s, artifacts in {rundir}")


class _PolyLayers(O.Layers):
    """`layout_oracle.Layers` plus `poly`, so `measure` can be pointed at the resistor.

    `Layers` checks itself against the PDK's LEF/DEF map, which has no poly row, so poly
    cannot go in `EXPECTED`; it is cross-checked against the DRC deck instead.
    """

    def __init__(self, path: Path = None):
        super().__init__(path)
        self.ld["poly"] = _check_layers()["poly"]


def _wrap(text: str, width: int) -> list:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])


if __name__ == "__main__":
    main(accuracy="--accuracy" in sys.argv)
