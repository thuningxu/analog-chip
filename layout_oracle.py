"""Layout -> parasitics oracle: geometry in GDS, R and C out, DRC in between.

This is the piece the rest of the project did not have.  `crossbar.CrossbarParams`
takes `r_row` / `r_col` / `c_row` / `c_col` as *numbers*, and up to now those numbers
were assumed and swept (docs/REPORT.md says so in its first paragraph).  Here they get
computed from drawn geometry and sky130 process constants instead.

What this is, stated precisely, because the distinction matters:

    R  = sheet_resistance(layer) * squares(polygon)
    C  = area_cap(layer) * area(polygon) + perimeter_cap(layer) * perimeter(polygon)

Both constants are parsed out of the sky130 magic technology file, never typed in.  That
captures a wire's *self* resistance and its area plus fringe capacitance to substrate.
It is **not** parasitic extraction: there is no field solver, no 3-D coupling matrix, no
current crowding, no frequency dependence, and -- most importantly -- no rail-to-rail
coupling term in the numbers this module returns.  `array_layout.py` quantifies that last
omission separately and it is not small.  Call this "geometry + tech constants", not PEX.

Layer numbers are read out of the PDK layer map rather than hardcoded, and every parsed
constant carries the file and line it came from (`TechValue.line`), so any number this
module prints can be traced back to a grep.

Three corner sets of resistances live in the tech file.  `TechConstants.parse` takes the
one under `variants (),(orig),(si)` -- the default/typical set, whose own comment reads
"Device values come from trtc.cor (typical corner)".  See `CORNER_NOTE`.

Run it:  uv run layout_oracle.py
"""
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import klayout.db as kdb

import crossbar as C

DBU = 0.001   # um per database unit; sky130 GDS is on a 1 nm grid

KLAYOUT_BIN = Path.home() / "klayout-local/klayout.app/Contents/MacOS/klayout"

# The tech file's own three `variants` blocks, by the selector that heads each one.  The
# empty selector `()` is magic's default extraction style, and the block it heads is the
# only one whose comment claims a typical corner -- the other two say "High-end corner
# resistances" and "Low-end corner resistances" in as many words.  Picking the wrong one
# scales every resistance in this project by 105/125 or 145/125 with no other symptom.
DEFAULT_VARIANT = "()"

CORNER_NOTE = (
    "typical: the `variants (),(orig),(si)` block, headed 'Device values come from "
    "trtc.cor (typical corner)'.  The other two blocks are `variants (hrhc),(hrlc)` "
    "('High-end corner resistances') and `variants (lrhc),(lrlc)` ('Low-end corner "
    "resistances')."
)

# Reference values the parser must reproduce, so a silent corner slip or a tech-file
# revision fails loudly instead of biasing every downstream number.  mOhm/square.
SHEET_RES_REFERENCE = {
    "allli": 12800.0, "allm1": 125.0, "allm2": 125.0,
    "allm3": 47.0, "allm4": 47.0, "allm5": 29.0,
    "mrp1": 48200.0, "xhrpoly": 319800.0,
}
CAP_REFERENCE = {"area": ("allm1", 25.78), "perim": ("allm1", 40.57)}


# --------------------------------------------------------------------------
# sky130 technology constants, parsed.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TechValue:
    """One number from the tech file, with the units it is in and where it came from."""
    value: float
    unit: str
    line: int
    raw: str

    def __repr__(self):
        return f"TechValue({self.value:g} {self.unit} @ line {self.line})"


@dataclass
class TechConstants:
    """The `extract` section of a magic tech file, one corner of it.

    Keys are the tech file's own type-set names -- `allm1`, `allli`, `xhrpoly`, `mcon`,
    `m2c` -- not friendly aliases, so that a lookup failure points at a real grep target.
    `LAYER_TYPESET` maps the GDS-level names this module draws with onto them.
    """
    path: Path
    variant: str
    sheet_res: dict = field(default_factory=dict)     # type-set -> mOhm/square
    contact_res: dict = field(default_factory=dict)   # contact name -> mOhm/contact
    area_cap: dict = field(default_factory=dict)      # type-set -> aF/um^2
    perim_cap: dict = field(default_factory=dict)     # type-set -> aF/um
    side_cap: dict = field(default_factory=dict)      # type-set -> aF/um, coplanar
    side_factor: dict = field(default_factory=dict)   # type-set -> the 2nd sidewall arg
    overlap_cap: dict = field(default_factory=dict)   # (above, below) -> aF/um^2
    sideoverlap_cap: dict = field(default_factory=dict)   # (edge, plane) -> aF/um

    @classmethod
    def parse(cls, path: Path = None, variant: str = DEFAULT_VARIANT) -> "TechConstants":
        """Walk the tech file tracking which `variants` block we are inside.

        A `variants` line is a corner *selector*: `variants (hrhc),(hrlc)` means every
        line until the next `variants` applies to those two corners only.  So the parse
        is a state machine over that, and lines are kept only when `variant` is one of
        the selected corners.  `variants *` blocks are kept too -- they hold the
        corner-independent leftovers -- but in this file they carry no numeric R or C.
        """
        path = Path(path or default_tech_file())
        tc = cls(path=path, variant=variant)
        selected = False
        for n, raw in enumerate(path.read_text().splitlines(), start=1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("variants"):
                sel = line.split(None, 1)[1].strip()
                selected = sel == "*" or variant in [s.strip() for s in sel.split(",")]
                continue
            if not selected:
                continue
            tok = line.split()
            if tok[0] == "resist" and len(tok) >= 3:
                # `resist (allm1)/metal1 125` and `resist xhrpoly/active 319800 0.5`:
                # the type-set is everything left of the plane, parens are cosmetic.
                types = tok[1].split("/")[0].strip("()")
                if _is_number(tok[2]):
                    for t in types.split(","):
                        tc.sheet_res[t.strip("*")] = TechValue(
                            float(tok[2]), "mOhm/square", n, raw.rstrip())
            elif tok[0] == "contact" and len(tok) == 3 and _is_number(tok[2]):
                for name in tok[1].split(","):
                    tc.contact_res[name] = TechValue(
                        float(tok[2]), "mOhm/contact", n, raw.rstrip())
            # The 4-token forms are `<kw> <types> <plane> <value>`.  Longer ones are the
            # `<types> <plane> <shieldtypes> <shieldplane> <value>` overlap variants,
            # which are cap to *another layer*, not to substrate -- skipped on purpose.
            elif tok[0] == "defaultareacap" and len(tok) == 4:
                tc.area_cap[tok[1]] = TechValue(float(tok[3]), "aF/um^2", n, raw.rstrip())
            elif tok[0] == "defaultperimeter" and len(tok) == 4:
                tc.perim_cap[tok[1]] = TechValue(float(tok[3]), "aF/um", n, raw.rstrip())
            elif tok[0] == "defaultsidewall" and len(tok) in (4, 5):
                tc.side_cap[tok[1]] = TechValue(float(tok[3]), "aF/um", n, raw.rstrip())
                if len(tok) == 5:
                    tc.side_factor[tok[1]] = TechValue(float(tok[4]), "1", n, raw.rstrip())
            # `<kw> <typesA> <planeA> <typesB> <planeB> <value>`: cap between two *different*
            # layers.  In a crossbar the m2-over-m1 term is row-to-column coupling at every
            # crossing, i.e. input straight through to output, so it is not a curiosity.
            elif tok[0] == "defaultoverlap" and len(tok) == 6:
                tc.overlap_cap[(tok[1], tok[3])] = TechValue(
                    float(tok[5]), "aF/um^2", n, raw.rstrip())
            elif tok[0] == "defaultsideoverlap" and len(tok) == 6:
                tc.sideoverlap_cap[(tok[1], tok[3])] = TechValue(
                    float(tok[5]), "aF/um", n, raw.rstrip())
        tc.check()
        return tc

    def check(self):
        """Assert the parse reproduces known sky130 typical-corner values.

        Cheap, and it is the only thing standing between a corner-selection bug and every
        resistance in docs/LAYOUT.md being 16% or 20% wrong with no other symptom.
        """
        for k, want in SHEET_RES_REFERENCE.items():
            got = self.sheet_res.get(k)
            if got is None or got.value != want:
                raise ValueError(
                    f"{self.path}: sheet resistance for {k!r} is {got}, expected {want} "
                    f"mOhm/square in the {self.variant!r} corner. Either the corner "
                    f"selection is wrong or the tech file changed. {CORNER_NOTE}")
        for kind, (key, want) in CAP_REFERENCE.items():
            got = (self.area_cap if kind == "area" else self.perim_cap).get(key)
            if got is None or got.value != want:
                raise ValueError(f"{self.path}: {kind} cap for {key!r} is {got}, "
                                 f"expected {want}")

    # Accessors that do the unit conversion once, so no caller divides by 1000 by hand.
    def r_sheet(self, layer: str) -> float:
        """Ohm per square for a drawn layer name (`met1`, `li1`, ...)."""
        return self.sheet_res[LAYER_TYPESET[layer]].value / 1e3

    def r_contact(self, cut: str) -> float:
        """Ohm per cut for a via/contact layer name (`via`, `mcon`, ...)."""
        return self.contact_res[CUT_CONTACT[cut]].value / 1e3

    def c_area(self, layer: str) -> float:
        """F per um^2 of drawn area, to substrate."""
        return self.area_cap[LAYER_TYPESET[layer]].value * 1e-18

    def c_perim(self, layer: str) -> float:
        """F per um of drawn perimeter: the fringe term, to substrate."""
        return self.perim_cap[LAYER_TYPESET[layer]].value * 1e-18

    def c_side(self, layer: str) -> float:
        """F per um of facing edge between two coplanar shapes on the same layer.

        This is the rail-to-rail coupling constant.  Nothing in `crossbar.Crossbar` uses
        it -- the generator only has shunt C to `vss` -- which is exactly why
        `array_layout.coupling_report` exists.
        """
        return self.side_cap[LAYER_TYPESET[layer]].value * 1e-18

    def c_overlap(self, above: str, below: str) -> float:
        """F per um^2 of overlap between two layers -- the plate cap where they cross."""
        return self.overlap_cap[(LAYER_TYPESET[above], LAYER_TYPESET[below])].value * 1e-18

    def c_sideoverlap(self, edge: str, plane: str) -> float:
        """F per um of `edge`'s perimeter fringing onto `plane`'s layer."""
        return self.sideoverlap_cap[(LAYER_TYPESET[edge],
                                     LAYER_TYPESET[plane])].value * 1e-18

    def provenance(self, *keys) -> str:
        """`file:line` for each requested constant, for pasting into a report.

        A key is a type-set name for the per-layer constants, or a `(above, below)` tuple
        for the two cross-layer ones.
        """
        out = []
        for k in keys:
            for name, d in (("resist", self.sheet_res), ("contact", self.contact_res),
                            ("areacap", self.area_cap), ("perimeter", self.perim_cap),
                            ("sidewall", self.side_cap), ("overlap", self.overlap_cap),
                            ("sideoverlap", self.sideoverlap_cap)):
                if k in d:
                    label = k if isinstance(k, str) else " over ".join(k)
                    out.append(f"{name} {label} = {d[k].value:g} {d[k].unit} "
                               f"({self.path.name}:{d[k].line})")
        return "\n".join(out)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def default_tech_file() -> Path:
    """`sky130A.tech` under the same PDK root `crossbar` already finds for SPICE models."""
    p = C.sky130_root() / "libs.tech/magic/sky130A.tech"
    if not p.is_file():
        raise RuntimeError(
            f"magic tech file not found at {p}. It ships with open_pdks/volare sky130A "
            "and is needed for sheet resistance and capacitance constants -- magic "
            "itself is not required.")
    return p


def default_drc_deck() -> Path:
    return C.sky130_root() / "libs.tech/klayout/drc/sky130A.lydrc"


def default_lvs_deck() -> Path:
    """The real LVS ruleset.

    Not `sky130.lylvs`, which is a 33-line KLayout macro wrapper whose only content is a
    *commented-out* `# %include sky130.lvs` -- running it checks nothing at all.
    """
    return C.sky130_root() / "libs.tech/klayout/lvs/sky130.lvs"


# --------------------------------------------------------------------------
# Layers.  Numbers come from the PDK's own map file; the table below is only the
# mapping from drawn-layer name to the tech file's type-set name.
# --------------------------------------------------------------------------
LAYER_TYPESET = {"li1": "allli", "met1": "allm1", "met2": "allm2",
                 "met3": "allm3", "met4": "allm4", "met5": "allm5"}
CUT_CONTACT = {"mcon": "mcon", "via": "m2c", "via2": "m3c"}


def layer_map(path: Path = None) -> dict:
    """Parse `sky130A.map` -> {name: (layer, datatype)} for drawing/via purposes.

    The map is the PDK's LEF/DEF purpose table: `met1 LEFPIN,NET,SPNET,PIN,VIA 68 20`.
    A routing layer's drawing purpose is the row carrying `NET`; a cut layer has no such
    row and its row carries `VIA`.  Reading it beats hardcoding 68/20 -- and it is the
    same numbering the DRC deck's `m1_wildcard = "68/0-4,6-43,45-*"` selects.
    """
    path = Path(path or C.sky130_root() / "libs.tech/klayout/tech/sky130A.map")
    net, via = {}, {}
    for line in path.read_text().splitlines():
        tok = line.split()
        if len(tok) != 4 or not tok[2].isdigit() or not tok[3].isdigit():
            continue
        name, purposes, ld = tok[0], tok[1].split(","), (int(tok[2]), int(tok[3]))
        if "NET" in purposes:
            net.setdefault(name, ld)
        elif "VIA" in purposes:
            via.setdefault(name, ld)
    return {**via, **net}


class Layers:
    """GDS layer/datatype for the names this module draws, checked against the PDK map."""

    EXPECTED = {"li1": (67, 20), "met1": (68, 20), "met2": (69, 20),
                "mcon": (67, 44), "via": (68, 44), "via2": (69, 44)}

    def __init__(self, path: Path = None):
        self.map_path = Path(path or C.sky130_root() / "libs.tech/klayout/tech/sky130A.map")
        pdk = layer_map(self.map_path)
        self.ld = {}
        for name, want in self.EXPECTED.items():
            got = pdk.get(name)
            if got != want:
                raise ValueError(
                    f"{self.map_path}: layer {name!r} is {got}, this module assumes "
                    f"{want}. Fix Layers.EXPECTED rather than the assumption.")
            self.ld[name] = got

    def __getitem__(self, name):
        return self.ld[name]


# --------------------------------------------------------------------------
# Geometry: draw it, write it, read it back, measure it.  Measurement always goes
# through the written file, so what gets measured is what a DRC deck would see.
# --------------------------------------------------------------------------
def um(x: float) -> int:
    """Microns -> database units, on grid."""
    return int(round(x / DBU))


def rect(x0: float, y0: float, x1: float, y1: float) -> kdb.Box:
    return kdb.Box(um(x0), um(y0), um(x1), um(y1))


def new_layout(cellname: str) -> tuple:
    ly = kdb.Layout()
    ly.dbu = DBU
    return ly, ly.create_cell(cellname)


def wire_gds(path: Path, w: float, length: float, layer: str = "met1",
             cellname: str = "WIRE", layers: Layers = None) -> Path:
    """One rectangular wire, `w` x `length` um, on `layer`. The Stage-1 test structure."""
    layers = layers or Layers()
    ly, top = new_layout(cellname)
    top.shapes(ly.layer(*layers[layer])).insert(rect(0.0, 0.0, length, w))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


@dataclass(frozen=True)
class Geometry:
    """What a GDS says about one layer of one cell, in microns."""
    layer: str
    area: float        # um^2
    perimeter: float   # um
    n_polys: int
    boxes: tuple       # ((w, l), ...) per polygon, short side first, um

    @property
    def squares(self) -> float:
        """Sum of L/W over the polygons.

        Only defined for rectangles, which is what `measure` enforces.  A serpentine is
        *not* this: its corners are worth roughly 0.5 square each rather than the ~1 that
        the bounding-box quotient charges them, so `serpentine_squares` handles that case
        separately and this property refuses the geometry.
        """
        return sum(l / w for w, l in self.boxes)


def measure(gds: Path, layer: str, layers: Layers = None,
            require_boxes: bool = True) -> Geometry:
    """Read `gds` back and measure one layer: area, perimeter, per-polygon box sizes.

    `require_boxes` is on by default because `Geometry.squares` is only meaningful for
    rectangles; turn it off to get area and perimeter of arbitrary shapes.
    """
    layers = layers or Layers()
    ly = kdb.Layout()
    ly.read(str(gds))
    idx = ly.layer(*layers[layer])
    reg = kdb.Region(ly.top_cell().begin_shapes_rec(idx))
    reg.merge()
    boxes = []
    for p in reg.each():
        if require_boxes and not p.is_box():
            raise ValueError(f"{gds}: a polygon on {layer} is not a rectangle, so its "
                             "square count is undefined; pass require_boxes=False for "
                             "area and perimeter only")
        b = p.bbox()
        d = sorted((b.width() * ly.dbu, b.height() * ly.dbu))
        boxes.append((d[0], d[1]))
    return Geometry(layer=layer, area=reg.area() * ly.dbu ** 2,
                    perimeter=reg.perimeter() * ly.dbu, n_polys=reg.size(),
                    boxes=tuple(boxes))


# --------------------------------------------------------------------------
# Geometry + constants -> R and C.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Parasitics:
    r: float           # ohm
    c: float           # F
    c_area: float      # F, the area term alone
    c_fringe: float    # F, the perimeter term alone
    squares: float
    geom: Geometry

    def __str__(self):
        return (f"{self.geom.layer}: {self.squares:.3f} squares -> R = {self.r:.4g} ohm; "
                f"C = {self.c / 1e-18:.1f} aF ({self.c_area / 1e-18:.1f} area + "
                f"{self.c_fringe / 1e-18:.1f} fringe)")


def parasitics(geom: Geometry, tech: TechConstants, squares: float = None) -> Parasitics:
    """R = Rs * squares, C = Ca * area + Cp * perimeter. Pass `squares` for non-rectangles.

    The C here is to substrate only.  Coupling to a neighbouring rail on the same layer is
    a separate constant (`TechConstants.c_side`) and a separate, larger, story.
    """
    sq = geom.squares if squares is None else squares
    ca = tech.c_area(geom.layer) * geom.area
    cp = tech.c_perim(geom.layer) * geom.perimeter
    return Parasitics(r=tech.r_sheet(geom.layer) * sq, c=ca + cp, c_area=ca,
                      c_fringe=cp, squares=sq, geom=geom)


def closed_form(w: float, length: float, layer: str, tech: TechConstants) -> Parasitics:
    """The same numbers from arithmetic on `w` and `length`, touching no GDS.

    The point of having both is that `hand_check` compares them: if the GDS round trip,
    the layer lookup, or the dbu scaling is wrong, the two disagree.
    """
    geom = Geometry(layer=layer, area=w * length, perimeter=2 * (w + length),
                    n_polys=1, boxes=((min(w, length), max(w, length)),))
    return parasitics(geom, tech, squares=length / w)


# --------------------------------------------------------------------------
# DRC.  Shelling out to the KLayout binary rather than driving the deck in-process:
# the deck is a KLayout DSL macro, and `-b -r` is how the PDK documents running it.
# --------------------------------------------------------------------------
@dataclass
class DrcResult:
    gds: Path
    report: Path
    clean: bool
    n_violations: int
    by_rule: dict          # rule name -> count
    n_rules_checked: int
    seconds: float
    returncode: int

    def __str__(self):
        if self.clean:
            return (f"DRC CLEAN: {self.gds.name}, {self.n_rules_checked} rules, "
                    f"{self.seconds:.2f} s")
        rules = ", ".join(f"{k} x{v}" for k, v in sorted(self.by_rule.items()))
        return (f"DRC FAIL: {self.gds.name}, {self.n_violations} violations "
                f"[{rules}], {self.seconds:.2f} s")


def run_drc(gds: Path, rundir: Path = None, deck: Path = None,
            klayout: Path = None, timeout: float = 900.0) -> DrcResult:
    """Run the sky130A KLayout DRC deck on `gds` and parse the report database.

    The stock deck has `FEOL = false`, so what actually runs is the back-end-of-line set
    plus the off-grid/angle checks -- every metal, via and local-interconnect rule, and
    no diffusion or poly rule.  That covers the interconnect skeleton this project draws
    completely, and covers a transistor barely at all; `nfet_gds` says so again on site.
    """
    gds = Path(gds)
    deck = Path(deck or default_drc_deck())
    klayout = Path(klayout or KLAYOUT_BIN)
    if not klayout.is_file():
        raise RuntimeError(f"KLayout binary not found at {klayout}. The pip `klayout` "
                           "module cannot run a .lydrc deck; the application can.")
    rundir = Path(rundir or gds.parent)
    rundir.mkdir(parents=True, exist_ok=True)
    report = rundir / f"{gds.stem}.lyrdb"

    cmd = [str(klayout), "-b", "-zz", "-r", str(deck),
           "-rd", f"input={gds}", "-rd", f"report={report}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if not report.is_file():
        raise RuntimeError(f"DRC produced no report database.\ncmd: {' '.join(cmd)}\n"
                           f"stdout tail:\n{proc.stdout[-2000:]}\n"
                           f"stderr tail:\n{proc.stderr[-2000:]}")

    root = ET.parse(report).getroot()
    items = root.find("items")
    counts = Counter((i.findtext("category") or "").strip().strip("'")
                     for i in (items if items is not None else []))
    cats = root.find("categories")
    return DrcResult(gds=gds, report=report, clean=not counts,
                     n_violations=sum(counts.values()), by_rule=dict(counts),
                     n_rules_checked=len(list(cats)) if cats is not None else 0,
                     seconds=elapsed, returncode=proc.returncode)


def drc_rule_value(rule: str, deck: Path = None) -> float:
    """The dimension a named DRC rule checks, e.g. `drc_rule_value("m1.1")` -> 0.14.

    Pulled off the deck line that emits the rule rather than typed in, so a claim like
    "min metal1 width is 0.14 um" in docs/LAYOUT.md has a `file:line` behind it the same
    way the parasitic constants do.  Takes the last metric argument on the line.

    Last resort is the rule's own description text.  The two-opposite-edges enclosure rules
    (`m2.5`, `via1.5a`) build their threshold on an earlier line through a temporary, so a
    single-line parse of the check cannot see it; their descriptions state it.
    """
    deck = Path(deck or default_drc_deck())
    for line in deck.read_text().splitlines():
        if f'.output("{rule}"' not in line:
            continue
        args = re.findall(r"(\d+(?:\.\d+)?)\s*,\s*(?:euclidian|projection)\s*\)", line)
        if not args:
            args = re.findall(r"\.with_area\(\s*\d+(?:\.\d+)?\s*\.\.\s*(\d+(?:\.\d+)?)", line)
        if not args:
            args = re.findall(r"\.(?:width|space|separation|isolated|enclosing|ongrid)"
                              r"\([^,)]*,?\s*(\d+(?:\.\d+)?)", line)
        if not args:
            args = re.findall(r":\s*(\d+(?:\.\d+)?)um", line)
        if args:
            return float(args[-1])
        raise ValueError(f"{deck.name}: rule {rule!r} found but no metric argument parsed "
                         f"from: {line.strip()[:120]}")
    raise KeyError(f"{deck.name}: no rule named {rule!r}")


def pdk_resistor_widths(device: str = "res_xhigh_po") -> list:
    """Drawn widths the PDK actually ships a binned model for, in um, ascending.

    Read off the model filenames -- `sky130_fd_pr__res_xhigh_po_0p35.model.spice` is the
    0.35 um bin.  The DRC minimum (`poly.3`) is narrower than the narrowest binned device,
    and a resistor with no model is not a resistor you can simulate, so the *device* list
    is the honest constraint on a `g_min` serpentine.
    """
    d = C.sky130_root() / "libs.ref/sky130_fd_pr/spice"
    widths = set()
    for p in d.glob(f"sky130_fd_pr__{device}_*.model.spice"):
        m = re.fullmatch(rf"sky130_fd_pr__{device}_(\d+)p(\d+)", p.name.split(".model")[0])
        if m:
            widths.add(float(f"{m.group(1)}.{m.group(2)}"))
    if not widths:
        raise RuntimeError(f"no binned {device} models under {d}")
    return sorted(widths)


def bad_wire_gds(path: Path, layers: Layers = None) -> Path:
    """Deliberately illegal metal1: 0.10 um wide (m1.1 wants 0.14) at 0.10 um spacing
    (m1.2 wants 0.14).

    A DRC harness that has only ever run on clean layouts has not been tested -- a broken
    report parser and a clean layout look identical.  `main` runs this and asserts the
    harness comes back dirty with exactly these two rules.
    """
    layers = layers or Layers()
    ly, top = new_layout("BADWIRE")
    m1 = ly.layer(*layers["met1"])
    top.shapes(m1).insert(rect(0.0, 0.00, 10.0, 0.10))
    top.shapes(m1).insert(rect(0.0, 0.20, 10.0, 0.50))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


# --------------------------------------------------------------------------
# LVS.  Timeboxed and reported honestly -- see docs/LAYOUT.md.
# --------------------------------------------------------------------------
# Front-end layers, needed only for the LVS test structure.  Not in `Layers`, which is
# checked against the map file: the map is a LEF/DEF routing map and has no diff or poly
# rows, so these come from the DRC deck's own `input()`/`polygons()` calls instead.
FEOL_LAYERS = {"diff": (65, 20), "tap": (65, 44), "poly": (66, 20), "licon": (66, 44),
               "nsdm": (93, 44), "psdm": (94, 20)}

# Text layers the LVS deck reads to name nets: `connect(li_con, li_label)` at
# sky130.lvs:1849 and `connect(poly_con, poly_label)` at :1848.  Without labels the
# layout's terminals are anonymous and the extracted subcircuit has no pins to pair with
# the schematic's -- which is the first thing the LVS attempt fell over on.
LABEL_LAYERS = {"li1_label": (67, 5), "poly_label": (66, 5), "met1_label": (68, 5)}

# The deck's global substrate net name, overridable with `-rd lvs_sub=...`.  The nfet's
# body terminal is the substrate, so the schematic port has to carry the same name.
LVS_SUBSTRATE = "b"


def _feol_layers_from_drc(deck: Path = None) -> dict:
    """Cross-check `FEOL_LAYERS` against the `polygons(l, d)` calls in the DRC deck."""
    text = Path(deck or default_drc_deck()).read_text()
    found = {}
    for name, l, d in re.findall(
            r"^(\w+)\s*=\s*(?:polygons|input)\((\d+),\s*(\d+)\)", text, re.M):
        found.setdefault(name, (int(l), int(d)))
    bad = {k: (v, found.get(k)) for k, v in FEOL_LAYERS.items() if found.get(k) != v}
    if bad:
        raise ValueError(f"FEOL layer numbers disagree with {deck or default_drc_deck()}: "
                         f"{bad}")
    return found


NFET_CELL = "nfet_ref"   # must equal the netlist's subckt name: the LVS deck pairs the
                         # layout top cell with the schematic top cell by name, and says
                         # "Can't find a schematic counterpart for the top cell" otherwise.


def nfet_gds(path: Path, w: float = 1.0, l: float = 0.15, layers: Layers = None,
             cellname: str = NFET_CELL) -> Path:
    """A hand-drawn `sky130_fd_pr__nfet_01v8`-shaped device, for the LVS attempt.

    Minimal recognizable nfet: an n+ diffusion island, a poly gate crossing it with
    endcaps, source/drain licon contacts up to li1, and a p+ tap to the substrate so the
    body terminal is not floating.  Dimensions follow the sky130 periphery rules
    (poly.7 endcap 0.13, licon.5 diff enclosure 0.06, diff/tap 0.125 implant enclosure),
    but note that the stock DRC deck runs with `FEOL = false`, so almost none of the rules
    that govern this geometry are actually checked -- which is one of the reasons the LVS
    attempt is reported as an attempt.
    """
    layers = layers or Layers()
    _feol_layers_from_drc()
    ly, top = new_layout(cellname)
    lay = {k: ly.layer(*v) for k, v in {**FEOL_LAYERS, **LABEL_LAYERS}.items()}
    lay["li1"] = ly.layer(*layers["li1"])

    sd = 0.29                     # source/drain diff extension past the gate
    gx0, gx1 = sd, sd + l         # gate in x
    dx1 = 2 * sd + l              # diff extent in x
    top.shapes(lay["diff"]).insert(rect(0.0, 0.0, dx1, w))
    top.shapes(lay["poly"]).insert(rect(gx0, -0.13, gx1, w + 0.13))
    top.shapes(lay["nsdm"]).insert(rect(-0.125, -0.125, dx1 + 0.125, w + 0.125))

    for name, x0 in (("s", 0.06), ("d", gx1 + 0.06)):   # licon.5: 0.06 diff enclosure
        top.shapes(lay["licon"]).insert(rect(x0, w / 2 - 0.085, x0 + 0.17, w / 2 + 0.085))
        top.shapes(lay["li1"]).insert(rect(x0 - 0.08, w / 2 - 0.115,
                                           x0 + 0.25, w / 2 + 0.115))
        top.shapes(lay["li1_label"]).insert(
            kdb.Text(name, um(x0 + 0.085), um(w / 2)))
    top.shapes(lay["poly_label"]).insert(
        kdb.Text("g", um((gx0 + gx1) / 2), um(w + 0.06)))

    ty0 = w + 0.5                                       # substrate tap, p+ to pwell
    top.shapes(lay["tap"]).insert(rect(0.0, ty0, 0.6, ty0 + 0.6))
    top.shapes(lay["psdm"]).insert(rect(-0.125, ty0 - 0.125, 0.725, ty0 + 0.725))
    top.shapes(lay["licon"]).insert(rect(0.215, ty0 + 0.215, 0.385, ty0 + 0.385))
    top.shapes(lay["li1"]).insert(rect(0.135, ty0 + 0.185, 0.465, ty0 + 0.415))
    top.shapes(lay["li1_label"]).insert(
        kdb.Text(LVS_SUBSTRATE, um(0.3), um(ty0 + 0.3)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


def hdl21_nfet_netlist(path: Path, w: float = 1.0, l: float = 0.15,
                       cellname: str = NFET_CELL) -> Path:
    """SPICE for a bare sky130 nfet, through the same hdl21 path `crossbar.Cell` uses.

    `Cell` itself cannot be the LVS reference: it contains a memristor, and sky130 has no
    resistive-memory device to match it against.  What *can* be matched is the access
    device, so this builds a one-instance module out of the identical `h.Nmos` call --
    same `si_literal` treatment, same `sky130_hdl21.compile` -- and netlists it.

    `w` and `l` are microns here, to line up with `nfet_gds`; `si_literal` wants SI metres
    for the reason its own docstring gives, hence the `* µ`.
    """
    import hdl21 as h
    import sky130_hdl21
    from hdl21.prefix import µ

    m = h.Module(name=cellname)
    m.d, m.g, m.s = h.Port(), h.Port(), h.Port()
    body = m.add(h.Port(), name=LVS_SUBSTRATE)   # named to match the deck's global net
    m.add(h.Nmos(w=C.si_literal(w * µ), l=C.si_literal(l * µ),
                 family=h.MosFamily.CORE)(d=m.d, g=m.g, s=m.s, b=body), name="macc")
    C.sky130_install()
    sky130_hdl21.compile(m)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        h.netlist(h.to_proto([m]), f, fmt="spice")
    return path


LVS_PASS = "Congratulations! Netlists match."   # sky130.lvs:2694
LVS_FAIL = "Netlists don't match"               # sky130.lvs:2689


def lvs_netlist_units(src: Path, dst: Path) -> Path:
    """Rescale `w` and `l` from microns to metres, the one edit LVS needs to converge.

    The two tools disagree about units and neither is wrong.  `sky130_hdl21` multiplies
    device sizes by 1e6 on the way out, because the SkyWater ngspice models are used with
    `.option scale 1E6` and therefore want microns -- that scaling is the whole reason
    `crossbar.si_literal` exists.  KLayout's SPICE reader reads a MOS `W`/`L` as SI metres
    and stores microns, so it multiplies by 1e6 again: a 1 um device arrives as a 1 m one
    and LVS reports a parameter mismatch.  Undoing exactly that one factor is the adapter.

    Only `w` and `l` are touched, because they are the only MOS4 parameters KLayout's
    comparer treats as primary.  `as`/`ad`/`ps`/`pd` come out of the adapted netlist
    nonsensical (their expressions reference the rescaled `w`) and LVS matches anyway --
    which is itself a limitation worth knowing: **the layout's source/drain area and
    perimeter are compared against nothing.**
    """
    text = Path(src).read_text()
    out = re.sub(r"\b([wl])='([^']*)'",
                 lambda m: f"{m.group(1)}='({m.group(2)})*1e-6'", text, flags=re.I)
    dst = Path(dst)
    dst.write_text(out)
    return dst


@dataclass
class LvsResult:
    gds: Path
    netlist: Path
    ok: bool
    summary: str
    seconds: float
    returncode: int
    report: Path
    extracted: Path
    stdout_tail: str

    def __str__(self):
        return (f"LVS {'MATCH' if self.ok else 'NO MATCH'}: {self.summary} "
                f"({self.seconds:.2f} s, rc={self.returncode})")


def run_lvs(gds: Path, netlist: Path, rundir: Path = None, deck: Path = None,
            klayout: Path = None, timeout: float = 900.0) -> LvsResult:
    """Run the sky130 KLayout LVS deck, invoked the way the PDK's own run_lvs.py does.

    Returns a result either way; the caller decides whether a non-match is a failure.
    Success is read off the deck's own stdout verdict rather than the return code, because
    `klayout -b` exits 0 on a clean run of a deck that reported a mismatch.
    """
    gds, netlist = Path(gds), Path(netlist)
    deck = Path(deck or default_lvs_deck())
    klayout = Path(klayout or KLAYOUT_BIN)
    rundir = Path(rundir or gds.parent)
    rundir.mkdir(parents=True, exist_ok=True)
    report = rundir / f"{gds.stem}.lvsdb"
    extracted = rundir / f"{gds.stem}_extracted.cir"

    # `convert_subckts=true` puts the deck's reader delegate (sky130.lvs:349, :416) on the
    # `X` calls hdl21 emits for `sky130_fd_pr__nfet_01v8`, turning them into MOS devices.
    # Without it the schematic keeps an empty subcircuit with no layout counterpart, the
    # deck flattens it away, and the comparison is one transistor against nothing.
    cmd = [str(klayout), "-b", "-zz", "-r", str(deck), "-rd", f"input={gds}",
           "-rd", f"report={report}", "-rd", f"schematic={netlist}",
           "-rd", f"target_netlist={extracted}", "-rd", "thr=4",
           "-rd", f"lvs_sub={LVS_SUBSTRATE}", "-rd", "convert_subckts=true"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, rc = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return LvsResult(gds, netlist, False, f"timed out after {timeout:g} s", timeout,
                         -1, report, extracted, "")
    elapsed = time.perf_counter() - t0

    # The deck's own verdict, verbatim from sky130.lvs:2689 and :2694.  Do not soften this
    # into a keyword search: "ERROR : Netlists don't match" contains the word "match".
    ok = LVS_PASS in out and LVS_FAIL not in out
    verdict = [ln.strip() for ln in out.splitlines()
               if LVS_PASS in ln or LVS_FAIL in ln or re.search(r"ERROR", ln)]
    return LvsResult(gds=gds, netlist=netlist, ok=ok,
                     summary="; ".join(verdict[-4:]) or "no verdict line in output",
                     seconds=elapsed, returncode=rc, report=report, extracted=extracted,
                     stdout_tail=out[-4000:])


# --------------------------------------------------------------------------
# Stage-1 self test: everything above, on a wire small enough to check by hand.
# --------------------------------------------------------------------------
HAND_CHECK_W, HAND_CHECK_L = 0.5, 10.0


def hand_check(tech: TechConstants, rundir: Path, w: float = HAND_CHECK_W,
               length: float = HAND_CHECK_L, layer: str = "met1") -> tuple:
    """Pipeline vs. closed form on one metal1 wire. Returns (from_gds, from_arithmetic).

    Raises if they disagree beyond float noise.  The independent side does its arithmetic
    on `w` and `length` directly, so agreement exercises the GDS write, the read back, the
    dbu scaling, the layer lookup and the constant lookup all at once.
    """
    gds = wire_gds(Path(rundir) / f"wire_{layer}.gds", w, length, layer=layer)
    got = parasitics(measure(gds, layer), tech)
    want = closed_form(w, length, layer, tech)
    for name, a, b in (("R", got.r, want.r), ("C", got.c, want.c),
                       ("squares", got.squares, want.squares),
                       ("area", got.geom.area, want.geom.area),
                       ("perimeter", got.geom.perimeter, want.geom.perimeter)):
        if abs(a - b) > 1e-12 * max(abs(a), abs(b), 1e-30):
            raise AssertionError(f"{name} from GDS {a!r} != closed form {b!r}")
    return got, want


def main(rundir: Path = Path("/tmp/layout_oracle")):
    t_start = time.perf_counter()
    rundir = Path(rundir)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)

    print("=" * 78)
    print("STAGE 1  layout -> parasitics oracle, on a hand-checkable structure")
    print("=" * 78)

    tech = TechConstants.parse()
    layers = Layers()
    print(f"\ntech file : {tech.path}")
    print(f"corner    : {CORNER_NOTE}")
    print(f"layer map : {layers.map_path}")
    print("\nconstants, as parsed (value, unit, file:line):")
    print(tech.provenance("allm1", "allm2", "allm3", "allm4", "allm5", "allli",
                          "mrp1", "xhrpoly", "mcon", "m2c", "m3c"))
    print(f"\n{len(SHEET_RES_REFERENCE)} sheet-resistance and {len(CAP_REFERENCE)} "
          "capacitance reference values asserted OK")
    print("layer/datatype from the PDK map, asserted OK: "
          + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in layers.ld.items()))

    print("\n" + "-" * 78)
    print(f"hand check: metal1 wire {HAND_CHECK_L} um x {HAND_CHECK_W} um")
    print("-" * 78)
    got, want = hand_check(tech, rundir)
    rs, ca, cp = tech.r_sheet("met1"), tech.c_area("met1"), tech.c_perim("met1")
    print(f"  from GDS      {got}")
    print(f"  closed form   {want}")
    print(f"  by hand       {HAND_CHECK_L}/{HAND_CHECK_W} = "
          f"{HAND_CHECK_L / HAND_CHECK_W:g} squares x {rs:g} ohm/sq = "
          f"{HAND_CHECK_L / HAND_CHECK_W * rs:g} ohm")
    print(f"                {HAND_CHECK_L * HAND_CHECK_W:g} um^2 x {ca / 1e-18:g} aF/um^2 "
          f"+ {2 * (HAND_CHECK_L + HAND_CHECK_W):g} um x {cp / 1e-18:g} aF/um = "
          f"{(HAND_CHECK_L * HAND_CHECK_W * ca + 2 * (HAND_CHECK_L + HAND_CHECK_W) * cp) / 1e-18:.2f} aF")
    print("  GDS and closed form agree to float precision")

    print("\n" + "-" * 78)
    print("DRC harness, both directions")
    print("-" * 78)
    clean = run_drc(rundir / "wire_met1.gds", rundir)
    print(f"  legal wire     {clean}")
    bad = run_drc(bad_wire_gds(rundir / "badwire.gds", layers), rundir)
    print(f"  illegal wire   {bad}")
    if not clean.clean:
        raise AssertionError(f"the legal wire should be DRC clean: {clean.by_rule}")
    if set(bad.by_rule) != {"m1.1", "m1.2"}:
        raise AssertionError("the illegal wire should trip exactly m1.1 (min width) and "
                             f"m1.2 (min spacing), got {bad.by_rule}")
    print("  harness proven: it passes a legal wire and flags a planted width and "
          "spacing violation")

    print("\n" + "-" * 78)
    print("LVS attempt: hand-drawn sky130 nfet vs. an hdl21 reference netlist")
    print("-" * 78)
    gds = nfet_gds(rundir / "nfet.gds", layers=layers)
    print(f"  layout   {gds}  (FEOL layers cross-checked against the DRC deck)")
    try:
        net = hdl21_nfet_netlist(rundir / "nfet_ref.spice")
        print(f"  netlist  {net}")
        print("  " + "\n  ".join(net.read_text().strip().splitlines()[-6:]))
    except Exception as e:
        print(f"  netlist  FAILED to build from hdl21: {type(e).__name__}: {e}")
        net = None
    if net is not None:
        raw = run_lvs(gds, net, rundir)
        print(f"  as hdl21 emits it       {raw}")
        si = lvs_netlist_units(net, rundir / "nfet_ref_si.spice")
        fixed = run_lvs(gds, si, rundir)
        print(f"  with the um->m adapter  {fixed}")
        if not fixed.ok:
            print("  --- deck output tail ---")
            print("  " + "\n  ".join(fixed.stdout_tail.strip().splitlines()[-12:]))
        else:
            print(f"  extracted netlist: {fixed.extracted}")
            print("  " + "\n  ".join(fixed.extracted.read_text().strip().splitlines()[-3:]))
            print("  matched on W and L only: KLayout's MOS4 comparer ignores AS/AD/PS/PD,"
                  "\n  so the drawn source/drain area and perimeter are checked against "
                  "nothing.")

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s, artifacts in {rundir}")


if __name__ == "__main__":
    main()
