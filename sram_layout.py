"""The 6T bitcell, drawn: six sky130 devices, DRC with FEOL on, LVS against Stage A.

`sram6t.py` proves the cell holds state.  This draws it, and the result is not a clean cell:
**the open sky130 DRC deck forbids the geometry its own SRAM device models are characterized
at.**  Two rules do it, and neither is negotiable from the layout side:

  * `difftap.1` -- minimum diff/tap width 0.15 um.  The access NMOS and the pull-up PMOS are
    characterized at W = 0.14 um and at no other width, so their channels are 0.01 um below
    the minimum drawable diffusion.
  * `difftap.2` -- minimum gate width 0.42 um, or 0.36 um inside the `areaid.sc` standard-cell
    marker.  All six devices are 0.21 or 0.14 um wide, so every gate in the cell is a
    violation, by a factor of two to three.

That is the concrete form of "foundry bitcells are drawn under waived or bitcell-specific
rules".  `sky130A.lydrc` has cell-name exemptions -- `not_in_cell1` at :264 is one, and it
does gate `difftap.2` -- but they name specific `s8*` macros, not a bitcell, and the open PDK
ships no waiver a third party can invoke.  So this module draws the cell, runs FEOL DRC, and
reports exactly which rules fire and how many times, rather than switching anything off.
`RULE_STORY` is the summary and `main` asserts that the violations are *only* those rules:
a new one appearing is a drawing error and must not hide behind the known two.

LVS has its own honest limit, and it is worth stating before the word MATCH appears anywhere.
`sky130.lvs` has no extraction rule for `special_nfet_latch`, `special_nfet_pass` or
`special_pfet_latch`; grep it and the only `special_*` MOS it knows is `special_nfet_01v8`, in
the *netlist reader*'s model list, not in an `extract_devices` call.  There is no marker layer
that distinguishes them either -- physically these are ordinary 1.8 V transistors, separately
characterized in the bitcell's context.  So the extractor calls them `nfet_01v8` / `pfet_01v8`
and a comparison against the Stage-A netlist verbatim cannot match on device class.
`lvs_variants` runs it both ways and `LVS_COMPARED` says precisely what a match does and does
not establish.

Run it:  uv run sram_layout.py
"""
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import klayout.db as kdb

import cell_layout as CL     # feol_deck(), GRID / SLACK, grid helpers
import layout_oracle as O
import sram6t as S

CELL_NAME = "sram_6t"        # must equal the LVS reference netlist's subckt name
BAD_CELL_NAME = "sram_6t_bad"

# Front-end and marker layers this module draws, keyed by the DRC deck's own variable names
# so `_check_layers` can compare them against the deck's `polygons(l, d)` calls.  The PDK's
# LEF/DEF map has no row for any of these, which is why they cannot come from `O.Layers`.
SRAM_LAYERS = {"diff": (65, 20), "tap": (65, 44), "poly": (66, 20), "licon": (66, 44),
               "nsdm": (93, 44), "psdm": (94, 20), "nwell": (64, 20), "npc": (95, 20)}

# Every DRC dimension the plan below uses, by rule name.  Nothing geometric is typed in
# except `IMPLANT_ENC` and the device sizes, and both say where they come from.
RULES = ("difftap.1", "difftap.2", "difftap.3", "difftap.7", "difftap.8", "difftap.9",
         "difftap.10", "difftap.11", "poly.1a", "poly.2", "poly.4", "poly.5", "poly.6",
         "poly.7", "poly.8", "licon.1", "licon.2", "licon.5", "licon.7", "licon.8",
         "licon.8a", "licon.11", "licon.13", "licon.14", "licon.15", "li.1", "li.3",
         "li.5", "li.6", "npc.1", "npc.2", "npc.4", "nsdm.1", "psdm.1", "nwell.1",
         "ct.1", "ct.2", "m1.1", "m1.2", "m1.4", "m1.5", "m1.6")

# nsdm/psdm enclosure of the diffusion they implant.  sky130's rule set calls this nsdm.5 /
# psdm.5 = 0.125 um; **this DRC deck does not check it** -- it checks implant width only -- so
# unlike everything else here the number cannot be parsed back out of the deck.  Drawn anyway,
# because LVS derives `ngate` as `nsdm.and(tgate)` and an unimplanted gate is not a transistor.
# `layout_oracle.nfet_gds` uses the same 0.125 for the same reason.
IMPLANT_ENC = 0.125

# Rail and column widths.  Wide enough that an mcon inside them satisfies m1.4 all round.
W_RAIL = 0.25
W_COL = 0.25

RULE_STORY = (
    "The two rules that block this cell are consequences of the device widths, not of the "
    "drawing.  `difftap.1` wants diffusion at least 0.15 um wide and the access NMOS and "
    "pull-up PMOS are characterized only at W = 0.14 um.  `difftap.2` wants a gate at least "
    "0.42 um wide (0.36 um inside areaid.sc) and the widest device here is 0.21 um.  Neither "
    "can be fixed by redrawing: the only way out is a wider device, and a wider device has no "
    "model -- ngspice answers a W outside the bin with `could not find a valid modelname`, "
    "which `sram6t.unit_check` demonstrates.  The foundry's own bitcell is drawn at these "
    "widths under a waiver the open PDK does not expose; `not_in_cell1` (sky130A.lydrc:264) "
    "shows the mechanism exists and names `s8*` macros rather than a bitcell."
)

LVS_COMPARED = (
    "What the match establishes, measured by `lvs_probes` rather than read off the deck: the "
    "six devices' graph, the five ports, the device class, and W and L per device -- W is "
    "caught 0.01% high, so the geometry comparison is effectively exact, and a broken "
    "cross-couple or a swapped device class fails.  What it does not establish, also measured: "
    "the source/drain area and perimeter, since a device carrying nonsense AS/AD/PS/PD still "
    "matches -- KLayout's MOS4 comparer ignores them, the same hollow spot docs/LAYOUT.md 5 "
    "records.  And the *model*: sky130.lvs has no `extract_devices` call for any "
    "`special_*fet_*` and no drawn layer distinguishes one from an `nfet_01v8`, so the "
    "extractor necessarily emits `nfet_01v8` / `pfet_01v8` -- which is why the Stage-A netlist "
    "does not match verbatim and the substituted one does.  A match here says the layout is "
    "the right circuit out of the right-sized transistors; it says nothing about the binding "
    "to the SRAM-specific models Stage A actually simulated."
)


def g(v: float) -> float:
    return CL.grid_up(v)


def sram_rules(deck: Path = None) -> dict:
    return {r: O.drc_rule_value(r, deck) for r in RULES}


def _check_layers(deck: Path = None) -> dict:
    """Cross-check `SRAM_LAYERS` against the DRC deck's own layer definitions."""
    found = O._feol_layers_from_drc(deck)
    bad = {k: (v, found.get(k)) for k, v in SRAM_LAYERS.items() if found.get(k) != v}
    if bad:
        raise ValueError(f"sram layer numbers disagree with the DRC deck: {bad}")
    return found


# --------------------------------------------------------------------------
# The plan.  Every y band is placed by a running cursor against the rule that binds it, so
# the numbers below are consequences rather than choices, and moving one rule moves the cell.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Device:
    """One drawn transistor: where its channel is, and which island it belongs to."""
    name: str
    role: str
    w: float           # um, channel width == diff width under the poly
    l: float           # um
    gate_x: float      # um, gate centreline
    y0: float          # um, channel bottom
    y1: float          # um, channel top


@dataclass(frozen=True)
class Island:
    """One diffusion island: a bar of channel width with a taller head at each contact."""
    x0: float
    x1: float
    w: float           # channel width
    y_sd0: float       # head band bottom
    y_sd1: float       # head band top
    gates: tuple       # gate centrelines, um
    licons: tuple      # contact centrelines, um
    nets: tuple        # net name per contact, aligned with `licons`


@dataclass(frozen=True)
class Spec:
    """Every dimension of the drawn cell."""
    rules: dict
    islands: dict           # name -> Island
    devices: tuple          # Device, in netlist order
    head: float             # um, contact-head size, both directions
    neck_ext: float         # um, how far the channel-width neck runs past each gate
    y: dict                 # named y coordinates
    x: dict                 # named x coordinates
    nwell: tuple            # (x0, y0, x1, y1)
    ptap: tuple
    ntap: tuple
    bbox: tuple

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) * (y1 - y0)

    @property
    def size(self) -> tuple:
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0, y1 - y0)

    def __str__(self):
        w, h = self.size
        return (f"{CELL_NAME}: {w:.3f} x {h:.3f} um = {self.area:.2f} um^2, "
                f"{len(self.devices)} devices")


def plan_layout(rules: dict = None, params: S.Sram6TParams = None) -> Spec:
    """Place the whole cell from the rule table.  See the module docstring for the topology.

    Reading order is bottom-up, which is the order the constraints chain in:
        pwell tap / vss rail | wordline poly + its contact | NMOS band |
        li1 cross-couple channel | PMOS band in nwell | nwell tap / vdd rail
    """
    R = rules or sram_rules()
    p = params or S.Sram6TParams()
    SL = CL.SLACK
    lic = R["licon.1"]

    # ---- x: three diffusion islands in the NMOS band, one in the PMOS band ----
    # A device 0.14 or 0.21 um wide cannot hold its own contact: `licon.5` wants 0.04 um of
    # diffusion around a 0.17 um cut, so the diffusion has to be 0.25 um across wherever a
    # contact lands.  Hence the dog-bone -- a neck at the channel width under each gate, a
    # taller head at each contact -- and hence two lengths, which together set the cell width:
    #   `head` (both directions)   the contact head, `lic` plus `licon.5` on each side
    #   `neck_ext`                 how far the neck must run past the gate before it may flare.
    # `neck_ext` is `poly.7`, min source/drain length, and it is the expensive one: a step in
    # the diffusion within 0.25 um of the gate presents an edge facing it, and the deck reports
    # exactly that -- the flare has to be a full source/drain length away, not just clear of
    # `poly.4`.  It is also *why* the drawn cell is over an order of magnitude larger than the
    # foundry's: the foundry contacts these devices without paying either length.
    head = g(lic + 2 * R["licon.5"] + SL)
    neck_ext = g(max(R["poly.7"], R["poly.4"]) + SL)
    ins = head / 2                                            # contact inset from an outer edge
    sd_out = head + neck_ext
    sd_mid = head + 2 * neck_ext                              # shared between two gates
    gap = g(R["difftap.3"] + SL)
    xm = g(IMPLANT_ENC + SL)                                  # left margin for the implants

    acc_w = 2 * sd_out + p.l_acc
    latch_w = 2 * sd_out + sd_mid + 2 * p.l_pd
    xL, xC = xm, xm + acc_w + gap
    xR = xC + latch_w + gap
    x_end = xR + acc_w

    def gates_of(x0, n, l):
        if n == 1:
            return (x0 + sd_out + l / 2,)
        return (x0 + sd_out + l / 2, x0 + sd_out + l + sd_mid + l / 2)

    def licons_of(x0, n):
        w_ = acc_w if n == 1 else latch_w
        if n == 1:
            return (x0 + ins, x0 + w_ - ins)
        return (x0 + ins, CL.grid(x0 + sd_out + p.l_pd + neck_ext + ins), x0 + w_ - ins)

    # ---- y, bottom up ----
    y = {}
    tap_h = g(lic + 2 * R["licon.7"] + SL)
    y["ptap0"] = g(IMPLANT_ENC + SL)
    y["ptap1"] = y["ptap0"] + tap_h
    y["ptap_lic0"] = g(y["ptap0"] + R["licon.7"])
    y["ptap_lic1"] = y["ptap_lic0"] + lic
    y["vss_li0"] = g(y["ptap_lic0"] - CL.SLACK)               # li1 need only cover the cut
    y["vss_li1"] = g(y["ptap_lic1"] + CL.SLACK)
    y["vss_m0"] = g(y["ptap_lic0"] - R["m1.4"])
    y["vss_m1"] = y["vss_m0"] + W_RAIL

    # The wordline: a poly bar across the cell with one widened contact pad.  Its li1 must
    # clear the tap's li1 (li.3) and its met1 pad the vss rail (m1.2); its poly-licon must
    # clear the tap by licon.14, which is the binding one and is easy to miss because the rule
    # reads "poly licon spacing to difftap" and the tap is difftap.
    y["wl_li0"] = g(max(y["vss_li1"] + R["li.3"], y["vss_m1"] + R["m1.2"] - R["m1.4"],
                        y["ptap1"] + R["poly.5"] + R["li.5"],
                        y["ptap1"] + R["licon.14"] - R["li.5"]) + SL)
    y["wl_lic0"] = g(y["wl_li0"] + R["li.5"])
    y["wl_lic1"] = y["wl_lic0"] + lic
    y["wl_li1"] = g(y["wl_lic1"] + R["li.5"])
    y["wl_pad0"], y["wl_pad1"] = y["wl_li0"], y["wl_li1"]     # poly pad, same span as its li1
    y["wl_npc0"] = g(y["wl_lic0"] - R["licon.15"])
    y["wl_npc1"] = g(y["wl_lic1"] + R["licon.15"])
    y["wl_m0"] = g(y["wl_lic0"] - R["m1.4"])
    y["wl_m1"] = y["wl_m0"] + W_RAIL

    # NMOS band.  Held off the wordline contact by licon.14 (poly-licon to difftap), off its
    # npc by npc.4 at the gate and licon.13 at the source/drain cuts -- and, separately, high
    # enough that the wordline *bar* fits between the p+ tap (poly.5) and the bottom endcap of
    # the latch gates (poly.2) while still being poly.1a wide.  That last one is a lower bound
    # on the whole NMOS band that has nothing to do with the contact chain above it.
    bar_floor = g(y["ptap1"] + R["poly.5"] + SL)
    bar_room = bar_floor + R["poly.1a"] + R["poly.2"] + 2 * SL + R["poly.8"]
    y["nsd0"] = g(max(y["wl_lic1"] + R["licon.14"],
                      y["wl_npc1"] + R["npc.4"],
                      y["wl_npc1"] + R["licon.13"] - R["licon.5"],
                      # the wordline's own li1 stub against this band's contact pads (li.3),
                      # which is euclidian and therefore bites diagonally
                      y["wl_li1"] + R["li.3"] + R["li.5"] - (head - lic) / 2,
                      y["ptap1"] + R["poly.6"],
                      bar_room - (head - p.w_pd) / 2) + SL)
    y["nsd1"] = y["nsd0"] + head
    y["n_lic0"] = g(y["nsd0"] + (head - lic) / 2)
    y["n_lic1"] = y["n_lic0"] + lic
    y["n_li0"] = g(y["n_lic0"] - R["li.5"])
    y["n_li1"] = g(y["n_lic1"] + R["li.5"])

    y["pd_ch0"] = g(y["nsd0"] + (head - p.w_pd) / 2)
    y["pd_ch1"] = y["pd_ch0"] + p.w_pd
    y["acc_ch0"] = g(y["nsd0"] + (head - p.w_acc) / 2)
    y["acc_ch1"] = y["acc_ch0"] + p.w_acc
    y["latch_poly0"] = g(y["pd_ch0"] - R["poly.8"])
    y["wl_bar0"] = bar_floor
    y["wl_bar1"] = g(y["latch_poly0"] - R["poly.2"] - SL)
    y["acc_poly1"] = g(y["acc_ch1"] + R["poly.8"] + SL)
    if y["wl_bar1"] - y["wl_bar0"] < R["poly.1a"]:
        raise AssertionError(
            f"the wordline bar came out {y['wl_bar1'] - y['wl_bar0']:.3f} um wide against a "
            f"poly.1a minimum of {R['poly.1a']:g}; the NMOS band is too low for it")

    # The cross-couple channel: two li1 tracks, the lower carrying `q` to the gate of the
    # other inverter and the upper carrying `qb`.  Each holds one poly contact; their npc
    # rectangles are merged into one, because npc.2 wants 0.27 between two of them and the
    # tracks are closer than that -- the deck's own rule text says to merge in that case.
    li_h = g(lic + 2 * R["li.5"] + SL)
    y["ta_li0"] = g(max(y["n_li1"] + R["li.3"], y["nsd1"] + R["licon.14"] - R["li.5"]) + SL)
    y["ta_li1"] = y["ta_li0"] + li_h
    y["ta_lic0"] = g(y["ta_li0"] + R["li.5"])
    y["ta_lic1"] = y["ta_lic0"] + lic
    y["tb_li0"] = g(y["ta_li1"] + R["li.3"] + SL)
    y["tb_li1"] = y["tb_li0"] + li_h
    y["tb_lic0"] = g(y["tb_li0"] + R["li.5"])
    y["tb_lic1"] = y["tb_lic0"] + lic
    y["npc0"] = g(y["ta_lic0"] - R["licon.15"])
    y["npc1"] = g(y["tb_lic1"] + R["licon.15"])

    # PMOS band.  Four separate constraints reach up from the channel; li.3 against the upper
    # track's li1 is the one that actually binds, which is worth knowing when reading the area.
    y["psd0"] = g(max(y["tb_lic1"] + R["licon.14"],
                      y["npc1"] + R["npc.4"] - (head - p.w_pu) / 2,
                      y["npc1"] + R["licon.13"] - (head - lic) / 2,
                      y["tb_li1"] + R["li.3"] + R["li.5"] - (head - lic) / 2) + SL)
    y["psd1"] = y["psd0"] + head
    y["p_lic0"] = g(y["psd0"] + (head - lic) / 2)
    y["p_lic1"] = y["p_lic0"] + lic
    y["p_li0"] = g(y["p_lic0"] - R["li.5"])
    y["p_li1"] = g(y["p_lic1"] + R["li.5"])
    y["pu_ch0"] = g(y["psd0"] + (head - p.w_pu) / 2)
    y["pu_ch1"] = y["pu_ch0"] + p.w_pu
    y["latch_poly1"] = g(y["pu_ch1"] + R["poly.8"] + SL)

    # nwell tap and the vdd rail.  poly.6 holds the tap clear of the pull-up gates, and
    # difftap.3 holds it clear of their diffusion -- `difftap = diff + tap` (sky130A.lydrc:262),
    # so the tap and the p+ source/drain are the same layer set to that rule.
    y["ntap0"] = g(max(y["pu_ch1"] + R["poly.6"], y["latch_poly1"] + R["poly.5"],
                       y["psd1"] + R["difftap.3"],
                       y["p_li1"] + R["li.3"] + R["licon.7"] - CL.SLACK) + SL)
    y["ntap1"] = y["ntap0"] + tap_h
    y["ntap_lic0"] = g(y["ntap0"] + R["licon.7"])
    y["ntap_lic1"] = y["ntap_lic0"] + lic
    y["vdd_li0"] = g(y["ntap_lic0"] - CL.SLACK)
    y["vdd_li1"] = g(y["ntap_lic1"] + CL.SLACK)
    y["vdd_m0"] = g(y["ntap_lic0"] - R["m1.4"])
    y["vdd_m1"] = y["vdd_m0"] + W_RAIL

    # Wells and implants.
    nwell = (g(xC - R["difftap.8"] - SL), g(y["psd0"] - R["difftap.8"] - SL),
             g(xC + latch_w + R["difftap.8"] + SL),
             g(y["ntap1"] + R["difftap.10"] + SL))
    ntap_x = (g(nwell[0] + R["difftap.10"] + SL), g(nwell[2] - R["difftap.10"] - SL))

    x = {"L": xL, "C": xC, "R": xR, "end": x_end, "margin": xm,
         "rail0": g(xm - IMPLANT_ENC + SL), "rail1": g(x_end + IMPLANT_ENC - SL)}
    latch_licons = licons_of(xC, 2)
    x["q"], x["mid"], x["qb"] = latch_licons
    x["wl_pad"] = gates_of(xL, 1, p.l_acc)[0]
    x["poly_qb"], x["poly_q"] = gates_of(xC, 2, p.l_pd)

    islands = {
        "accL": Island(xL, xL + acc_w, p.w_acc, y["nsd0"], y["nsd1"],
                       gates_of(xL, 1, p.l_acc), licons_of(xL, 1), ("bl", "q")),
        "latchN": Island(xC, xC + latch_w, p.w_pd, y["nsd0"], y["nsd1"],
                         gates_of(xC, 2, p.l_pd), latch_licons, ("q", "vss", "qb")),
        "accR": Island(xR, xR + acc_w, p.w_acc, y["nsd0"], y["nsd1"],
                       gates_of(xR, 1, p.l_acc), licons_of(xR, 1), ("qb", "blb")),
        "latchP": Island(xC, xC + latch_w, p.w_pu, y["psd0"], y["psd1"],
                         gates_of(xC, 2, p.l_pu), licons_of(xC, 2), ("q", "vdd", "qb")),
    }
    devices = (
        Device("mnq", "pull-down", p.w_pd, p.l_pd, x["poly_qb"], y["pd_ch0"], y["pd_ch1"]),
        Device("mpq", "pull-up", p.w_pu, p.l_pu, x["poly_qb"], y["pu_ch0"], y["pu_ch1"]),
        Device("mnqb", "pull-down", p.w_pd, p.l_pd, x["poly_q"], y["pd_ch0"], y["pd_ch1"]),
        Device("mpqb", "pull-up", p.w_pu, p.l_pu, x["poly_q"], y["pu_ch0"], y["pu_ch1"]),
        Device("maq", "access", p.w_acc, p.l_acc, islands["accL"].gates[0],
               y["acc_ch0"], y["acc_ch1"]),
        Device("maqb", "access", p.w_acc, p.l_acc, islands["accR"].gates[0],
               y["acc_ch0"], y["acc_ch1"]),
    )
    ptap = (x["rail0"], y["ptap0"], x["rail1"], y["ptap1"])
    ntap = (ntap_x[0], y["ntap0"], ntap_x[1], y["ntap1"])
    # The drawn extent: the widest things are the nsdm over the NMOS band and the two met1
    # rails, and the tallest are the psdm under the p+ tap and the nwell.
    bbox = (min(x["rail0"], g(xL - IMPLANT_ENC)), g(y["ptap0"] - IMPLANT_ENC),
            max(x["rail1"], g(x_end + IMPLANT_ENC)),
            max(nwell[3], g(y["ntap1"] + IMPLANT_ENC)))
    return Spec(rules=R, islands=islands, devices=devices, head=head, neck_ext=neck_ext,
                y=y, x=x, nwell=nwell, ptap=ptap, ntap=ntap, bbox=bbox)


# --------------------------------------------------------------------------
# Drawing.
# --------------------------------------------------------------------------
def _layers(ly, layers: O.Layers) -> dict:
    lay = {k: ly.layer(*v) for k, v in SRAM_LAYERS.items()}
    for name in ("li1", "mcon", "met1"):
        lay[name] = ly.layer(*layers[name])
    for name, ld in O.LABEL_LAYERS.items():
        lay[name] = ly.layer(*ld)
    return lay


def _island(cell, lay, isl: Island, R: dict, neck_l: float, neck_ext: float):
    """One diffusion island: contact heads `head` tall, necks `isl.w` tall under each gate.

    The neck is the transistor.  It runs `neck_ext` past each gate edge before the diffusion
    flares out to a contact head, because a flare closer than that presents a diffusion edge
    facing the gate and `poly.7` reads that as a source/drain shorter than 0.25 um.
    """
    ch0 = CL.grid(isl.y_sd0 + (isl.y_sd1 - isl.y_sd0 - isl.w) / 2)
    ch1 = CL.grid(ch0 + isl.w)
    edges = [isl.x0]
    for gx in isl.gates:
        edges += [CL.grid(gx - neck_l / 2 - neck_ext), CL.grid(gx + neck_l / 2 + neck_ext)]
    edges.append(isl.x1)
    for i in range(0, len(edges) - 1, 2):          # heads
        cell.shapes(lay["diff"]).insert(O.rect(edges[i], isl.y_sd0, edges[i + 1], isl.y_sd1))
    for i in range(1, len(edges) - 1, 2):          # necks
        cell.shapes(lay["diff"]).insert(O.rect(edges[i], ch0, edges[i + 1], ch1))
    return ch0, ch1


def draw_cell(ly, cell, sp: Spec, layers: O.Layers, poly_w: float = None,
              labels: bool = True):
    """The whole cell.  `poly_w` narrows every gate, for the planted-violation harness test."""
    lay = _layers(ly, layers)
    R, y, x = sp.rules, sp.y, sp.x
    lic = R["licon.1"]
    # A contact stub at `li.1` exactly would be 0.17 x 0.33 = 0.0561 um^2, which is `li.6`'s
    # own threshold; two grid steps wider puts it clear of the boundary case.
    li_pad = g(R["li.1"] + 2 * CL.GRID)
    L = sp.devices[0].l
    pw = R["poly.1a"] if poly_w is None else poly_w

    def licon(cx, y0):
        cell.shapes(lay["licon"]).insert(O.rect(cx - lic / 2, y0, cx + lic / 2, y0 + lic))

    def mcon(cx, y0):
        cell.shapes(lay["mcon"]).insert(O.rect(cx - lic / 2, y0, cx + lic / 2, y0 + lic))

    def li_stub(cx, y0, y1):
        cell.shapes(lay["li1"]).insert(O.rect(cx - li_pad / 2, y0, cx + li_pad / 2, y1))

    # ---- diffusion islands, their contacts and their li1 pads ----
    channels = {}
    for name, isl in sp.islands.items():
        channels[name] = _island(cell, lay, isl, R, L, sp.neck_ext)
        for cx in isl.licons:
            licon(cx, y["n_lic0"] if isl.y_sd0 == y["nsd0"] else y["p_lic0"])

    # ---- poly ----
    # The two latch gates run the full height of the cell, gating an NMOS low down and a PMOS
    # high up; that single vertical line *is* the cross-coupling on one side of each inverter.
    for cx in (x["poly_qb"], x["poly_q"]):
        cell.shapes(lay["poly"]).insert(
            O.rect(cx - pw / 2, y["latch_poly0"], cx + pw / 2, y["latch_poly1"]))
    # The wordline: one bar under the NMOS band, a stub up into each access gate, one pad.
    accL, accR = sp.islands["accL"], sp.islands["accR"]
    cell.shapes(lay["poly"]).insert(
        O.rect(accL.gates[0] - pw / 2, y["wl_bar0"], accR.gates[0] + pw / 2, y["wl_bar1"]))
    for cx in (accL.gates[0], accR.gates[0]):
        cell.shapes(lay["poly"]).insert(
            O.rect(cx - pw / 2, y["wl_bar0"], cx + pw / 2, y["acc_poly1"]))
    pad_w = g(lic + 2 * R["licon.8"] + CL.SLACK)
    cell.shapes(lay["poly"]).insert(
        O.rect(x["wl_pad"] - pad_w / 2, y["wl_pad0"], x["wl_pad"] + pad_w / 2, y["wl_pad1"]))
    licon(x["wl_pad"], y["wl_lic0"])
    cell.shapes(lay["npc"]).insert(
        O.rect(x["wl_pad"] - pad_w / 2 - R["licon.15"], y["wl_npc0"],
               x["wl_pad"] + pad_w / 2 + R["licon.15"], y["wl_npc1"]))
    li_stub(x["wl_pad"], y["wl_li0"], y["wl_li1"])
    mcon(x["wl_pad"], y["wl_lic0"])
    # m1.6 is a min *area*, not a width; the pad is centred on its cut, so its half-width has
    # to be on grid too or the off-grid checks fire instead.
    m_wl = 2 * g(max(pad_w, R["m1.6"] / W_RAIL) / 2 + CL.SLACK)
    cell.shapes(lay["met1"]).insert(
        O.rect(x["wl_pad"] - m_wl / 2, y["wl_m0"], x["wl_pad"] + m_wl / 2, y["wl_m1"]))

    # ---- the cross-couple pads on the latch poly, one per channel track ----
    for cx, y0, y1, ylic in ((x["poly_q"], y["ta_li0"], y["ta_li1"], y["ta_lic0"]),
                             (x["poly_qb"], y["tb_li0"], y["tb_li1"], y["tb_lic0"])):
        cell.shapes(lay["poly"]).insert(O.rect(cx - pad_w / 2, y0, cx + pad_w / 2, y1))
        licon(cx, ylic)
    cell.shapes(lay["npc"]).insert(
        O.rect(min(x["poly_q"], x["poly_qb"]) - pad_w / 2 - R["licon.15"], y["npc0"],
               max(x["poly_q"], x["poly_qb"]) + pad_w / 2 + R["licon.15"], y["npc1"]))

    # ---- li1: the NMOS-band straps, the two channel tracks, the head pads ----
    lstub = R["li.5"]
    # `q`: access-L inner head to the latch's left head.  `qb`: mirror on the right.
    cell.shapes(lay["li1"]).insert(
        O.rect(accL.licons[1] - li_pad / 2, y["n_li0"], x["q"] + li_pad / 2, y["n_li1"]))
    cell.shapes(lay["li1"]).insert(
        O.rect(x["qb"] - li_pad / 2, y["n_li0"], accR.licons[0] + li_pad / 2, y["n_li1"]))
    li_stub(accL.licons[0], y["n_li0"], y["n_li1"])       # bl
    li_stub(accR.licons[1], y["n_li0"], y["n_li1"])       # blb
    li_stub(x["mid"], y["n_li0"], y["n_li1"])             # vss
    for cx in sp.islands["latchP"].licons:                # q, vdd, qb heads in the PMOS band
        li_stub(cx, y["p_li0"], y["p_li1"])
    # Track A carries `q` from the met1 column to the *other* inverter's gate, track B `qb`.
    cell.shapes(lay["li1"]).insert(
        O.rect(x["q"] - li_pad / 2 - lstub, y["ta_li0"],
               x["poly_q"] + pad_w / 2 + lstub, y["ta_li1"]))
    cell.shapes(lay["li1"]).insert(
        O.rect(x["poly_qb"] - pad_w / 2 - lstub, y["tb_li0"],
               x["qb"] + li_pad / 2 + lstub, y["tb_li1"]))

    # ---- wells and taps ----
    cell.shapes(lay["nwell"]).insert(O.rect(*sp.nwell))
    for tap, imp, ylic in ((sp.ptap, "psdm", y["ptap_lic0"]), (sp.ntap, "nsdm", y["ntap_lic0"])):
        cell.shapes(lay["tap"]).insert(O.rect(*tap))
        # A row of cuts across the tap, on the coarser of the licon and mcon spacings, since
        # the two are stacked here: licon.2 wants 0.17 between cuts and ct.2 wants 0.19.
        pitch = lic + max(R["licon.2"], R["ct.2"]) + CL.SLACK
        span = tap[2] - tap[0] - 2 * R["licon.7"] - lic
        n = max(1, int(span // pitch) + 1)
        for k in range(n):
            cx = CL.grid(tap[0] + R["licon.7"] + lic / 2 + (span * k / max(1, n - 1)))
            licon(cx, ylic)
            mcon(cx, ylic)
    # li1 and met1 over each tap row, plus the two rails.
    for y0, y1, m0, m1 in ((y["vss_li0"], y["vss_li1"], y["vss_m0"], y["vss_m1"]),
                           (y["vdd_li0"], y["vdd_li1"], y["vdd_m0"], y["vdd_m1"])):
        lo = sp.ptap if y0 == y["vss_li0"] else sp.ntap
        cell.shapes(lay["li1"]).insert(O.rect(lo[0], y0, lo[2], y1))
        cell.shapes(lay["met1"]).insert(O.rect(x["rail0"], m0, x["rail1"], m1))

    # ---- met1: the three vertical columns, the two rail spurs, bl and blb ----
    cell.shapes(lay["met1"]).insert(
        O.rect(x["q"] - W_COL / 2, y["n_lic0"] - R["m1.4"], x["q"] + W_COL / 2,
               y["p_lic1"] + R["m1.4"]))
    cell.shapes(lay["met1"]).insert(
        O.rect(x["qb"] - W_COL / 2, y["n_lic0"] - R["m1.4"], x["qb"] + W_COL / 2,
               y["p_lic1"] + R["m1.4"]))
    mcon(x["q"], y["n_lic0"])
    mcon(x["q"], y["ta_lic0"])
    mcon(x["q"], y["p_lic0"])
    mcon(x["qb"], y["n_lic0"])
    mcon(x["qb"], y["tb_lic0"])
    mcon(x["qb"], y["p_lic0"])
    cell.shapes(lay["met1"]).insert(                      # vss rail up to the latch's source
        O.rect(x["mid"] - W_COL / 2, y["vss_m0"], x["mid"] + W_COL / 2,
               y["n_lic1"] + R["m1.4"]))
    cell.shapes(lay["met1"]).insert(                      # vdd rail down to the latch's source
        O.rect(x["mid"] - W_COL / 2, y["p_lic0"] - R["m1.4"], x["mid"] + W_COL / 2,
               y["vdd_m1"]))
    mcon(x["mid"], y["n_lic0"])
    mcon(x["mid"], y["p_lic0"])
    for cx in (accL.licons[0], accR.licons[1]):           # bl, blb stubs
        cell.shapes(lay["met1"]).insert(
            O.rect(cx - W_COL / 2, y["n_lic0"] - R["m1.4"], cx + W_COL / 2, y["npc1"]))
        mcon(cx, y["n_lic0"])

    # ---- implants ----
    e = IMPLANT_ENC
    cell.shapes(lay["nsdm"]).insert(
        O.rect(sp.islands["accL"].x0 - e, y["nsd0"] - e, sp.islands["accR"].x1 + e,
               y["nsd1"] + e))
    cell.shapes(lay["psdm"]).insert(O.rect(sp.ptap[0], sp.ptap[1] - e, sp.ptap[2],
                                           sp.ptap[3] + e))
    # The p+ implant over the pull-ups and the n+ over the well tap would otherwise meet; a
    # gate under both implants is neither an ngate nor a pgate to the LVS deck, so they are
    # held apart at the one place they come close.
    p_top = min(g(y["psd1"] + e), g(y["ntap0"] - e - CL.SLACK))
    cell.shapes(lay["psdm"]).insert(
        O.rect(sp.islands["latchP"].x0 - e, y["psd0"] - e, sp.islands["latchP"].x1 + e, p_top))
    cell.shapes(lay["nsdm"]).insert(
        O.rect(sp.ntap[0] - e, g(max(y["ntap0"] - e, p_top + CL.SLACK)), sp.ntap[2] + e,
               y["ntap1"] + e))

    if labels:
        for net, cx, cy in (("wl", x["wl_pad"], (y["wl_m0"] + y["wl_m1"]) / 2),
                            ("vss", x["rail0"] + 0.2, (y["vss_m0"] + y["vss_m1"]) / 2),
                            ("vdd", x["rail0"] + 0.2, (y["vdd_m0"] + y["vdd_m1"]) / 2),
                            ("bl", accL.licons[0], y["npc1"] - 0.1),
                            ("blb", accR.licons[1], y["npc1"] - 0.1)):
            cell.shapes(lay["met1_label"]).insert(kdb.Text(net, O.um(cx), O.um(cy)))


def cell_gds(path: Path, sp: Spec, layers: O.Layers = None, labels: bool = True,
             poly_w: float = None, cellname: str = CELL_NAME) -> Path:
    layers = layers or O.Layers()
    _check_layers()
    ly, top = O.new_layout(cellname)
    draw_cell(ly, top, sp, layers, poly_w=poly_w, labels=labels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(path))
    return path


def narrow_poly(sp: Spec) -> float:
    """The widest illegal poly width that keeps every gate vertex on grid.

    `draw_cell` centres the gate on its centreline, so the width has to come down in steps of
    `2 * GRID` or the off-grid rules fire instead of the one being tested.
    """
    return CL.grid(sp.rules["poly.1a"] - 2 * CL.GRID)


def bad_cell_gds(path: Path, sp: Spec, layers: O.Layers = None) -> Path:
    """The real cell with every gate narrowed below `poly.1a`, and nothing else changed.

    The cell already trips two front-end rules for reasons no drawing can fix, so a harness
    test has to plant a *distinguishable* one: `poly.1a` is a front-end rule this cell does not
    otherwise violate, invisible with `FEOL = false`, and it fires once per poly shape.
    """
    return cell_gds(path, sp, layers, labels=False, poly_w=narrow_poly(sp),
                    cellname=BAD_CELL_NAME)


# --------------------------------------------------------------------------
# Measuring the written file back.
# --------------------------------------------------------------------------
def verify_layout(gds: Path, sp: Spec) -> dict:
    """Re-derive the six devices from the GDS and check them against the plan.

    docs/LAYOUT.md 5.1's rule: assert a known dimension end to end rather than trust the
    convention.  Here the dimension is the channel -- `poly AND diff`, which is what both the
    DRC deck and the LVS deck call the gate -- measured out of the file in microns and compared
    against the widths `sram6t` simulated.  A units slip anywhere in the drawing path shows up
    as a factor of 1000 here rather than as a mysterious LVS parameter mismatch.
    """
    ly = kdb.Layout()
    ly.read(str(gds))
    top = ly.top_cell()
    reg = {k: kdb.Region(top.begin_shapes_rec(ly.layer(*v))) for k, v in SRAM_LAYERS.items()}
    for r in reg.values():
        r.merge()
    gate = reg["poly"] & reg["diff"]
    gate.merge()
    # (L, W) per gate, read as (x extent, y extent) rather than (shorter, longer): every gate
    # here has its poly running in y and its current in x, and W = 0.14 um is *narrower* than
    # L = 0.15 um, so sorting by size would silently transpose two of the six devices.
    boxes = sorted((round(p.bbox().width() * ly.dbu, 6), round(p.bbox().height() * ly.dbu, 6))
                   for p in gate.each())
    got = {"n_gates": gate.size(), "gates": boxes,
           "widths": sorted({b[1] for b in boxes}), "lengths": sorted({b[0] for b in boxes}),
           "nwell_area": reg["nwell"].area() * ly.dbu ** 2,
           "diff_area": reg["diff"].area() * ly.dbu ** 2,
           "bbox": tuple(round(v * ly.dbu, 6) for v in
                         (top.bbox().left, top.bbox().bottom, top.bbox().right,
                          top.bbox().top))}
    want_w = sorted({d.w for d in sp.devices})
    want_l = sorted({d.l for d in sp.devices})
    for name, a, b in (("gate count", got["n_gates"], len(sp.devices)),
                       ("drawn channel widths", got["widths"], want_w),
                       ("drawn channel lengths", got["lengths"], want_l)):
        if a != b:
            raise AssertionError(f"{name} measured back out of {gds.name} is {a!r}, the plan "
                                 f"says {b!r}")
    # And that the per-device widths are the multiset the netlist has, not just the set.
    want = sorted((d.l, d.w) for d in sp.devices)
    if boxes != want:
        raise AssertionError(f"{gds.name}: gate (L, W) pairs are {boxes!r}, the netlist has "
                             f"{want!r}")
    return got


# --------------------------------------------------------------------------
# LVS.
# --------------------------------------------------------------------------
# The extractor's names for what this layout draws.  There is no `special_*` extraction rule
# in sky130.lvs and no layer that would key one, so these are forced, not chosen.
EXTRACTED = {S.NFET_LATCH.name: "sky130_fd_pr__nfet_01v8",
             S.NFET_PASS.name: "sky130_fd_pr__nfet_01v8",
             S.PFET_LATCH.name: "sky130_fd_pr__pfet_01v8"}

LVS_EXTRA = {"lvs_sub": "vss"}   # the cell's own name for the substrate net


def hdl21_netlist(path: Path, sp: Spec = None, params: S.Sram6TParams = None,
                  cellname: str = CELL_NAME) -> Path:
    """Stage A's cell, netlisted.  The LVS reference, straight from `sram6t.build_sram6t`."""
    import hdl21 as h

    m = S.build_sram6t(params or S.Sram6TParams(), name=cellname)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        h.netlist(h.to_proto([m]), f, fmt="spice")
    return path


def substituted_netlist(src: Path, dst: Path) -> Path:
    """The same netlist with each `special_*` device renamed to what the extractor emits.

    Not a fix and not a cheat -- a second experiment.  Running LVS both ways separates "the
    deck cannot see these device models" from "the layout is the wrong circuit", which one run
    cannot do.  See `LVS_COMPARED`.
    """
    text = Path(src).read_text()
    for a, b in EXTRACTED.items():
        text = text.replace(a, b)
    dst = Path(dst)
    dst.write_text(text)
    return dst


def lvs_variants(gds: Path, netlist: Path, rundir: Path) -> list:
    """[(label, LvsResult), ...] for the netlist as Stage A writes it and as extracted."""
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    si = O.lvs_netlist_units(netlist, rundir / "ref_si.spice")
    sub = substituted_netlist(si, rundir / "ref_si_subst.spice")
    return [("Stage-A netlist verbatim (special_* devices)",
             O.run_lvs(gds, si, rundir / "verbatim", extra=LVS_EXTRA)),
            ("with the device classes the extractor emits",
             O.run_lvs(gds, sub, rundir / "substituted", extra=LVS_EXTRA))]


def _perturb(text: str, param: str, factor: float, once: bool = True) -> str:
    """Scale one (or every) `param='(x)*1e-6'` in an already unit-adapted netlist."""
    pat = re.compile(rf"\b{param}='\(([\d.eE+-]+)\)\*1e-6'", re.I)
    return pat.sub(lambda m: f"{param}='({float(m.group(1)) * factor:.8f})*1e-6'", text,
                   count=1 if once else 0)


def lvs_probes(gds: Path, netlist: Path, rundir: Path) -> list:
    """Measure what LVS compares here, by breaking one thing at a time.

    `netlist` is the adapted, device-class-substituted reference that already matches.  Each
    probe changes exactly one thing and re-runs, so "W is compared" and "the source/drain area
    is not" become observations instead of a reading of the deck source -- which matters,
    because docs/LAYOUT.md 5 records a MATCH that was far weaker than the word implies and the
    same trap is available here.  Returns [(label, expectation, matched), ...].
    """
    rundir = Path(rundir)
    base = Path(netlist).read_text()
    probes = [
        ("reference, unchanged", True, base),
        ("W of one device x1.5", False, _perturb(base, "w", 1.5)),
        ("L of one device x1.5", False, _perturb(base, "l", 1.5)),
        ("W of one device x1.001 (0.1% high)", False, _perturb(base, "w", 1.001)),
        ("W of one device x1.0001 (0.01% high)", False, _perturb(base, "w", 1.0001)),
        ("nonsense AS/AD/PS/PD declared on one device", True,
         base.replace("w='(0.21)*1e-6'", "w='(0.21)*1e-6' as='9' ad='9' ps='9' pd='9'", 1)),
        ("nfet_01v8 -> nfet_01v8_lvt on one device", False,
         base.replace("sky130_fd_pr__nfet_01v8\n", "sky130_fd_pr__nfet_01v8_lvt\n", 1)),
        ("cross-coupling broken on one leg", False,
         base.replace("+ qb q vss vss", "+ qb qb vss vss", 1)),
        # Not a miss: the cell is symmetric under (bl,q) <-> (blb,qb), so a swapped pair of
        # bitlines is a genuinely isomorphic netlist.  Kept to show the comparer is graph-based
        # and that a "MATCH" is a statement about the circuit, not about the port names.
        ("bl and blb swapped (an isomorphism of this cell)", True,
         base.replace("+ bl wl", "+ TMP wl").replace("+ blb wl", "+ bl wl")
             .replace("+ TMP wl", "+ blb wl")),
    ]
    out = []
    for i, (label, want, text) in enumerate(probes):
        p = rundir / f"probe{i}.spice"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        r = O.run_lvs(gds, p, rundir / f"probe{i}", extra=LVS_EXTRA)
        out.append((label, want, r.ok))
    return out


def extracted_devices(result: O.LvsResult) -> dict:
    """{device class: count} out of the extracted netlist, so the class claim is measured."""
    if not Path(result.extracted).is_file():
        return {}
    out = {}
    for line in Path(result.extracted).read_text().splitlines():
        m = re.match(r"\s*[Xx]\S+\s+.*?(sky130_fd_pr__\S+)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


# --------------------------------------------------------------------------
# The area comparison, against a bitcell this PDK actually ships.
# --------------------------------------------------------------------------
# The one bitcell in this install: the OpenRAM dual-port cell, inside the SRAM macros' GDS.
# This install ships no single-port 6T cell, so the widely quoted ~1 um^2 sky130 6T figure
# cannot be measured here and is deliberately not asserted.  What can be measured is this: an
# *eight*-transistor, two-port cell drawn by the foundry, which is still far smaller than the
# six-transistor one-port cell drawn below under the published rules.
FOUNDRY_CELL = "sky130_fd_bd_sram__openram_dp_cell"


@dataclass(frozen=True)
class FoundryCell:
    """The PDK's own bitcell, cut out of its macro so the same harness can measure it."""
    name: str
    gds: Path
    source: Path
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def __str__(self):
        return (f"{self.name}: {self.w:.3f} x {self.h:.3f} um = {self.area:.2f} um^2 "
                f"(from {self.source.name})")


def foundry_bitcell(path: Path, name: str = FOUNDRY_CELL) -> FoundryCell:
    """Extract the PDK's bitcell into its own GDS, so DRC and area run on it unchanged.

    A copy, because the PDK stays read-only.  Cutting one cell out of an abutting array does
    change what the boundary rules see -- neighbours would have supplied the geometry just
    outside -- so the interconnect and well-spacing counts on it are not comparable to a
    stand-alone cell's.  The intra-device rules (`difftap.1`, `difftap.2`, `licon.5`, `poly.7`,
    `poly.8`) are not affected by that, and they are the ones this comparison is about.
    """
    root = O.C.sky130_root()
    for src in sorted(root.glob("libs.ref/sky130_sram_macros/gds/*.gds")):
        ly = kdb.Layout()
        ly.read(str(src))
        if not ly.cell(name):
            continue
        cell = ly.cell(name)
        out = kdb.Layout()
        out.dbu = ly.dbu
        top = out.create_cell(name)
        cm = kdb.CellMapping()
        cm.for_single_cell_full(out, top.cell_index(), ly, cell.cell_index())
        out.copy_tree_shapes(ly, cm)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write(str(path))
        b = out.cell(name).dbbox()
        return FoundryCell(name=name, gds=path, source=src, w=b.width(), h=b.height())
    raise RuntimeError(f"{name} not found in any GDS under "
                       f"{root / 'libs.ref/sky130_sram_macros/gds'}; this PDK install ships no "
                       "bitcell to compare against")


# --------------------------------------------------------------------------
def main(rundir: Path = Path("/tmp/sram_layout")):
    t_start = time.perf_counter()
    rundir = Path(rundir)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)

    layers = O.Layers()
    _check_layers()
    CL._check_grid()
    rules = sram_rules()
    sp = plan_layout(rules)

    print("=" * 78)
    print("STAGE B  the 6T bitcell, drawn: DRC with FEOL on, LVS against the Stage-A netlist")
    print("=" * 78)
    print(f"grid  {CL.GRID} um, from the deck's own ongrid() threshold")
    print(f"{len(rules)} DRC dimensions, every one parsed off the line that emits the rule:")
    items = sorted(rules.items())
    for i in range(0, len(items), 4):
        print("  " + "".join(f"{k:>11s} {v:<7g}" for k, v in items[i:i + 4]))
    print(f"plus IMPLANT_ENC = {IMPLANT_ENC} um, which this deck does not check -- see the "
          "constant.")

    print("\n" + "-" * 78)
    print("the cell")
    print("-" * 78)
    print(f"  {sp}")
    for d in sp.devices:
        print(f"    {d.name:<6s} {d.role:<10s} W/L = {d.w:g}/{d.l:g} um, gate at "
              f"x = {d.gate_x:.3f}, y = {d.y0:.3f}..{d.y1:.3f}")
    print(f"  islands (diffusion), left to right:")
    for name, isl in sp.islands.items():
        print(f"    {name:<7s} x {isl.x0:.3f}..{isl.x1:.3f}, heads y "
              f"{isl.y_sd0:.3f}..{isl.y_sd1:.3f}, channel {isl.w:g} um, "
              f"contacts {', '.join(f'{n}@{c:.3f}' for c, n in zip(isl.licons, isl.nets))}")
    box = lambda b: "(" + ", ".join(f"{float(v):.3f}" for v in b) + ")"
    print(f"  nwell {box(sp.nwell)}, n+ tap {box(sp.ntap)}, p+ tap {box(sp.ptap)} "
          "(x0, y0, x1, y1)")
    print(f"  the cross-couple sits in a {sp.y['psd0'] - sp.y['nsd1']:.3f} um li1 channel "
          f"between the two diffusion bands:\n    track A (q)  li1 y "
          f"{sp.y['ta_li0']:.3f}..{sp.y['ta_li1']:.3f}, poly contact on the qb-side gate\n"
          f"    track B (qb) li1 y {sp.y['tb_li0']:.3f}..{sp.y['tb_li1']:.3f}, poly contact "
          "on the q-side gate\n"
          "  each track's li1 crosses the *other* gate's poly with no contact between them, "
          "which\n  is what lets two nets swap sides on one interconnect layer.")

    gds = cell_gds(rundir / "cell.gds", sp, layers)
    m = verify_layout(gds, sp)
    print(f"  measured back out of {gds.name}: {m['n_gates']} gates, (L, W) = "
          f"{m['gates']}\n  bbox {m['bbox']} um, diff {m['diff_area']:.3f} um^2, "
          f"nwell {m['nwell_area']:.3f} um^2; matches the plan")

    print("\n" + "-" * 78)
    print("DRC, with FEOL enabled")
    print("-" * 78)
    deck = CL.feol_deck(rundir / "sky130A_feol.lydrc")
    print(f"  patched deck: {deck}  (FEOL = false -> true; the PDK copy is untouched)")
    off = O.run_drc(gds, rundir, deck=None)
    on = O.run_drc(gds, rundir / "feol", deck=deck)
    print(f"  cell, FEOL off  {off.n_rules_checked:4d} rules  {off}")
    print(f"  cell, FEOL on   {on.n_rules_checked:4d} rules  {on}")
    blocked = {"difftap.1", "difftap.2"}
    extra = set(on.by_rule) - blocked
    if off.by_rule:
        raise AssertionError(f"the back end of the cell should be clean: {off.by_rule}")
    if extra:
        raise AssertionError(
            f"FEOL DRC found rules beyond the two the device widths force: {extra}. Those are "
            f"drawing errors and must not hide behind {sorted(blocked)}. Full count: "
            f"{on.by_rule}")
    print(f"  the back end is clean and the front end trips exactly "
          f"{', '.join(f'{k} x{v}' for k, v in sorted(on.by_rule.items()))}, nothing else.")
    for line in CL._wrap(RULE_STORY, 74):
        print(f"    {line}")

    bad = bad_cell_gds(rundir / "badcell.gds", sp, layers)
    bad_off = O.run_drc(bad, rundir, deck=None)
    bad_on = O.run_drc(bad, rundir / "feol", deck=deck)
    print(f"  planted violation: every gate narrowed to {narrow_poly(sp):g} um, poly.1a wants "
          f"{rules['poly.1a']:g}")
    print(f"    FEOL off      {bad_off}")
    print(f"    FEOL on       {bad_on}")
    if bad_off.by_rule:
        raise AssertionError("the planted front-end violation should be invisible with FEOL "
                             f"off, got {bad_off.by_rule}")
    if "poly.1a" not in bad_on.by_rule:
        raise AssertionError(f"FEOL on should flag poly.1a on the narrowed gates, got "
                             f"{bad_on.by_rule}")
    print(f"  harness proven on the front end: {on.n_rules_checked - off.n_rules_checked} "
          "extra rule categories, a narrowing\n  that is invisible with FEOL off and trips "
          "poly.1a with it on, and a clean back end\n  in both cases.")

    print("\n" + "-" * 78)
    print("LVS against the Stage-A netlist")
    print("-" * 78)
    net = hdl21_netlist(rundir / "cell_ref.spice", sp)
    print(f"  layout {gds}\n  netlist {net}")
    results = lvs_variants(gds, net, rundir / "lvs")
    for label, r in results:
        print(f"  {label:52s} {r}")
    if results[0][1].ok:
        raise AssertionError(
            "the Stage-A netlist matched verbatim, which contradicts sky130.lvs having no "
            "extraction rule for the special_* devices; re-read the deck before believing it")
    devs = extracted_devices(results[-1][1])
    if devs:
        print("  device classes in the extracted netlist: "
              + ", ".join(f"{k.split('__')[-1]} x{v}" for k, v in sorted(devs.items())))
    if not results[-1][1].ok:
        print("  --- deck output tail ---")
        print("  " + "\n  ".join(results[-1][1].stdout_tail.strip().splitlines()[-16:]))
        raise AssertionError("the substituted netlist should LVS-match the drawn cell")
    print("  extracted netlist:")
    print("  " + "\n  ".join(Path(results[-1][1].extracted).read_text().strip().splitlines()[2:]))
    print("\n  what LVS compares, measured one broken thing at a time:")
    sub = rundir / "lvs" / "ref_si_subst.spice"
    for label, want, got in lvs_probes(gds, sub, rundir / "probes"):
        verdict = "MATCH" if got else "NO MATCH"
        print(f"    {label:50s} {verdict:9s}{'' if got == want else '  <-- UNEXPECTED'}")
        if got != want:
            raise AssertionError(f"LVS probe {label!r}: expected "
                                 f"{'MATCH' if want else 'NO MATCH'}")
    for line in CL._wrap(LVS_COMPARED, 74):
        print(f"    {line}")

    print("\n" + "-" * 78)
    print("area, against a bitcell this PDK actually ships")
    print("-" * 78)
    w, h = sp.size
    print(f"  drawn here   {w:.3f} x {h:.3f} um = {sp.area:.2f} um^2   (6T, single port)")
    fc = foundry_bitcell(rundir / "foundry_cell.gds")
    print(f"  {fc}")
    print(f"  ratio        the drawn cell is {sp.area / fc.area:.2f}x the foundry cell's area "
          f"while holding two\n               fewer transistors and one fewer port: "
          f"{sp.area / len(sp.devices):.2f} um^2 per device against\n               "
          f"{fc.area / 8:.2f}, so the foundry cell is {(sp.area / len(sp.devices)) / (fc.area / 8):.1f}x "
          "denser per transistor.")
    f_off = O.run_drc(fc.gds, rundir / "foundry")
    f_on = O.run_drc(fc.gds, rundir / "foundry_feol", deck=deck)
    print(f"  and the same harness on it:")
    print(f"    FEOL off   {f_off}")
    print(f"    FEOL on    {f_on}")
    shared = sorted(set(f_on.by_rule) & blocked)
    print(f"  It trips {f_on.n_violations} violations across {len(f_on.by_rule)} rules against "
          f"this cell's {on.n_violations} across {len(on.by_rule)},\n  and {', '.join(shared)} "
          "are among them -- the same rules, for the same reason: those are\n  properties of "
          "the device widths, and the foundry's cell uses the same widths.")
    print("  Read the area gap in that light. It is the price of obeying rules the foundry's")
    print("  own cell does not obey, not an implementation shortfall to be optimised away.")
    print("  Two caveats, stated rather than buried: this is an 8T dual-port cell and not a")
    print("  6T, so the transistor counts differ; and cutting one cell out of an abutting")
    print("  array removes the neighbours that would have supplied its boundary geometry, so")
    print("  its interconnect and well-spacing counts are inflated by the extraction. The")
    print("  intra-device rules are not affected by either caveat.")
    # The necks that set the width are the ones in the NMOS band; latchP shares latchN's x.
    n_necks = 2 * sum(len(isl.gates) for k, isl in sp.islands.items() if k != "latchP")
    print(f"  What sets *this* cell's size is in the plan above: {sp.neck_ext:g} um of "
          f"source/drain at the\n  channel width on each side of every gate (poly.7) before "
          f"the diffusion may flare to a\n  contact head -- {n_necks} of those across the row, "
          f"{n_necks * sp.neck_ext:.2f} um, or {n_necks * sp.neck_ext / w * 100:.0f}% of the "
          f"width; and a\n  {sp.y['psd0'] - sp.y['nsd1']:.2f} um li1 channel for the "
          "cross-couple, because two li1 tracks plus npc\n  enclosure plus npc-to-gate "
          "clearance do not fit in less.")

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s, artifacts in {rundir}")


if __name__ == "__main__":
    main()
