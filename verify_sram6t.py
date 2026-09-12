"""Machine checks on the 6T bitcell's margins, and on the drawn cell.

Same shape as `verify_mna.py`: a `CHECKS` list of predicates, a printed verdict per check,
`VERIFY: PASS` and a nonzero exit on failure.  One structural difference, and it is
deliberate: every margin check reads a single `sram6t.Margins` measured once, instead of
running its own simulations.  ngspice spends ~11 s parsing the sky130 model library and ~1 ms
solving these circuits, so per-check simulation would cost half an hour to produce the same
numbers.  `sram6t.measure` is the one place they come from.

The Stage-B checks (`--layout`) need KLayout and take a few seconds; they are off by default so
this file still runs where only ngspice is installed.  They assert the *known* failure as well
as the successes: the drawn cell is not DRC clean, and the two rules it trips are exactly the
two the device widths force, so a new violation cannot hide behind them.

Checks, in order:
    1  bins            each device's geometry lands in exactly one model bin, parsed out of
                       its own model file -- the "no sizing exercise" claim, asserted.
    2  units           the micron convention holds end to end: the three drive currents
                       reproduce their references, and the same devices written as SI metres
                       are refused by ngspice with a modelname error.
    3  netlist         six devices, the right three types, the right connectivity, and the
                       geometry reaching the deck as microns rather than Prefixed metres.
    4  quasi-static    the ramp that stands in for a DC sweep moves the SNM by < 5 mV when
                       it is run ten times slower.
    5  butterfly       the largest-square solver is right: the square fits, one 2 mV larger
                       does not, and its corners sit on the two curves.
    6  read SNM        > 100 mV at every corner measured, and worse than hold SNM at each --
                       the acceptance criterion.
    7  hold            a written value survives 200 ns of wordline-low with both bitlines
                       precharged, drifting < 1 mV.
    8  read stability  wordline high, both bitlines at VDD: the stored 0 stays below the
                       other inverter's trip point, and its peak equals the read VTC's low
                       level, which is the same number two independent ways.
    9  write margin    the cell writes, the boundary is monotone in the bitline voltage, and
                       the margin is > 300 mV at every corner.
   10  layout          (--layout) the drawn cell's six gates measure back at the widths that
                       were simulated; the back end is DRC clean; FEOL DRC trips difftap.1 and
                       difftap.2 and nothing else; a planted poly.1a narrowing is invisible
                       with FEOL off and caught with it on.
   11  lvs             (--layout) the Stage-A netlist does *not* match verbatim, the one with
                       the extractor's device classes does, and the probes pin what that match
                       covers -- W and L exactly, source/drain area not at all.
"""
import sys

import numpy as np

import sram6t as S

# Acceptance thresholds.  Read SNM is the one that matters: below ~100 mV at 1.8 V a bitcell
# is not manufacturable, and at or below zero it does not hold state at all.
READ_SNM_MIN = 0.100      # V
WRITE_MARGIN_MIN = 0.300  # V
HOLD_DRIFT_MAX = 1e-3     # V over the hold window
RAMP_SNM_MAX = 5e-3       # V, how much the quasi-static substitution may move the SNM
UNIT_TOL = 1e-3           # relative, drive current vs. its reference

_MARGINS = None


def margins() -> S.Margins:
    """The one measurement pass, shared by every check below."""
    global _MARGINS
    if _MARGINS is None:
        _MARGINS = S.measure()
    return _MARGINS


def report(checks: dict) -> bool:
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_bins() -> bool:
    """Every device sits in exactly one bin of its own model file, and W is single-valued."""
    mg = margins()
    checks = {}
    for d in S.DEVICES:
        b = mg.bins[d.name]
        print(f"  {d.role:<10s} W/L = {d.w:g}/{d.l:g} um; bin {b}")
        checks[f"{d.role} in exactly one bin"] = b.holds(d.w, d.l)
        # A bin whose W window is wider than a couple of nanometres would mean there *was* a
        # sizing choice to make; these are 1 nm wide, which is the point.
        checks[f"{d.role} W window is a single value"] = (b.w_max - b.w_min) < 0.002
    # And the negative direction: half the drawn width has no model at all.
    devs = [S.SramDevice(d.role, d.name, d.w / 2, d.l, d.module, d.pmos) for d in S.DEVICES]
    for d in devs:
        checks[f"{d.role} at half W is in no bin"] = not any(b.holds(d.w, d.l)
                                                            for b in d.bins)
    return report(checks)


def check_units() -> bool:
    """The micron convention, asserted in both directions.  See `sram6t.UNIT_NOTE`."""
    mg = margins()
    scale, where = mg.scale
    print(f"  .option scale = {scale:g} from {where}")
    neg = mg.units["si_negative_control"]
    checks = {"the corner's include chain sets scale = 1e-6": scale == 1e-6}
    for d in S.DEVICES:
        r = mg.units[d.name]
        print(f"  {d.role:<10s} |Id| {r['i'] * 1e6:8.3f} uA vs reference "
              f"{r['want'] * 1e6:.3f} uA  ({100 * r['rel']:.3f}%)")
        checks[f"{d.role} drive current matches its reference"] = r["rel"] < UNIT_TOL
    checks["SI-metre geometry is refused, not silently mis-sized"] = neg["failed"]
    checks["and refused for the bin miss specifically"] = neg["modelname"]
    return report(checks)


def check_netlist() -> bool:
    """The six devices, their connectivity, and the geometry as it reaches the deck."""
    import io

    import hdl21 as h

    m = S.build_sram6t()
    kinds = {}
    for inst in m.instances.values():
        kinds[inst.of.module.name] = kinds.get(inst.of.module.name, 0) + 1
    buf = io.StringIO()
    h.netlist(h.to_proto([S.build_sram6t()]), buf, fmt="spice")
    text = buf.getvalue()
    print(f"  {len(m.instances)} instances: "
          + ", ".join(f"{k.split('__')[-1]} x{v}" for k, v in sorted(kinds.items())))
    conns = {n: {k: v.name for k, v in i.conns.items()} for n, i in m.instances.items()}
    checks = {
        "six devices": len(m.instances) == 6,
        "two pull-downs, two pull-ups, two access": kinds == {
            S.NFET_LATCH.name: 2, S.PFET_LATCH.name: 2, S.NFET_PASS.name: 2},
        "ports are bl, blb, wl, vdd, vss": list(m.ports) == ["bl", "blb", "wl", "vdd", "vss"],
        # The cross-coupling: each inverter's gate is the *other* node.
        "mnq/mpq gated by qb, drain on q": (
            conns["mnq"] == {"d": "q", "g": "qb", "s": "vss", "b": "vss"}
            and conns["mpq"] == {"d": "q", "g": "qb", "s": "vdd", "b": "vdd"}),
        "mnqb/mpqb gated by q, drain on qb": (
            conns["mnqb"] == {"d": "qb", "g": "q", "s": "vss", "b": "vss"}
            and conns["mpqb"] == {"d": "qb", "g": "q", "s": "vdd", "b": "vdd"}),
        "access devices join q/qb to bl/blb, gated by wl": (
            conns["maq"] == {"d": "bl", "g": "wl", "s": "q", "b": "vss"}
            and conns["maqb"] == {"d": "blb", "g": "wl", "s": "qb", "b": "vss"}),
        # The unit hazard, in the emitted text: microns, not the 2.1e-07 a Prefixed gives.
        "geometry emitted in microns": ("w='0.21' l='0.15'" in text
                                        and "w='0.14' l='0.15'" in text),
        "no SI-metre geometry anywhere in the netlist": "2.1e-07" not in text,
        "devices netlist as subckt calls": text.count("\nxm") == 6,
    }
    return report(checks)


def check_quasi_static() -> bool:
    """A ramp is standing in for a DC sweep; slowing it 10x must not move the answer."""
    r = margins().ramp
    print(f"  {r}")
    checks = {
        f"SNM moves < {RAMP_SNM_MAX * 1e3:g} mV when the ramp is {r.factor:g}x slower":
            r.d_snm < RAMP_SNM_MAX,
    }
    return report(checks)


def check_butterfly() -> bool:
    """The largest-square solver, checked rather than trusted.

    Three independent facts: the square it found is inside the lobe, a square 2 mV larger at
    the same corner is not, and its two opposite corners lie on the two curves.  The third is
    what distinguishes "largest inscribed square" from "some square that fits".
    """
    p = S.Sram6TParams()
    checks = {}
    for tag, wl_v in (("read", S.VDD), ("hold", 0.0)):
        vin, vout = S.vtc(p, "tt", wl_v=wl_v, rundir=f"/tmp/sram6t_verify/vtc_{tag}")
        s = S.snm(vin, vout)
        o = np.argsort(vin)
        x, y = vin[o], vout[o]
        yi = np.argsort(y)
        on_upper = abs(np.interp(s.x0 + s.snm, x, y) - (s.y0 + s.snm))
        on_lower = abs(np.interp(s.x0, y[yi], x[yi]) - s.y0)
        print(f"  {tag:<5s} SNM {s.snm * 1e3:7.2f} mV; corners on the two curves to "
              f"{max(on_upper, on_lower) * 1e6:.1f} uV")
        checks[f"{tag}: the square fits"] = S.square_fits(vin, vout, s.snm, s.x0, s.y0)
        checks[f"{tag}: a 2 mV larger square does not"] = not S.square_fits(
            vin, vout, s.snm + 2e-3, s.x0, s.y0)
        checks[f"{tag}: both corners lie on the curves"] = max(on_upper, on_lower) < 1e-4
    return report(checks)


def check_read_snm() -> bool:
    """The acceptance criterion: read SNM positive and comfortably above zero, every corner.

    Also that read SNM is *worse* than hold SNM everywhere it was measured, because the
    access device lifting the low node off the rail is the entire mechanism and a corner where
    that did not happen would mean the read condition was not actually being applied.
    """
    mg = margins()
    checks = {}
    for c in (*mg.corners, *mg.skew):
        s = mg.read_snm(c)
        print(f"  {c:<4s} read SNM {s.snm * 1e3:7.2f} mV, low level {s.v_low * 1e3:6.1f} mV, "
              f"gain {s.gain:5.1f}")
        checks[f"{c}: read SNM > {READ_SNM_MIN * 1e3:g} mV"] = s.snm > READ_SNM_MIN
        checks[f"{c}: read low level is off the rail"] = s.v_low > 0.05
    for c, r in mg.corners.items():
        checks[f"{c}: read SNM < hold SNM"] = r.read.snm < r.hold.snm
    return report(checks)


def check_hold() -> bool:
    """A written 0 survives 200 ns of wordline-low with both bitlines held at VDD."""
    h_ = margins().hold
    print(f"  {h_}")
    checks = {
        "q stays low through the hold window": h_.q_max < 0.1 * S.VDD,
        "qb stays high through the hold window": h_.qb_min > 0.9 * S.VDD,
        f"drift < {HOLD_DRIFT_MAX * 1e3:g} mV over {h_.t_hold * 1e9:g} ns":
            h_.drift < HOLD_DRIFT_MAX,
    }
    return report(checks)


def check_read_stability() -> bool:
    """The read the beta ratio exists to survive, and a cross-check on the DC butterfly."""
    mg = margins()
    checks = {}
    for c, r in mg.corners.items():
        up, trip = r.upset, r.hold.v_trip
        gap = trip - up.q_peak3
        agree = abs(up.q_peak3 - r.read.v_low)
        print(f"  {c:<4s} stored 0 peaks at {up.q_peak3 * 1e3:6.1f} mV vs a "
              f"{trip:.3f} V trip point (margin {gap * 1e3:5.1f} mV); "
              f"transient vs DC butterfly agree to {agree * 1e6:.1f} uV")
        checks[f"{c}: the cell does not flip during a read"] = not up.flipped
        checks[f"{c}: the disturbed node stays below the trip point"] = gap > 0.05
        checks[f"{c}: transient read peak == read VTC low level"] = agree < 1e-3
    return report(checks)


def check_write_margin() -> bool:
    """The cell writes, the boundary is clean, and the margin is large at every corner."""
    mg = margins()
    checks = {}
    for c, r in mg.corners.items():
        w = r.write
        print(f"  {w}")
        checks[f"{c}: the flip boundary is monotone in the bitline voltage"] = w.monotone
        checks[f"{c}: write margin > {WRITE_MARGIN_MIN * 1e3:g} mV"] = (
            w.margin > WRITE_MARGIN_MIN)
        checks[f"{c}: boundary bracketed to < 1 mV"] = w.resolution < 1e-3
        # `write_margin` raises if blb = 0 fails to write or blb = VDD flips the cell, so
        # reaching here already means both; assert the recorded VDD point explicitly anyway.
        checks[f"{c}: blb = VDD does not write"] = not w.at_vdd.flipped
    return report(checks)


# --------------------------------------------------------------------------
# Stage B.  Needs the KLayout application, so it is opt-in.
# --------------------------------------------------------------------------
LAYOUT_RUNDIR = "/tmp/sram6t_verify/layout"
BLOCKED_RULES = {"difftap.1", "difftap.2"}   # the two the device widths force

_LAYOUT = None


def layout_artifacts():
    """Draw the cell, its planted-violation twin, and the reference netlist, once."""
    global _LAYOUT
    if _LAYOUT is None:
        from pathlib import Path

        import cell_layout as CL
        import layout_oracle as O
        import sram_layout as SL

        rundir = Path(LAYOUT_RUNDIR)
        rundir.mkdir(parents=True, exist_ok=True)
        sp = SL.plan_layout()
        _LAYOUT = {
            "SL": SL, "O": O, "sp": sp, "rundir": rundir,
            "gds": SL.cell_gds(rundir / "cell.gds", sp),
            "bad": SL.bad_cell_gds(rundir / "badcell.gds", sp),
            "deck": CL.feol_deck(rundir / "feol.lydrc"),
            "net": SL.hdl21_netlist(rundir / "cell_ref.spice", sp),
        }
    return _LAYOUT


def check_layout() -> bool:
    """The drawn cell: geometry back out of the file, then DRC with FEOL on both ways."""
    a = layout_artifacts()
    SL, O, sp = a["SL"], a["O"], a["sp"]
    m = SL.verify_layout(a["gds"], sp)          # raises if the drawn W/L are not the plan's
    off = O.run_drc(a["gds"], a["rundir"])
    on = O.run_drc(a["gds"], a["rundir"] / "feol", deck=a["deck"])
    bad_off = O.run_drc(a["bad"], a["rundir"])
    bad_on = O.run_drc(a["bad"], a["rundir"] / "feol", deck=a["deck"])
    print(f"  {sp}")
    print(f"  gates (L, W) out of the GDS: {m['gates']}")
    print(f"  cell     FEOL off {off}")
    print(f"  cell     FEOL on  {on}")
    print(f"  narrowed FEOL off {bad_off}")
    print(f"  narrowed FEOL on  {bad_on}")
    checks = {
        "the drawn channels are the simulated widths": (
            m["widths"] == sorted({d.w for d in sp.devices})),
        "the back end is DRC clean": off.clean,
        "FEOL on checks more rules than FEOL off": on.n_rules_checked > off.n_rules_checked,
        "FEOL on trips only the rules the device widths force": (
            set(on.by_rule) == BLOCKED_RULES),
        "difftap.1 fires once per 0.14 um device": (
            on.by_rule.get("difftap.1") == sum(1 for d in sp.devices if d.w < 0.15)),
        "difftap.2 fires on every gate": (
            on.by_rule.get("difftap.2") == 2 * len(sp.devices)),
        "the planted narrowing is invisible with FEOL off": bad_off.clean,
        "and trips poly.1a with FEOL on": "poly.1a" in bad_on.by_rule,
    }
    return report(checks)


def check_lvs() -> bool:
    """LVS, and what the match actually covers -- measured, not read off the deck."""
    a = layout_artifacts()
    SL = a["SL"]
    variants = SL.lvs_variants(a["gds"], a["net"], a["rundir"] / "lvs")
    for label, r in variants:
        print(f"  {label:52s} {r}")
    devs = SL.extracted_devices(variants[-1][1])
    print("  extracted device classes: "
          + ", ".join(f"{k.split('__')[-1]} x{v}" for k, v in sorted(devs.items())))
    checks = {
        # A positive finding, not a failure: the deck has no extraction rule for the special_*
        # devices, so a verbatim match would mean the comparison was not doing what it says.
        "the Stage-A netlist does not match verbatim": not variants[0][1].ok,
        "with the extractor's device classes it does": variants[-1][1].ok,
        "the extractor emits 4 nfet_01v8 and 2 pfet_01v8": (
            devs == {"sky130_fd_pr__nfet_01v8": 4, "sky130_fd_pr__pfet_01v8": 2}),
    }
    for label, want, got in SL.lvs_probes(a["gds"], a["rundir"] / "lvs" / "ref_si_subst.spice",
                                          a["rundir"] / "probes"):
        print(f"    probe: {label:50s} {'MATCH' if got else 'NO MATCH'}")
        checks[f"probe: {label}"] = got == want
    return report(checks)


CHECKS = [
    ("bins", check_bins),
    ("units", check_units),
    ("netlist", check_netlist),
    ("quasi-static", check_quasi_static),
    ("butterfly", check_butterfly),
    ("read SNM", check_read_snm),
    ("hold", check_hold),
    ("read stability", check_read_stability),
    ("write margin", check_write_margin),
]

LAYOUT_CHECKS = [
    ("layout", check_layout),
    ("lvs", check_lvs),
]


def main(layout: bool = False) -> int:
    results = {}
    for name, fn in CHECKS + (LAYOUT_CHECKS if layout else []):
        print(f"\n== {name} ==")
        results[name] = fn()
    if not layout:
        print("\n(Stage-B layout checks skipped; pass --layout to run them. They need the "
              "KLayout application.)")

    print()
    for name, hit in results.items():
        print(f"{'PASS' if hit else 'FAIL'}  {name}")
    ok = all(results.values())
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(layout="--layout" in sys.argv))
