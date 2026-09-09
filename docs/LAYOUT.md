# Layout → parasitics: replacing the assumed wire R and C with geometry-derived ones

[`REPORT.md`](REPORT.md) opens by saying that `r_row` / `r_col` / `c_row` / `c_col` are
**assumed** per-pitch values swept over a plausible range, and that pinning them to real
geometry is the missing step. This document takes that step. Two new modules draw sky130
geometry, DRC it, and compute R and C from the drawn dimensions and process constants parsed
out of the PDK — then re-run the published int8 accuracy study on the result.

**Read this first, because the distinction decides what the numbers below are worth.** What
runs here is

```
R = sheet_resistance(layer) · squares(polygon)
C = area_cap(layer) · area(polygon) + perimeter_cap(layer) · perimeter(polygon)
```

with every constant parsed from `sky130A.tech`, none typed in. That captures a wire's **self
resistance** and its **area plus fringe capacitance to substrate**. It is **not parasitic
extraction**: there is no field solver, no 3-D coupling matrix, no current crowding, no
frequency dependence. Call it *geometry + technology constants*. Where the difference bites
is documented rather than glossed — §7 shows that rail-to-rail coupling, which this model has
no term for, is the *same order* as the shunt capacitance it does model.

`magic` and `netgen` are not installed on this machine and have no homebrew formula, so
`ext2spice` was never an option; the constants come from magic's technology file, which ships
with the PDK regardless of whether magic does.

**Headline.** The project's default assumption of `r_row = r_col = 1 Ω/pitch` is optimistic.
Any layout that physically holds a `g_min = 1 µS` cell needs a ~30 µm pitch, which puts the
row and column rails at **3.9 Ω/pitch** — near the top of the range the study swept as a
pessimistic case. At that value the int8 match rate with per-channel calibration is **76.6%**,
not the 94.5% the 1 Ω/pitch assumption reports.

Every number below comes from a run on this machine: KLayout 0.30.10 (application) and the
0.30.12 pip module, sky130A via volare, ngspice-47, Apple silicon.

---

## 1. The chain, and how to run it

| module | what it does |
| --- | --- |
| `layout_oracle.py` | tech-constant parser, GDS generation and measurement, R/C computation, the hand check, the DRC harness, the LVS attempt |
| `array_layout.py` | the N×N crossbar interconnect skeleton, per-pitch R and C, the via and coupling analyses, the pitch/resistance loop, the accuracy re-run |

```bash
uv run layout_oracle.py                # Stage 1: constants, hand check, DRC both ways, LVS   3.6 s
uv run array_layout.py                 # Stage 2: skeleton, DRC, derived R/C, the analyses    2.5 s
uv run array_layout.py --accuracy      # Stage 2 + the int8 and float re-runs in ngspice     28.6 s
```

Artifacts (GDS, DRC report databases, LVS databases, extracted netlists) land in
`/tmp/layout_oracle` and `/tmp/xbar_layout`.

Nothing in the existing project is modified. Both modules import `crossbar`, `fp_matmul` and
`int8_matmul` and use them as they are.

Two environment requirements that `pyproject.toml` does not express, and should:

- The **`klayout` pip module** (0.30.12 here) provides `klayout.db`, which both modules import
  for GDS generation and measurement. It is installed in the venv but is **not** in
  `pyproject.toml` or `uv.lock`, so a `uv sync --exact` would remove it and break both
  modules. Add it to `[project].dependencies` before relying on this chain.
- The **KLayout application binary** is separate from the pip module and is what runs the DRC
  and LVS decks — the module cannot execute a `.lydrc`. `layout_oracle.KLAYOUT_BIN` points at
  `~/klayout-local/klayout.app/Contents/MacOS/klayout`; override it or pass `klayout=` to
  `run_drc` / `run_lvs` elsewhere.

---

## 2. Technology constants, and which corner

`sky130A.tech` carries **three** complete sets of resistances and **three** of capacitances,
selected by magic's `variants` mechanism. Getting this wrong scales every resistance in the
project by 105/125 or 145/125 with no other symptom, so here is the resolution in full:

| block | selector | tech-file line | metal1 | what its own comment says |
| --- | --- | --- | --- | --- |
| resistance 1 | `variants (),(orig),(si)` | `sky130A.tech:5100` | **125** mΩ/□ | "Device values come from trtc.cor (typical corner)" |
| resistance 2 | `variants (hrhc),(hrlc)` | `sky130A.tech:5146` | 145 mΩ/□ | "High-end corner resistances" |
| resistance 3 | `variants (lrhc),(lrlc)` | `sky130A.tech:5194` | 105 mΩ/□ | "Low-end corner resistances" |
| capacitance 1 | `variants (),(orig),(si)` | `sky130A.tech:5270` | **25.78** aF/µm² | "Nominal capacitances" |
| capacitance 2 | `variants (hrlc),(lrlc)` | `sky130A.tech:5488` | 20.2 aF/µm² | low-C |
| capacitance 3 | `variants (hrhc),(lrhc)` | `sky130A.tech:5706` | 35.7 aF/µm² | high-C |

**This work uses the `()` blocks** — magic's default extraction style, and the only ones whose
own comments claim typical/nominal. `TechConstants.parse` is a state machine over the
`variants` lines and keeps a value only when the selected corner is in the current block's
selector list.

`TechConstants.check` then asserts the parse against known values and raises with the corner
note attached if any of them moves. That assertion is the only thing standing between a
corner-selection bug and a document full of numbers that are quietly 16% or 20% wrong:

```
resist allm1 = 125 mOhm/square (sky130A.tech:5123)
resist allm2 = 125 mOhm/square (sky130A.tech:5124)
resist allm3 = 47 mOhm/square (sky130A.tech:5125)
resist allm4 = 47 mOhm/square (sky130A.tech:5126)
resist allm5 = 29 mOhm/square (sky130A.tech:5127)
resist allli = 12800 mOhm/square (sky130A.tech:5122)
resist mrp1 = 48200 mOhm/square (sky130A.tech:5115)
resist xhrpoly = 319800 mOhm/square (sky130A.tech:5116)
areacap allm1 = 25.78 aF/um^2 (sky130A.tech:5325)
perimeter allm1 = 40.57 aF/um (sky130A.tech:5326)
sidewall allm1 = 44 aF/um (sky130A.tech:5324)
overlap allm2 over allm1 = 133.86 aF/um^2 (sky130A.tech:5367)
contact mcon = 9300 mOhm/contact (sky130A.tech:5137)
contact m2c = 4500 mOhm/contact (sky130A.tech:5138)
contact nsc = 185000 mOhm/contact (sky130A.tech:5130)
```

Every `TechValue` carries its unit and its source line, and `TechConstants.provenance()`
prints the block above, so any number in this document can be checked with one `grep`.

Layer numbers are read from the PDK's own map, not hardcoded. `Layers` parses
`libs.tech/klayout/tech/sky130A.map` and asserts against what it intends to draw —
`li1 67/20, met1 68/20, met2 69/20, mcon 67/44, via 68/44, via2 69/44` — which is the same
numbering the DRC deck's `m1_wildcard = "68/0-4,6-43,45-*"` selects. Design-rule dimensions
are parsed too, off the deck line that emits each rule (`drc_rule_value("m1.1")` → `0.14`),
so §8's poly-resistor geometry has a `file:line` behind it as well.

---

## 3. The Stage-1 hand check

A 10 µm × 0.5 µm metal1 wire, small enough to check with a pencil. `hand_check` runs the full
pipeline — build GDS with `klayout.db`, write it, read it back, measure area/perimeter/box
sizes, look up constants, compute — and independently computes the same numbers by arithmetic
on `w` and `length` with no GDS involved, then asserts they agree to float precision:

```
  from GDS      met1: 20.000 squares -> R = 2.5 ohm; C = 980.9 aF (128.9 area + 852.0 fringe)
  closed form   met1: 20.000 squares -> R = 2.5 ohm; C = 980.9 aF (128.9 area + 852.0 fringe)
  by hand       10.0/0.5 = 20 squares x 0.125 ohm/sq = 2.5 ohm
                5 um^2 x 25.78 aF/um^2 + 21 um x 40.57 aF/um = 980.87 aF
```

Agreement exercises the GDS write, the read back, the database-unit scaling, the layer lookup
and the constant lookup at once — the failure modes that silently produce numbers off by a
factor of 1000.

---

## 4. The DRC harness, and the deliberate violation

DRC runs the stock PDK deck through the KLayout application in batch:

```bash
klayout -b -zz -r $PDK/libs.tech/klayout/drc/sky130A.lydrc \
        -rd input=<layout>.gds -rd report=<layout>.lyrdb
```

`run_drc` parses the resulting report database (KLayout RDB, XML) into a pass/fail plus a
per-rule violation count. **A DRC harness that has only ever run on clean layouts is not a
tested harness** — a broken report parser and a clean layout are indistinguishable. So
`bad_wire_gds` plants two known violations, 0.10 µm metal1 width against the 0.14 µm minimum
(`m1.1`) and a 0.10 µm gap against the 0.14 µm minimum (`m1.2`), and `main` asserts that the
harness comes back dirty with *exactly* those two rules:

```
  legal wire     DRC CLEAN: wire_met1.gds, 145 rules, 1.20 s
  illegal wire   DRC FAIL: badwire.gds, 2 violations [m1.1 x1, m1.2 x1], 0.79 s
```

**One scope limit on every DRC result in this document.** The stock deck ships with
`FEOL = false` (`sky130A.lydrc:46`), so what actually runs is the back-end-of-line set plus
the off-grid and angle checks: 145 rule categories covering every metal, via and
local-interconnect rule, and **no diffusion or poly rule at all**. The interconnect skeleton
of §6 is drawn entirely in metal1, metal2 and via, so it is fully covered. The transistor of
§5 is barely covered, which is one reason the LVS result there is reported as a device-
recognition result and not a sign-off. The deck was not modified to enable FEOL.

---

## 5. LVS: matched, on `W` and `L` only

**Status: MATCH — but read the next sentence before quoting it.** KLayout's
`DeviceClassMOS4Transistor` treats `AS` / `AD` / `PS` / `PD` as non-primary parameters and
**does not compare them**, so the match is on `W` and `L` alone and the drawn source/drain
areas and perimeters are checked against nothing. This was verified rather than inferred: the
adapted netlist reaches the comparer carrying `AS = 2.9e-7` against the layout's extracted
`0.29`, off by a factor of 10⁶, and LVS reports a match anyway. "LVS MATCH" is a much weaker
statement here than the phrase normally implies, and the weakness is in KLayout's comparison
semantics rather than in the geometry.

With that stated: a hand-drawn `sky130_fd_pr__nfet_01v8` does match a reference netlist
generated through the same hdl21 path `crossbar.Cell` uses, after one documented unit adapter.
Getting there took four distinct fixes, and each one is worth recording because they are the
failure modes anyone repeating this will hit.

First, a correction to the obvious starting point: `libs.tech/klayout/lvs/sky130.lylvs` is a
33-line KLayout macro wrapper whose entire body is a **commented-out** `# %include
sky130.lvs`. Running it checks nothing. The real 109 KB ruleset is `sky130.lvs` beside it, and
`run_lvs` invokes that, the way the PDK's own `run_lvs.py` does.

`crossbar.Cell` itself can never LVS-match: it contains a memristor and sky130 has no
resistive-memory device to match one against. What can be matched is the access device, so
`hdl21_nfet_netlist` builds a one-instance module from the identical `h.Nmos(...,
family=h.MosFamily.CORE)` call, with the same `si_literal` treatment and the same
`sky130_hdl21.compile`, and netlists it.

The four fixes, in the order they were needed:

1. **Top-cell name.** The deck pairs the layout's top cell with the schematic's top circuit
   by name. `Can't find a schematic counterpart for the top cell NFET`. Fixed by naming both
   `nfet_ref` (`NFET_CELL`).
2. **Terminal labels.** The deck reads net names from text layers — `connect(li_con,
   li_label)` at `sky130.lvs:1849`, `connect(poly_con, poly_label)` at `:1848`. Without them
   the extracted subcircuit had **one** pin (`sky130_gnd`) against the schematic's four. Fixed
   with `d`/`s`/`b` texts on li1 (67/5) and `g` on poly (66/5), plus `-rd lvs_sub=b` so the
   deck's global substrate net carries the same name as the schematic's body port.
3. **Subcircuit-to-device conversion.** hdl21 emits the nfet as an `X` subcircuit call. With
   the deck's default `convert_subckts=false` the schematic keeps an *empty, undefined*
   subcircuit, the deck flattens it away as having no layout counterpart, and the comparison
   is one transistor against nothing. Fixed with `-rd convert_subckts=true`, which puts the
   deck's reader delegate (`sky130.lvs:349`, `:416`) on the `X` element and turns it into a
   MOS device.
4. **Units.** This was the actual blocker, and it is a genuine incompatibility rather than an
   oversight on either side. `sky130_hdl21` multiplies device sizes by 1e6 on the way out,
   because the SkyWater ngspice models are used with `.option scale 1E6` and therefore want
   microns — that scaling is the entire reason `crossbar.si_literal` exists. KLayout's SPICE
   reader reads a MOS `W`/`L` as **SI metres** and stores microns, so it multiplies by 1e6
   again. The schematic device arrived as `L = 150000 µm, W = 1000000 µm`. `lvs_netlist_units`
   undoes exactly that one factor on `w` and `l`.

With the adapter, the layout extracts and matches:

```
.SUBCKT nfet_ref b d g s
XM$1 s g d b sky130_fd_pr__nfet_01v8 L=0.15 W=1 AS=0.29 AD=0.29 PS=2.58 PD=2.58
.ENDS nfet_ref

INFO : Congratulations! Netlists match.
```

The extraction recognized the hand-drawn geometry as the right device with the right
dimensions, which is the substantive result: `L = 0.15`, `W = 1`, and source/drain areas and
perimeters that agree with the expressions `sky130_hdl21` emits (`ad = 1·1/1·0.29 = 0.29`,
`pd = 2·1·(1 + 0.29) = 2.58`).

Beyond the `W`/`L`-only limit stated at the top of this section, one more scope caveat:
**this is one transistor, not the array.** No crossbar layout has been LVS'd, because a
crossbar layout with no device in it has nothing to match against — see §6.

### 5.1 The SI-versus-micron boundary is a recurring hazard in this toolchain, not a one-off

Fix 4 above is the **third** time this project has been bitten by the same class of bug. Listed
together they stop looking like three unrelated defects and start looking like a property of
the toolchain: sky130 flows carry geometry in microns, hdl21's public API is SI metres, and
every tool at that boundary picks its own convention with no type to enforce it.

| # | boundary | symptom | resolution |
| --- | --- | --- | --- |
| 1 | `sky130_hdl21.Sky130Walker.scale_param` (`pdk_logic.py:351`) skips the SI→µm scaling for `h.Prefixed` and applies it for `h.Literal`, with upstream's own `FIXME` | a `1 * µ` device reaches ngspice as 1 pm, misses every binned `.model`, and fails with only `could not find a valid modelname` | `crossbar.si_literal` converts `Prefixed` → `Literal` so the scaling fires; see [REPORT.md §8](REPORT.md) and the README's units section |
| 2 | reading the deck's post-scaling `w='(1e-06 * 1e6)'` as if it were already microns | a wrong micron-units reading of correct output | corrected in REPORT.md; the deck value is SI-in, µm-out |
| 3 | KLayout's SPICE reader reads MOS `W`/`L` as **SI metres** and stores microns, while `sky130_hdl21` has already converted to microns | the schematic device arrives at LVS as `L = 150000 µm, W = 1000000 µm`; LVS reports a parameter mismatch with no unit diagnostic | `layout_oracle.lvs_netlist_units` undoes exactly that one 1e6 on `w` and `l` |

All three are silent: none produces a units error, each produces a plausible-looking wrong
number or an error message about something else. The practical rule this suggests for anything
new crossing that boundary is to assert a known dimension end-to-end rather than trust the
convention — which is what §3's hand check and §5's `W = 1, L = 0.15` extraction both do.

---

## 6. The array skeleton

`array_layout.Skeleton` draws the **interconnect** of an N×N crossbar and nothing else:

- N row rails on **metal1**, running in x, width `w_row`.
- N column rails on **metal2**, running in y, width `w_col`, simply crossing over the rows —
  which is what puts a crossbar's two rail families on two layers in the first place.
- One **metal1→metal2 via** per cell off the row rail onto a small metal2 island, offset in x
  so it clears the column rail. The gap between that island and the column rail is where the
  memory device would sit. Nothing is drawn in it.

No memory device is drawn for the blunt reason that **sky130 has none**: `libs.ref` holds
`sky130_fd_pr`, `sky130_fd_io`, `sky130_fd_sc_hd`, `sky130_fd_sc_hvl`, `sky130_ml_xx_hd` and
`sky130_sram_macros`, and no resistive-memory element. That is the scope rather than a gap:
`r_row` / `r_col` / `c_row` / `c_col` *are* interconnect parameters. The cell enters only
through `g_min`, which turns out to set the pitch — §8.

`Skeleton.check` refuses a parameter set that cannot be DRC clean and names the rule, with
every threshold read out of the deck. The DRC run is still the authority, and all three
skeletons below are clean against 145 rule categories — **with the deck's `FEOL = false`
default, so every metal, via and local-interconnect rule ran and no diffusion or poly rule
did** (§4). For this skeleton that is full coverage: it is drawn entirely in metal1, metal2 and
via. The claim "DRC clean" here therefore means clean against the back-end-of-line set, which
is the whole applicable set, and it would not mean that for anything with a transistor in it.

Two things the skeleton does not include, stated so they are not read into the numbers: there
is **no select rail** (a 1T1R array needs one per row; `crossbar.Crossbar` has no `r_sel`
parameter, and adding the rail would widen the pitch), and there is **no periphery** — no
drivers, sense amplifiers or decode.

---

## 7. Derived per-pitch R and C

`r_row` means the resistance of **one more cell pitch** of rail, so that is what is measured:
two rails one pitch apart in length are drawn, measured through the same GDS round trip the
Stage-1 hand check pins, and differenced. Differencing matters — a single rail's perimeter
carries two endcaps worth `2·width`, which at a 0.74 µm pitch is a 30% error on the per-pitch
fringe term if you just divide by N. The difference cancels them exactly.

Three skeletons, `n = 16`:

| skeleton | pitch [µm] | `w_row` / `w_col` [µm] | die area | `r_row` [Ω] | `r_col` [Ω] | `c_row` [F] | `c_col` [F] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0.74 | 0.32 / 0.14 | 11.8 × 11.8 µm | **0.289** | **0.661** | 6.62e-17 | 5.77e-17 |
| relaxed | 2.0 | 0.5 / 0.5 | 32 × 32 µm | **0.500** | **0.500** | 1.88e-16 | 1.69e-16 |
| g_min-driven | 31 | 1.0 / 1.0 | 496 × 496 µm | **3.875** | **3.875** | 3.32e-15 | 2.88e-15 |

The `dense` pitch is the minimum this skeleton supports — set by the metal2 island clearing
the column rail on both sides — and it is not reachable in practice because no memory device
fits in 0.74 µm. `g_min-driven` is the one to take seriously; §8 says why 31 µm.

The capacitance splits are worth reading: at these widths the **fringe term dominates the area
term by 3:1 to 31:1** (widest on the 0.14 µm column rail, narrowest on the 1 µm rails).
`c_row` for the dense row rail is 6.1 aF of area against 60.0 aF of fringe. A model that only
counted plate capacitance would be wrong by nearly an order of magnitude.

Settling, using the Elmore expression [`REPORT.md` §6](REPORT.md) already validated
(`R_tot·C_tot/2 = r·c·N²/2`, measured low by a consistent 1.3× against a distributed line):

| skeleton | `r·c·N²/2` | ×1.3 → `tau_63` |
| --- | --- | --- |
| dense | 2.45e-15 s | 3.2e-15 s |
| relaxed | 1.20e-14 s | 1.6e-14 s |
| g_min-driven | 1.64e-12 s | 2.1e-12 s |

**Wire RC is a non-issue at these sizes.** Even the 496 µm array settles in single-digit
picoseconds; the driver and the sense amplifier will set the read time, not the rail.

### The via and contact resistance the circuit model has no term for

`crossbar.Crossbar` puts series R along the rails and nothing at the cell. Priced from the
tech file's `contact` lines, for the `dense` skeleton:

| route | tap R | × rail R per pitch | `g_max` compression |
| --- | --- | --- | --- |
| 0T1R as drawn (1 via) | 4.50 Ω | 15.6× | 0% → **0.045%** |
| 1T1R (via + mcon + li1-to-diff) | 198.80 Ω | 687.7× | 7.570% → **9.238%** |

**The "× rail R per pitch" column is the wrong comparison and is shown only to defuse it.** A
tap resistance sits in series with **one** cell and is shared with nothing, so it produces
none of the shared-segment loading that [`DESIGN.md`](DESIGN.md) measured as ~93% of the droop
at 5 Ω/pitch. Its entire effect is the weight-dependent conductance compression
`crossbar.ideal_1t1r` already models, with `ron` raised by the tap resistance — which is why
the compression column is the one that matters.

- **0T1R as drawn: genuinely negligible.** 4.5 Ω against a 10 kΩ cell at `g_max` is 0.045%.
- **1T1R: not negligible.** The chain down to the access transistor's drain (`via` 4.5 Ω +
  `mcon` 9.3 Ω + `nsc` 185 Ω) is 198.8 Ω, a **24% increase** on the 819 Ω in-array Ron
  [`REPORT.md` §5.2](REPORT.md) measured. That moves `g_max` compression from 7.569% to
  9.238%. It is small, but it is the same order as effects the study treats as significant,
  and it is missing from `crossbar.Crossbar` and from the `ideal_1t1r` reference alike.

### Rail-to-rail coupling: the C model is structurally incomplete

`crossbar.Crossbar` has shunt C to `vss` and nothing else. Three coupling terms exist in the
tech file and none of them has a counterpart in the model. For the `dense` skeleton:

| term | value | vs. the modelled shunt |
| --- | --- | --- |
| row shunt to substrate (modelled) | 66.1 aF/pitch | — |
| row to its two metal1 neighbours | 65.1 aF/pitch | **0.98×**, at 0.42 µm spacing |
| column shunt to substrate (modelled) | 57.7 aF/pitch | — |
| column to its two metal2 neighbours | 74.0 aF/pitch | **1.28×**, at 0.60 µm spacing |
| row-to-column at each crossing | 48.9 aF/cell (6.0 plate + 42.9 edge fringe) | **0.74×**, no counterpart at all |

**Verdict: the capacitance model is structurally incomplete, not merely imprecise.** Coupling
to the neighbouring rails is the same order as the shunt capacitance the model has, so
`c_row`/`c_col` set from geometry still understate the total node capacitance by roughly a
factor of two.

And the row-to-column term is a different kind of error entirely: it is a direct feedthrough
from a **driven row line to a sensed column line**, i.e. crosstalk, not a shunt load. A shunt
slows settling; a feedthrough injects a wrong charge into the readout. Two components:
`defaultoverlap allm2 metal2 allm1 metal1` = 133.86 aF/µm² over the `w_row · w_col` plate, and
`defaultsideoverlap allm2 metal2 allm1 metal1` = 67.05 aF/µm over the column rail's two edges
crossing the row rail. **The edge fringe is 7× the plate term** at these widths, which is the
same lesson as §7's fringe-dominates-area finding. The `2 · w_row` edge length is this
module's own accounting rather than magic's, so treat that term as an estimate with the
geometry stated, not a parsed constant.

None of this touches the DC MAC accuracy the study measures, because a `.op` solve has no
capacitors in it. It matters for transient reads, and it means any future settling or
crosstalk study on this array cannot use `c_row`/`c_col` alone.

**Two honest caveats on the sidewall numbers.**

- The tech file gives **one** sidewall constant per layer and **no reference spacing**. magic
  scales coupling with separation inside a `sidehalo` (`sky130A.tech:5027`) and this model does
  not. Taking the constant at the drawn spacing is a *model choice*; it is the conservative
  one, and it becomes more conservative — i.e. more of an overestimate — as the pitch opens
  up. The ratios above are therefore upper bounds at the relaxed pitches.
- Adding the **full** perimeter-fringe term and the **full** sidewall term double-counts the
  same physical edge. A close neighbour shields the fringe field that would otherwise reach
  the substrate; magic has `fringeshieldhalo` (`sky130A.tech:5028`) for exactly this and this
  model does not. So the absolute totals are upper bounds too, and the **ratio** is the robust
  part of the finding: coupling and shunt are the same order, whatever the exact shielding.

---

## 8. The pitch/resistance loop: `g_min` sets everything

This is the design loop that decides which row of §9's table a real array lands on, and it
runs entirely against the cell, not the wire.

`CrossbarParams.g_min` defaults to 1 µS, i.e. **1 MΩ**. sky130's highest-sheet resistor is
`xhrpoly` at **319.8 Ω/□** (`sky130A.tech:5116`). So:

```
1 MΩ / 319.8 Ω/□                          = 3127 squares
3127 squares × 0.35 µm width              = 1094 µm of line
serpentined at 0.35 + 0.48 = 0.83 µm      = a 30.1 × 30.1 µm block (36 stripes)
```

The 0.35 µm width is the narrowest **binned** `res_xhigh_po` model the PDK ships
(`res_xhigh_po_0p35`), not the DRC minimum: `poly.3` allows 0.33 µm but there is no 0.33 µm
model, and a resistor you cannot simulate is not a design. The 0.48 µm stripe spacing is rule
`poly.9` (`sky130A.lydrc:298`), poly-resistor to poly. Both are parsed from the PDK, not
asserted.

**So a cell that physically holds one `g_min` resistor is ~30 µm on a side, and the cell pitch
cannot be smaller than that.** Which closes the loop: a 31 µm pitch puts 31 squares of metal1
rail in every cell, and 31 × 0.125 Ω/□ = **3.875 Ω/pitch** — four times the project's default
assumption, and 78% of the value it swept as its pessimistic case.

Widening the rail is the available lever and it is a linear one: at 3 µm the rail costs
1.29 Ω/pitch, at 5 µm 0.78 Ω/pitch. It is not free — a 5 µm rail in a 31 µm pitch spends 16%
of the cell's width on one of two rail families — but it is the difference between §9's
bottom row and its middle rows.

### Routing on local interconnect instead

At the `dense` 0.74 µm pitch, each layer at its own minimum width:

| layer | sheet R | min width | R per pitch | × metal1 |
| --- | --- | --- | --- | --- |
| met1 | 0.125 Ω/□ | 0.14 µm | 0.661 Ω | 1.0× |
| met2 | 0.125 Ω/□ | 0.14 µm | 0.661 Ω | 1.0× |
| li1 | **12.800 Ω/□** | 0.17 µm | **55.7 Ω** | 84.3× |

Local interconnect is **102.4× metal1 per square** and must be wider on top of that, which
nets 84.3× per pitch at minimum widths. `MATMUL.md` §3.2 measures 74.2% int8 match at
5 Ω/pitch *with* per-channel calibration; 55.7 Ω/pitch is an order of magnitude past that.
**Routing crossbar rails on local interconnect is not a design.** (The prior estimate of "~100×
worse, tens of ohms per pitch" is confirmed: 102.4× per square, 55.7 Ω/pitch at the dense
pitch, rising linearly with pitch.)

---

## 9. The payoff: the accuracy study on derived parasitics

The configuration is [`MATMUL.md` §3.2](MATMUL.md)'s exactly — `A (16,32) @ B (32,8)`, 16×16
tile, 0T1R, unipolar two-pass, batched, seed 0 — so the derived rows drop straight into the
published table. The assumed rows are re-run here as a control and **reproduce the published
numbers exactly**, including the worst-LSB column, which is the check that the re-run is the
same experiment and not a new one.

int8 outputs exactly correct, %:

| configuration | per-tensor raw | per-tensor + global gain | per-channel raw | **per-channel + per-channel gain** | worst LSB raw → cal | accum bits (rms) |
| --- | --- | --- | --- | --- | --- | --- |
| assumed `r_wire` = 0 | 100.00 | 100.00 | 100.00 | **100.00** | 0 → 0 | 48.63 |
| assumed `r_wire` = 0.25 | 92.97 | 98.44 | 89.06 | **99.22** | 1 → 1 | 11.54 |
| assumed `r_wire` = 1 | 74.22 | 90.62 | 71.88 | **94.53** | 2 → 1 | 9.55 |
| assumed `r_wire` = 5 | 32.03 | 62.50 | 26.56 | **74.22** | 7 → 1 | 7.26 |
| **derived** dense (0.289 / 0.661) | 90.62 | 97.66 | 84.38 | **96.09** | 1 → 1 | 10.54 |
| **derived** relaxed (0.5 / 0.5) | 86.72 | 96.09 | 82.03 | **96.88** | 1 → 1 | 10.55 |
| **derived** g_min-driven (3.88 / 3.88) | 35.94 | 68.75 | 32.81 | **76.56** | 6 → 1 | 7.62 |

The float64 engine on the same operands:

| configuration | rms rel error | max rel error | effective bits (rms) |
| --- | --- | --- | --- |
| assumed `r_wire` = 0 | 2.498e-15 | 3.288e-15 | 48.51 |
| assumed `r_wire` = 0.25 | 1.580e-03 | 1.960e-03 | 9.31 |
| assumed `r_wire` = 1 | 6.290e-03 | 7.800e-03 | 7.31 |
| assumed `r_wire` = 5 | 3.067e-02 | 3.800e-02 | 5.03 |
| **derived** dense | 2.182e-03 | 2.552e-03 | 8.84 |
| **derived** relaxed | 3.155e-03 | 3.913e-03 | 8.31 |
| **derived** g_min-driven | 2.394e-02 | 2.967e-02 | 5.38 |

**What this says about the project's assumption.** The default `r_row = r_col = 1.0` is not a
neutral placeholder — it is optimistic for any array that can actually be built. The
interconnect on its own is cheap: at a pitch tight enough to be interconnect-limited the rails
cost 0.29–0.66 Ω/pitch and the analog path keeps 10.5 accumulator bits. But the pitch is not
set by the interconnect. It is set by the `g_min` resistor, and once the cell is 30 µm wide the
rails cost 3.9 Ω/pitch and the path keeps 7.6 bits. The honest summary is that **this project's
accuracy is limited by `g_min` through the cell area, not by the wire directly** — and the
`5 Ω/pitch` column, which the study presented as a pessimistic bound, is closer to a realistic
sky130 design point than the `1 Ω/pitch` column it treated as nominal.

Per-channel calibration holds up: it never misses by more than one LSB at any derived value,
including the g_min-driven one, which is the same conclusion `MATMUL.md` reached at 5 Ω/pitch.

`c_row` and `c_col` are set on the derived rows and cannot move these numbers, because a `.op`
solve has no capacitors in it. What they change is settling — §7.

---

## 10. Limitations

Read these before quoting anything above.

1. **This is geometry + technology constants, not extraction.** Self R from sheet resistance
   and squares; C from area and perimeter constants. No field solver, no 3-D coupling matrix,
   no current crowding, no skin effect, no frequency dependence, no temperature dependence.
   The word "PEX" does not apply and is not used.
2. **The capacitance model is structurally incomplete, by the amount §7 measures.** Coupling to
   neighbouring rails is the same order as the modelled shunt to substrate, and row-to-column
   crossing capacitance has no counterpart in `crossbar.Crossbar` at all. Anything downstream
   of `c_row`/`c_col` inherits that. The row-to-column **edge-fringe** component additionally
   assumes the column rail presents `2 · w_row` of edge per crossing — that geometry is this
   work's accounting, not a parsed constant.
3. **Coupling capacitance has no spacing dependence here.** The tech file gives one sidewall
   constant per layer; magic scales it with separation inside a halo and this does not. The
   coupling figures are upper bounds, more so at wider pitches. Adding the full fringe and full
   sidewall terms also double-counts the same edge, since neither shields the other here.
4. **Interconnect only. There is no memory device, because sky130 has none.** The skeleton is
   rails, crossings and taps. The cell is represented by an empty gap and by its effect on the
   pitch. Nothing here characterizes a memristor.
5. **The skeleton is not a floorplan.** No select rail, no drivers, no sense amplifiers, no
   decode, no power grid, no fill, no antenna handling, no seal ring. A real array's pitch
   would be larger than §8's number, not smaller, and its rail R correspondingly worse.
6. **DRC ran with `FEOL = false`,** the deck's own default. Every metal, via and
   local-interconnect rule ran; no diffusion or poly rule did. The interconnect skeleton is
   fully covered by what ran. The §5 transistor is not.
7. **LVS matched one transistor, on `W` and `L` only.** KLayout's MOS4 comparison ignores
   `AS`/`AD`/`PS`/`PD`, so the drawn source/drain areas and perimeters are unchecked. And a
   match required a µm↔m adapter on the reference netlist, documented in §5 — the raw hdl21
   output does not match, and that is a real tool-interface incompatibility, not a fixed bug.
8. **No crossbar layout has been LVS'd,** because a crossbar layout with no device in it has
   nothing to match against.
9. **Nothing has been signed off or fabricated.** DRC clean against one deck at one setting is
   not tapeout sign-off: no double-patterning checks, no density or antenna sign-off, no
   metal-fill, no corner or reliability analysis. No silicon exists and none is planned here.
10. **The typical corner only.** Every number uses the `()` variants block. The tech file's own
    high-R block would scale metal resistances by 145/125 = 1.16× and the high-C block would
    scale metal1 area capacitance by 35.7/25.78 = 1.38×. Corner spread is not swept.
11. **The `g_min` serpentine is arithmetic, not a drawn layout.** §8 computes squares, length
    and block size from parsed sheet resistance, the PDK's binned device widths and the deck's
    `poly.9` spacing. No serpentine was drawn or DRC'd; corner squares, contact heads and the
    `rpm` marker layer would all make the real block somewhat larger.
