# A 6T SRAM bitcell in sky130: measured margins, and a cell the open PDK forbids

Everything else in this repository stores its weights in **conductance**. This section stores a
bit in a **latch**, which is a different problem with a different acceptance criterion: a
crossbar cell is judged by how linearly it turns a voltage into a current, a bitcell by whether
it holds its value while you look at it. So the deliverable is not a netlist — it is a set of
measured margins, and `verify_sram6t.py` asserts them.

Two results are worth stating before the details.

1. **The cell works, and read SNM is the number that says so.** 297 mV at `tt`, 216 mV at the
   worst of the three required corners (`ff`), 177 mV at the worst corner measured (`sf`), all
   at VDD = 1.8 V. Hold SNM is 691 mV at `tt`. Write margin is 977 mV at `tt` and never below
   883 mV. The stored 0 rises to 258 mV during a read against a 745 mV trip point.

2. **The drawn cell cannot be DRC clean, and the reason is the PDK's, not the layout's.** The
   open sky130 DRC deck forbids the geometry its own SRAM device models are characterized at:
   `difftap.1` wants diffusion ≥ 0.15 µm wide and two of the three devices exist only at
   W = 0.14 µm; `difftap.2` wants a gate ≥ 0.42 µm wide and the widest device here is 0.21 µm.
   Nothing else in the cell violates anything. The PDK's own bitcell, run through the same
   harness, trips **246 violations across 30 rules** — including those same two.

Run it:

```bash
uv run sram6t.py                     # devices, cell, margins, corners  (~4 min, ~24 ngspice runs)
uv run verify_sram6t.py              # 9 machine checks on the margins
uv run verify_sram6t.py --layout     # + 2 machine checks on the drawn cell (needs KLayout)
uv run sram_layout.py                # draw it, DRC with FEOL on, LVS, area  (~16 s)
```

---

## 1. The devices, and why there is no sizing exercise

sky130 ships three transistors specifically for SRAM. They are `.subckt` wrappers around a
binned BSIM3 card, and `sky130_hdl21` does not expose them — its `xtors` are only the standard
`nfet_01v8` / `pfet_01v8` families — so `sram6t.py` wraps them as `h.ExternalModule` with
`spicetype=SpiceType.SUBCKT`, the same move `cell_layout.py` makes for the poly resistor.

| device | role | W [µm] | L [µm] | \|Id\| at \|Vgs\| = \|Vds\| = 1.8 V | model bin (W, L in µm) |
| --- | --- | --- | --- | --- | --- |
| `sky130_fd_pr__special_nfet_latch` | pull-down | 0.21 | 0.15 | 91.406 µA | W 0.2095–0.2105, L 0.075–0.1505 |
| `sky130_fd_pr__special_nfet_pass` | access | 0.14 | 0.15 | 70.566 µA | W 0.1395–0.1405, L 0.075–0.1505 |
| `sky130_fd_pr__special_pfet_latch` | pull-up | 0.14 | 0.15 | 20.815 µA | W 0.1395–0.1405, L 0.1495–0.1505 |

Currents are one `.op` per device at the `tt` corner. The bin limits are parsed out of each
device's own `*.pm3.spice` by `sram6t.model_bins` — the card carries `binunit = 2.0`, so its
`lmin`/`lmax`/`wmin`/`wmax` are metres and the parse converts. `check_bins` requires each
geometry to land in **exactly one** bin and refuses otherwise.

The ratios that matter fall out of the foundry's choices rather than being tuned:

- **β = pull-down / access = 1.295** — read stability. The pull-down has to sink the access
  device's current without letting the stored 0 rise past the other inverter's trip point.
- **access / pull-up = 3.390** — write margin. The access device has to overpower the pull-up
  on the node being flipped.

### The "one bin" claim, corrected

The width really is single-valued: every W window above is 1 nm wide, i.e. one value plus
round-off. **The length is not.** The two NMOS bins accept L from 0.075 to 0.1505 µm, and the
PMOS has a *second* bin at L 0.0245–0.0805 µm (`sky130_fd_pr__special_pfet_latch.pm3.spice:188`)
that this cell does not use. So there is no width choice to make, but there is a length window;
0.15 µm is the length the foundry's own bitcell draws, and it is what this cell uses.

Leaving a bin is not a style violation — it removes the device. `sram6t.check_bins` asserts that
half the drawn width lands in no bin at all, and ngspice's answer to an out-of-bin W is
`could not find a valid modelname` and nothing else.

### The unit hazard, fourth instance — and this one is not where the grep says

[LAYOUT.md §5.1](LAYOUT.md) lists three silent SI-versus-micron failures in this toolchain.
This is the fourth, with a twist: **geometry in this deck is microns, and the `.option scale`
responsible is not in the file the deck names.**

`grep -n 'option scale' sky130.lib.spice` finds nothing. Grep the whole `libs.tech/ngspice`
tree and the option appears in exactly two places, `all.spice:2` and
`parameters/montecarlo.spice:2`, and the second is reachable only through the `.lib mc` section.
The path that actually applies it is one include deeper:

```
corners/tt.spice:30  .include "../all.spice"
all.spice:2          .option scale=1.0u
all.spice:65         .include ".../sky130_fd_pr__special_nfet_latch.pm3.spice"
```

`all.spice` is also the only file in the `tt` chain that defines the `special_*` subckts at all,
and `corners/{ss,ff,sf,fs}.spice:30` include it the same way. `sram6t.scale_provenance` walks
that chain and returns `1e-06, 'corners/tt.spice:30 -> all.spice:2'`, so the claim is a parse
rather than a comment, and it raises if a PDK revision breaks the chain.

The consequence for this module: nothing here goes through `sky130_hdl21.compile()`, so the
SI→µm scaling that `crossbar.si_literal` exists to trigger never fires, and the geometry has to
be written in microns by hand. `sram6t.um` does that and says why.

Asserted in both directions, because the positive half alone is weak:

| check | result |
| --- | --- |
| the three devices at their characterized geometry, written in microns | reproduce the currents above to ≤ 0.002 % |
| the *same* device with the *same* number as SI metres (`0.21 * µ`) | ngspice refuses it: `could not find a valid modelname` |

The negative control matters because a wrong scale factor produces *some* current whenever the
bin is wide enough to absorb it. These bins are 0.5 % wide, so they do not, and the control
proves it.

---

## 2. The cell

Six devices in one flat module, built out of three add-a-piece helpers rather than one
monolithic generator, so an 8T/10T variant is `storage_core` + `add_access` + a new read port
and not a rewrite:

```
sram_6t: ports bl, blb, wl, vdd, vss; 6 devices
  mnq    sky130_fd_pr__special_nfet_latch       d=q,   g=qb, s=vss, b=vss
  mpq    sky130_fd_pr__special_pfet_latch       d=q,   g=qb, s=vdd, b=vdd
  mnqb   sky130_fd_pr__special_nfet_latch       d=qb,  g=q,  s=vss, b=vss
  mpqb   sky130_fd_pr__special_pfet_latch       d=qb,  g=q,  s=vdd, b=vdd
  maq    sky130_fd_pr__special_nfet_pass        d=bl,  g=wl, s=q,   b=vss
  maqb   sky130_fd_pr__special_nfet_pass        d=blb, g=wl, s=qb,  b=vss
```

`add_inverter` builds one latch leg, `add_latch` calls it twice cross-coupled, `add_access`
adds one write port device. `build_sram6t` composes them into a flat, name-controlled module —
flat because LVS needs it that way, and named because the LVS deck pairs layout and schematic
top cells *by name*. `Sram6T` is the `@h.generator` front end used inside the testbenches, and
`build_half_cell` reuses the *same* `add_inverter` and `add_access` to build the butterfly
device under test, so the VTC measured is the cell's own leg and not a look-alike.

Bodies go to the rails — NMOS to `vss`, PMOS to `vdd` — which is what the drawn cell does too
(the pwell tap and the nwell tap), so the netlist and the layout agree on four terminals per
device rather than three plus an assumption.

---

## 3. Why the butterfly curve is a ramp, not a DC sweep

`vlsirtools` 7.0.0 cannot run a DC sweep on ngspice. Two independent breaks:

- `NgspiceNetlister.write_dc` (`netlist/spice.py:622`) emits `.dc param start=… stop=…
  step=…`, which is not ngspice syntax and drops `an.indep_name` — the swept source's name —
  entirely.
- `NgSpiceSim.parse_results` (`spice/ngspice.py:97`) looks the result up under
  `Plotname: DC Analysis`, where ngspice writes `DC transfer characteristic`.

Rather than patch two internals, `sram6t.vtc` ramps a source slowly under `hs.Tran`
(100 ns for 1.8 V, 6001 output points) and **checks the substitution instead of asserting it**:

```
x10 slower: SNM moves 404.3 uV (297.30 -> 296.90 mV); raw vertical VTC gap 27.76 mV
```

The acceptance number is the change in SNM, not the pointwise VTC gap, because *no* pointwise
gap is well conditioned on a VTC. Compared vertically, the high-gain region turns a sub-mV
difference in where the two runs' adaptive timesteps landed into tens of mV of apparent error;
compared horizontally, the two flat regions do the same in reverse, and worse. SNM is both the
quantity being reported and a functional of the whole curve, so its movement — 0.4 mV for a
10× change in ramp rate — is the error bar that means something.

### The SNM definition actually used

The butterfly is the measured VTC `qb = f(q)` together with its mirror `q = f(qb)`. In the
upper-left lobe `f` is the upper boundary and `f⁻¹` the lower one, and both fall with `q`, so an
axis-aligned square `[x₀, x₀+s] × [y₀, y₀+s]` fits **exactly** when

```
y₀ + s ≤ f(x₀ + s)      (top-right corner, where f is lowest over the interval)
y₀     ≥ f⁻¹(x₀)        (bottom-left corner, where f⁻¹ is highest)
```

Pushing both to equality makes `s` the root of `s = f(x₀+s) − f⁻¹(x₀)`, which is decreasing in
`s`; a bisection per `x₀` and a maximum over `x₀` is the answer, with no rotation bookkeeping.
The square's diagonal runs at +45° between the two curves, which is the same square Seevinck's
rotate-and-take-the-maximum-distance construction finds. The containment argument is exact for
these monotone boundaries, so `sram6t.square_fits` checks it rather than assuming it, and
`verify_sram6t` asserts three separate things about the solver's answer: the square fits, one
2 mV larger at the same corner does not, and both corners lie on the two curves (to 0.0 µV).

The cell's two legs are identical, so the mirror is an exact reflection about `q = qb` and the
two lobes are congruent. One number is reported rather than the same measurement twice.

---

## 4. Measured margins

All at VDD = 1.8 V, 27 °C, ngspice-47, `.lib sky130.lib.spice <corner>`.

### Static noise margin, `tt`

| condition | wordline | bitlines | SNM | VTC low level | VTC high | trip point | max gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hold | 0 V | both at 1.8 V | **690.8 mV** | 0.0 mV | 1.8000 V | 0.745 V | 24.8 |
| read | 1.8 V | both at 1.8 V | **297.3 mV** | 258.0 mV | 1.8000 V | 0.861 V | 17.9 |

Read SNM is 57 % below hold SNM, and the whole mechanism is in the "VTC low level" column: with
the wordline high, the access device sources current from a bitline at 1.8 V into the node
holding a 0, and the pull-down cannot hold it at the rail. It settles 258 mV up. That is why
read SNM is the number that limits the cell and hold SNM is decoration.

### Hold, as a transient

Write a 0, drop the wordline, hold both bitlines at 1.8 V, wait 200 ns — two orders of magnitude
longer than the write phase, so this is a statement about leakage through the off access devices
and not about write settling:

```
tt, 200 ns: q -0.019 -> 0.000 mV (max 0.017), qb 1.79999 -> 1.80000 V (min 1.79999);
            worst drift 19.1 uV
```

### Read stability / read upset

Wordline high, both bitlines precharged to 1.8 V, stored value must survive. No `.ic` anywhere:
every state is reached by driving the cell the way an array would, in a three-phase sequence —
write `q = 0`, hold, then raise the wordline.

| corner | stored 0 peaks at | other inverter's trip point | margin | agrees with the read VTC's low level to |
| --- | --- | --- | --- | --- |
| tt | 258.0 mV | 0.745 V | 486.5 mV | 17.6 µV |
| ss | 225.4 mV | 0.763 V | 537.3 mV | 20.3 µV |
| ff | 306.3 mV | 0.712 V | 406.1 mV | 18.8 µV |

The last column is the cheapest cross-check available here and it is not decoration: the
transient peak and the DC butterfly's low level are the same physical quantity reached by two
independent routes — a three-phase `.tran` on the full six-device cell, and a slow ramp on a
half cell with the loop broken. They agree to 20 µV.

### Write margin

Wordline high, `bl` held at 1.8 V, `blb` driven to a test voltage; the highest `blb` that still
flips the cell is the write-margin voltage, and `VDD −` it is the conventional margin. The
boundary is a discontinuous function of the bitline voltage — there is no partial write — so it
is a boundary hunt, done as three refinements of a 25-point grid. All 25 points live in **one**
deck as 25 independent copies of the cell, because ngspice spends ~11 s parsing this model
library and ~1 ms solving the circuit; a bisection would pay that 11 s a dozen times for one
number. Monotonicity across the whole grid is asserted, which is what makes the refinement
legitimate.

| corner | writes at blb ≤ | fails at | write margin | bracket |
| --- | --- | --- | --- | --- |
| tt | 0.8234 V | 0.8236 V | **976.6 mV** | 0.130 mV |
| ss | 0.7507 V | 0.7508 V | **1049.3 mV** | 0.130 mV |
| ff | 0.9174 V | 0.9176 V | **882.6 mV** | 0.130 mV |

`blb = 0` must write and `blb = VDD` must not; both ends are asserted rather than assumed,
because a cell that cannot be written at 0 V, or that flips with both bitlines high, is broken
in a way a boundary hunt would happily report a number for.

### Corners

`tt`, `ss` and `ff` are the required set. `sf` and `fs` are added for read SNM only, because
**neither `ss` nor `ff` is a skewed corner** — they move NMOS and PMOS the same way — and the
skewed ones are where read SNM is expected to be worst. What the letters mean is measured, not
read off the name:

| corner | I pull-down | I access | I pull-up | β | acc/pu | read SNM | hold SNM | read low | trip | gain | write V | WM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tt | 91.4 µA | 70.6 µA | 20.8 µA | 1.295 | 3.390 | **297.3 mV** | 690.8 mV | 258.0 mV | 0.861 V | 17.9 | 0.8234 V | 976.6 mV |
| ss | 61.8 | 47.2 | 11.3 | 1.310 | 4.192 | **335.5 mV** | 719.5 mV | 225.4 mV | 0.861 V | 21.3 | 0.7507 V | 1049.3 mV |
| ff | 120.4 | 95.4 | 30.6 | 1.262 | 3.112 | **216.2 mV** | 627.1 mV | 306.3 mV | 0.867 V | 9.3 | 0.9174 V | 882.6 mV |
| sf | 119.6 | 94.7 | 11.3 | 1.263 | 8.365 | **177.2 mV** | — | 258.3 mV | 0.845 V | 13.7 | — | — |
| fs | 62.1 | 47.5 | 30.3 | 1.307 | 1.566 | **402.6 mV** | — | 250.3 mV | 0.921 V | 21.1 | — | — |

Worst read SNM: **177.2 mV at `sf`**, 40 % below `tt`. Worst among the three required corners:
216.2 mV at `ff`, 27 % below `tt`.

The drive-current columns explain the table and are why they are in it. **β barely moves with
corner** (1.262–1.310) because both devices in it are NMOS, so the read low level barely moves
either — 225 to 306 mV across five corners. What the corners actually move is the pull-up, from
11.3 µA at `sf` to 30.6 µA at `ff`, and the weakest pull-up is the worst read SNM. `sf` is
therefore fast-NMOS / slow-PMOS as these devices see it; that is measured, and it is the corner
the required three cannot show.

Read SNM at the worst corner measured is 177 mV, i.e. positive with a factor of ~1.8 over the
100 mV threshold `verify_sram6t` asserts. The cell is not marginal. What this does **not** cover
is device mismatch, which is what actually sets bitcell yield — see §8.

---

## 5. The drawn cell

`sram_layout.py` draws the six devices with `klayout.db`. Every dimension comes from a DRC rule
parsed off the deck line that emits it (42 of them), except `IMPLANT_ENC = 0.125 µm`, which this
deck does not check at all — it checks implant *width* only — and which is drawn anyway because
LVS derives `ngate` as `nsdm.and(tgate)` and an unimplanted gate is not a transistor.

```
sram_6t: 5.310 x 3.980 um = 21.13 um^2, 6 devices
  mnq    pull-down  W/L = 0.21/0.15 um, gate at x = 2.200, y = 1.270..1.480
  mpq    pull-up    W/L = 0.14/0.15 um, gate at x = 2.200, y = 2.855..2.995
  mnqb   pull-down  W/L = 0.21/0.15 um, gate at x = 3.130, y = 1.270..1.480
  mpqb   pull-up    W/L = 0.14/0.15 um, gate at x = 3.130, y = 2.855..2.995
  maq    access     W/L = 0.14/0.15 um, gate at x = 0.730, y = 1.305..1.445
  maqb   access     W/L = 0.14/0.15 um, gate at x = 4.600, y = 1.305..1.445

islands (diffusion), left to right:
  accL    x 0.135..1.325, heads y 1.245..1.505, channel 0.14 um, contacts bl@0.265, q@1.195
  latchN  x 1.605..3.725, heads y 1.245..1.505, channel 0.21 um, contacts q@1.735, vss@2.665, qb@3.595
  accR    x 4.005..5.195, heads y 1.245..1.505, channel 0.14 um, contacts qb@4.135, blb@5.065
  latchP  x 1.605..3.725, heads y 2.795..3.055, channel 0.14 um, contacts q@1.735, vdd@2.665, qb@3.595
nwell (1.415, 2.605, 3.915, 3.990), n+ tap (1.605, 3.380, 3.725, 3.800),
p+ tap (0.020, 0.135, 5.310, 0.555)
```

Bottom to top: p+ tap and the `vss` met1 rail; the wordline poly bar and its contact stack;
the NMOS band (both access devices and the cross-coupled pull-down pair); the li1 cross-couple
channel; the pull-up pair in an nwell; the n+ tap and the `vdd` met1 rail. `bl` and `blb` are
met1 stubs at the two ends, `q` and `qb` are met1 columns linking the NMOS and PMOS bands, and
the `vss` and `vdd` rails each send one met1 spur to the latch's shared source.

### Two things the geometry is forced into, and they set the area

**The dog-bone.** A 0.14 or 0.21 µm device cannot hold its own contact: `licon.5` wants 0.04 µm
of diffusion around a 0.17 µm cut, so the diffusion must be 0.25 µm across wherever a contact
lands. Hence a neck at the channel width under each gate and a taller head at each contact. And
the neck cannot flare immediately: `poly.7` (min source/drain length, 0.25 µm) reads a diffusion
step within 0.25 µm of the gate as an edge facing it, so the flare has to be a full source/drain
length away. That is **0.26 µm on each side of every gate, eight of them across the row,
2.08 µm — 39 % of the cell width.**

**The cross-couple channel.** Two nets have to swap sides on one interconnect layer. The trick
is two li1 tracks: the lower one carries `q` from its met1 column to the *other* inverter's
gate contact, the upper one carries `qb` the other way, and each track's li1 crosses the other
gate's poly with no contact between them. The npc rectangles around the two poly contacts are
merged into one, because `npc.2` wants 0.27 µm between two of them and the tracks are closer
than that — the deck's own rule text says to merge in that case. Total: **1.29 µm of the
3.98 µm height**, because two li1 tracks plus npc enclosure plus npc-to-gate clearance do not
fit in less.

The written file is measured back before anything else runs: `verify_layout` intersects `poly`
and `diff` — which is exactly what both decks call the gate (`gate = diff & poly`,
`sky130A.lydrc:202`) — and asserts the six (L, W) pairs against the netlist's, in microns. A
units slip anywhere in the drawing path shows up as a factor of 1000 there rather than as a
mysterious LVS parameter mismatch:

```
6 gates, (L, W) = [(0.15, 0.14), (0.15, 0.14), (0.15, 0.14), (0.15, 0.14),
                   (0.15, 0.21), (0.15, 0.21)]
bbox (0.01, 0.01, 5.32, 3.99) um, diff 1.333 um^2, nwell 3.462 um^2
```

Read as (x extent, y extent), not (shorter, longer): W = 0.14 µm is *narrower* than
L = 0.15 µm, so sorting by size would silently transpose four of the six devices.

---

## 6. DRC with FEOL enabled — and the rules that block it

The stock deck has `FEOL = false`, which would make "clean" close to meaningless for a
front-end cell. `cell_layout.feol_deck` writes a patched copy with `FEOL = true` (the PDK copy
stays untouched) — 381 rule categories instead of 145.

```
cell, FEOL off   145 rules  DRC CLEAN: cell.gds, 145 rules, 1.22 s
cell, FEOL on    381 rules  DRC FAIL:  cell.gds, 16 violations [difftap.1 x4, difftap.2 x12]
```

**The back end is clean. The front end trips exactly two rules and nothing else**, which
`sram_layout.main` asserts: a rule beyond those two is a drawing error and must not be allowed
to hide behind them.

| rule | what it wants | what the cell has | count |
| --- | --- | --- | --- |
| `difftap.1` | diff/tap width ≥ 0.15 µm | four channels at 0.14 µm (2 access, 2 pull-up) | 4 |
| `difftap.2` | gate width ≥ 0.42 µm, or ≥ 0.36 µm inside `areaid.sc` | six gates at 0.21 or 0.14 µm | 12 (2 edges/gate) |

Neither is fixable by redrawing. The only way out of `difftap.1` or `difftap.2` is a wider
device, and a wider device **has no model** — the bins in §1 are 1 nm wide in W, and ngspice
answers an out-of-bin W with `could not find a valid modelname`. So the open PDK's DRC rules and
its own SRAM device models are mutually inconsistent, and no third party can satisfy both.

The deck does have cell-name exemptions, so the mechanism exists:
`not_in_cell1 = layout(source.cell_obj).select("s8cell_ee_plus_sseln_a", …)`
(`sky130A.lydrc:264`) gates `difftap.2`. It names specific `s8*` macros, not a bitcell, and the
open PDK ships no waiver a third party can invoke. Nothing here disables a rule to make the
number look better.

### The harness is proven on the front end

A DRC harness that has only ever run on layouts it fails is not a tested harness. The cell
already trips two front-end rules for reasons no drawing can fix, so the planted violation has
to be *distinguishable*: `bad_cell_gds` narrows every gate to 0.14 µm, below `poly.1a`'s
0.15 µm, and changes nothing else.

```
planted violation: every gate narrowed to 0.14 um, poly.1a wants 0.15
  FEOL off      DRC CLEAN: badcell.gds, 145 rules, 0.80 s
  FEOL on       DRC FAIL:  badcell.gds, 22 violations [difftap.1 x4, difftap.2 x12, poly.1a x6]
```

236 extra rule categories with FEOL on; a front-end narrowing that is invisible with FEOL off
and caught with it on; a clean back end in both cases. The harness sees the front end.

---

## 7. LVS — and precisely what the match covers

Layout against the Stage-A netlist, using `sky130.lvs` (not `sky130.lylvs`, which is a wrapper
whose only content is a *commented-out* include and checks nothing —
`layout_oracle.default_lvs_deck` says so). Run with `-rd lvs_sub=vss`, so the deck's global
substrate net carries the cell's own name for it.

```
Stage-A netlist verbatim (special_* devices)     LVS NO MATCH: ERROR : Netlists don't match
with the device classes the extractor emits      LVS MATCH:    Congratulations! Netlists match.
device classes in the extracted netlist:         nfet_01v8 x4, pfet_01v8 x2
```

**The first line is a finding, not a failure, and `verify_sram6t` asserts it.** `sky130.lvs` has
no `extract_devices` call for `special_nfet_latch`, `special_nfet_pass` or
`special_pfet_latch`; grep it and the only `special_*` MOS it knows is `special_nfet_01v8`, in
the netlist *reader*'s model list at `:212`, not in an extraction rule. Nor is there a marker
layer that would key one — physically these are ordinary 1.8 V transistors, separately
characterized in the bitcell's context. So the extractor necessarily emits `nfet_01v8` and
`pfet_01v8`, and a comparison against the Stage-A netlist verbatim cannot match on device class.
`substituted_netlist` renames the classes to what the extractor emits and reruns; that is a
second experiment, not a fix, and running both is what separates "the deck cannot see these
models" from "the layout is the wrong circuit".

The extracted netlist, verbatim:

```
.SUBCKT sram_6t bl blb vdd vss wl
XM$1 \$4 \$5 vdd vdd sky130_fd_pr__pfet_01v8 L=0.15 W=0.14 AS=0.104  AD=0.0702 PS=1.56 PD=1.04
XM$2 vdd \$4 \$5 vdd sky130_fd_pr__pfet_01v8 L=0.15 W=0.14 AS=0.0702 AD=0.104  PS=1.04 PD=1.56
XM$3 \$5 wl blb vss sky130_fd_pr__nfet_01v8 L=0.15 W=0.14 AS=0.104  AD=0.104  PS=1.56 PD=1.56
XM$4 \$4 \$5 vss vss sky130_fd_pr__nfet_01v8 L=0.15 W=0.21 AS=0.1222 AD=0.0884 PS=1.56 PD=1.04
XM$5 vss \$4 \$5 vss sky130_fd_pr__nfet_01v8 L=0.15 W=0.21 AS=0.0884 AD=0.1222 PS=1.04 PD=1.56
XM$6 bl wl \$4 vss sky130_fd_pr__nfet_01v8 L=0.15 W=0.14 AS=0.104  AD=0.104  PS=1.56 PD=1.56
.ENDS sram_6t
```

### What the match compares, measured one broken thing at a time

[LAYOUT.md §5](LAYOUT.md) records a MATCH that was far weaker than the word implies. Rather
than read the deck source and trust it, `lvs_probes` changes exactly one thing at a time and
reruns:

| probe | verdict | what it establishes |
| --- | --- | --- |
| reference, unchanged | MATCH | — |
| W of one device × 1.5 | NO MATCH | W is compared |
| L of one device × 1.5 | NO MATCH | L is compared |
| W of one device × 1.001 (0.1 % high) | NO MATCH | the W tolerance is tighter than 0.1 % |
| W of one device × 1.0001 (0.01 % high) | NO MATCH | …and tighter than 0.01 %: effectively exact |
| nonsense `AS`/`AD`/`PS`/`PD` on one device | **MATCH** | **source/drain area and perimeter are compared against nothing** |
| `nfet_01v8` → `nfet_01v8_lvt` on one device | NO MATCH | the device class is compared |
| cross-coupling broken on one leg | NO MATCH | the connectivity is compared |
| `bl` and `blb` swapped | **MATCH** | the comparison is graph-based: this cell is symmetric under (bl,q) ↔ (blb,qb), so a swapped pair is a genuine isomorphism, not a miss |

So a match here establishes: the six devices' graph, the five ports, the device class, and W and
L per device essentially exactly. It does **not** establish the drawn source/drain area or
perimeter — KLayout's MOS4 comparer ignores `AS`/`AD`/`PS`/`PD`, the same hollow spot §5 of
LAYOUT.md records, now demonstrated rather than cited. And it does **not** establish the binding
to the SRAM-specific models Stage A actually simulated. A match says the layout is the right
circuit built out of the right-sized transistors, and no more than that.

---

## 8. Area, honestly

This PDK install ships no single-port 6T bitcell, so the widely quoted ~1 µm² sky130 6T figure
cannot be measured here and is deliberately not asserted. What it does ship, inside the SRAM
macros' GDS, is the OpenRAM **dual-port, eight-transistor** cell. `foundry_bitcell` cuts it out
into its own GDS — a copy; the PDK stays read-only — so the same harness measures it.

| cell | size | area | devices | ports |
| --- | --- | --- | --- | --- |
| `sram_6t`, drawn here | 5.310 × 3.980 µm | **21.13 µm²** | 6 | 1 |
| `sky130_fd_bd_sram__openram_dp_cell` | 3.960 × 3.035 µm | **12.02 µm²** | 8 | 2 |

The drawn cell is **1.76× the foundry cell's area while holding two fewer transistors and one
fewer port**: 3.52 µm² per device against 1.50, so the foundry cell is 2.3× denser per
transistor.

**That gap is the price of obeying rules the foundry's own cell does not obey.** The same
harness on the same deck:

```
foundry cell, FEOL off   69 violations [ct.4 x3, li.1 x3, li.3 x37, li.5 x4, m1.4 x17,
                                        m2.4 x4, via.4a x1]
foundry cell, FEOL on   246 violations across 30 rules [ct.4 x3, difftap.1 x2, difftap.10 x1,
   difftap.2 x26, difftap.4 x4, difftap.9 x4, hvtp.3 x4, li.1 x3, li.3 x37, li.5 x4,
   licon.11 x20, licon.11c x6, licon.14 x5, licon.15 x12, licon.17 x6, licon.4 x8,
   licon.5 x20, licon.7 x2, licon.8 x10, licon.8a x1, m1.4 x17, m2.4 x4, poly.10 x12,
   poly.12 x4, poly.2 x1, poly.4 x4, poly.5 x8, poly.7 x6, poly.8 x11, via.4a x1]
```

246 violations across 30 rules, against this cell's 16 across 2 — and `difftap.1` and
`difftap.2` are among them, the same two rules for the same reason, because those are properties
of the device widths and the foundry uses the same widths. It also breaks the two rules that set
this cell's size: `licon.5` (×20) and `poly.7` (×6), i.e. it contacts these devices without
paying either the 0.25 µm of diffusion around a cut or the 0.25 µm of source/drain before the
flare. That is where the 1.76× lives.

Two caveats, stated rather than buried:

- It is an 8T dual-port cell, not a 6T, so the transistor counts differ. That works *against*
  the drawn cell, not for it.
- Cutting one cell out of an abutting array removes the neighbours that would have supplied its
  boundary geometry, so its interconnect and well-spacing counts (`li.3 ×37`, `m1.4 ×17`,
  `difftap.9 ×4`) are inflated by the extraction. The intra-device rules — `difftap.1`,
  `difftap.2`, `licon.5`, `poly.7`, `poly.8` — are not affected by that, and they are the ones
  this comparison rests on.

---

## 9. Verification

`uv run verify_sram6t.py` — nine checks on the margins. `--layout` adds two on the drawn cell
(needs the KLayout application). Every margin check reads a single `sram6t.Margins` measured
once rather than re-simulating: at ~11 s of library parsing per ngspice invocation, per-check
simulation would cost half an hour to produce the same numbers.

| # | check | gate |
| --- | --- | --- |
| 1 | bins | each geometry in exactly one bin; every W window a single value; half the width in no bin at all |
| 2 | units | the corner's include chain sets `scale = 1e-6`; the three currents match their references to 0.1 %; SI-metre geometry is refused with a modelname error |
| 3 | netlist | six devices, the right three types, the cross-coupling, the access pair, and `w='0.21' l='0.15'` in the emitted text with no `2.1e-07` anywhere |
| 4 | quasi-static | a 10× slower ramp moves SNM by < 5 mV (measured: 0.4 mV) |
| 5 | butterfly | the square fits, one 2 mV larger does not, both corners lie on the curves |
| 6 | read SNM | > 100 mV at all five corners; the low level is off the rail at each; read SNM < hold SNM |
| 7 | hold | q below 0.1·VDD and qb above 0.9·VDD throughout; drift < 1 mV over 200 ns |
| 8 | read stability | no flip; ≥ 50 mV below the trip point; the transient peak equals the read VTC low level to < 1 mV |
| 9 | write margin | monotone boundary; margin > 300 mV; bracket < 1 mV; `blb = VDD` does not write |
| 10 | layout | the drawn channels are the simulated widths; back end clean; FEOL trips `difftap.1` ×4 and `difftap.2` ×12 and *only* those; the planted narrowing is invisible with FEOL off and caught with it on |
| 11 | lvs | the Stage-A netlist does not match verbatim; the substituted one does; 4 `nfet_01v8` + 2 `pfet_01v8` extracted; all nine probes as expected |

Ends in `VERIFY: PASS`, nonzero exit on failure.

---

## 10. Limitations

1. **No mismatch, and mismatch is what sets bitcell yield.** Every number here is a nominal
   corner. Real bitcell design is a 6σ problem in random dopant fluctuation: the models ship
   `__mismatch.corner.spice` parameters and an `mc` library section with `mc_pr_switch=1`, and
   none of it is exercised. A 177 mV nominal read SNM at `sf` says the topology and the ratios
   are right; it says nothing about the σ of the distribution, which is the number a memory
   compiler would be judged on.
2. **`AS`/`AD`/`PS`/`PD` are left at the subckt defaults, i.e. zero.** ngspice warns
   `Pd = 0 is less than W`. Junction area and perimeter are therefore zero and the junction
   capacitance is understated. This does not touch the DC margins (SNM, the trip points, the
   write boundary), which is where every headline number comes from; it does slightly speed up
   the transients. The defaults were kept because they are what the reference drive currents in
   §1 were measured with, and inventing geometry would break that chain. The layout's real
   values — `AS = 0.104`, `PS = 1.56` and so on — are in §7's extracted netlist and are exactly
   the parameters LVS proved it does not compare.
3. **No array, and therefore no bitline capacitance, no wordline RC, no sense amplifier and no
   precharge.** Both bitlines are ideal voltage sources throughout. That makes the write margin
   an upper bound and the read a best case: a real read discharges a loaded bitline through the
   access device, and the wordline pulse would be set by an RC the cell does not see here.
4. **The wordline is a poly rail.** Fine for one cell and for LVS; a real array would strap it
   on met2 or met3, since poly at ~48 Ω/□ across even sixteen cells is not a wordline. `bl` and
   `blb` are met1 stubs rather than columns that abut, so the drawn cell is **not tileable** as
   it stands.
5. **The DRC result is a diagnosis, not a clean cell.** §6's two rules block it and no drawing
   fixes them. Anyone who needs a clean SRAM cell in the open sky130 flow has to either obtain
   a bitcell-specific waiver or use devices whose characterized widths the deck permits — which
   means giving up the SRAM-specific models and the ratios they come with.
6. **LVS compares a substituted netlist.** The device classes had to be renamed to the ones the
   extractor emits, because the deck has no rule for the `special_*` devices. The match is
   therefore about topology and W/L, not about the model binding — §7 measures exactly how far
   it goes, and it does not go further than that.
7. **The butterfly comes from a slow ramp, not a DC sweep**, because `vlsirtools`' ngspice DC
   path is broken in two places (§3). Quantified rather than assumed: 0.4 mV of SNM movement per
   decade of ramp rate.
8. **One temperature (27 °C) and one supply (1.8 V).** No VDD scaling, which is the axis SNM
   collapses along and the one a low-power design would care about most.
9. **The area comparison is against an 8T dual-port cell**, because this PDK install ships no
   6T single-port bitcell. §8 states the caveat and it works against the drawn cell rather than
   for it.
