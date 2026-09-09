# analog-chip — an Hdl21 compute-in-memory crossbar generator

`crossbar.py` is a parameterized [Hdl21](https://github.com/dan-fritchman/Hdl21) generator for
an analog compute-in-memory (CIM) crossbar — a resistive-memory array that performs a
matrix–vector product in the analog domain — together with the testbenches and cross-checks
that make its results trustworthy. The defining idea is that the **weight matrix is a
generator parameter and the SPICE netlist is a compile output**: programming a weight is an
elaboration-time act, so every distinct cell conductance becomes its own subcircuit. Wire
parasitics are stamped as real devices (one series resistor per cell pitch on every row and
column, plus an optional shunt capacitor per pitch), so IR drop and RC settling fall out of
the simulator instead of out of a spreadsheet. Every operating-point result is cross-checked
against an independent nodal-analysis solve written directly in numpy.

## Scope

**This repository is a netlist generator and a simulation study. It is not a chip, and there
is no layout.**

What is here:

- Hdl21 generators for the cell, the array, and a signed differential tile (`Cell` →
  `Crossbar` → `Tile`).
- Testbenches that emit ngspice decks, run them, and parse the results back into numpy.
- An operating-point and transient study of three effects: wire IR drop, access-device series
  resistance, and wire RC settling.
- Two access-device routes: an uncalibrated placeholder `.model`, and the real sky130
  `sky130_fd_pr__nfet_01v8` via `sky130_hdl21.compile()`.
- An independent numpy MNA solve of the same topology, used as a correctness oracle.

What is **not** here — do not read any of it into the numbers below:

- **No layout.** No floorplan, no GDS, no place-and-route, no routing, no DRC or LVS, no
  parasitic extraction, no tapeout, no silicon. Nothing in this repository has a physical
  dimension other than the access transistor's W and L.
- **No extracted parasitics.** `r_row` / `r_col` / `c_row` / `c_col` are per-pitch values
  *you assume*, not values back-annotated from a physical implementation.
- **No memristor device physics.** The cell is a linear resistor whose value is set by the
  weight. There is no nonlinearity, drift, read noise, cycle-to-cycle or device-to-device
  variation, and no write/programming dynamics.
- **No peripheral circuits.** No DAC, no ADC, no sense amplifier, no transimpedance
  amplifier. Columns are held at virtual ground by ideal 0 V voltage sources — a perfect TIA
  with infinite gain and bandwidth and no noise.
- **No inference-accuracy study.** One stimulus vector and one random weight matrix, not a
  statistical campaign over networks, corners, or device populations.

`docs/REPORT.md` has a full [limitations
section](docs/REPORT.md#limitations-and-threats-to-validity); read it before quoting any
number here.

## The array

![32x32 crossbar array with distributed wire resistance, drivers at the column-0 end of each row and readout at the last-row end of each column](docs/diagrams/crossbar-array.svg)

Rows are driven at the column-0 end; columns are read out at the last-row end. That asymmetry
is deliberate — it is what produces a spatial IR-drop gradient instead of a uniform offset.
See [`docs/DESIGN.md`](docs/DESIGN.md) for the generator hierarchy and every parameter.

## Setup

```bash
uv sync                 # hdl21 7.0.0, sky130-hdl21 7.0.0, numpy, matplotlib
brew install ngspice    # tested on ngspice-47
```

Optional, and only needed for the `access="sky130"` route:

```bash
uv tool install volare
volare enable --pdk sky130 <version>    # installs under ~/.volare by default
```

The PDK is discovered from `$PDK_ROOT`, then `$VOLARE_ROOT`, then `~/.volare`, accepting both
volare's `<root>/sky130A` symlink and the `.../sky130/versions/<hash>/sky130A` behind it.
Everything except the two sky130 checks runs with no PDK installed; those two skip themselves,
passing.

## Run

```bash
uv run crossbar.py      # 32x32 MAC sweep vs. wire resistance
uv run verify_mna.py    # 9 machine checks, incl. an independent numpy MNA solve
```

`crossbar.py` output, verbatim:

```
32x32, r_wire=0.0 ohm/pitch: mean err   0.00%  worst column  -0.00%  |  after gain fit (a=1.0000): residual rms  0.00%  worst  0.00%
32x32, r_wire=1.0 ohm/pitch: mean err  -3.59%  worst column  -4.72%  |  after gain fit (a=0.9637): residual rms  0.84%  worst  1.87%
32x32, r_wire=5.0 ohm/pitch: mean err -15.50%  worst column -20.01%  |  after gain fit (a=0.8433): residual rms  3.83%  worst  8.70%
```

`verify_mna.py` ends in `VERIFY: PASS` and exits nonzero if any check fails.

## Headline results

All from a 32×32 array, `g_min` = 1 µS, `g_max` = 100 µS, 0T1R cells, `x` uniform in
[0, 200 mV], seed 0, ngspice-47. Full methodology and error bars in
[`docs/REPORT.md`](docs/REPORT.md).

| result | number | why it matters |
| --- | --- | --- |
| ngspice vs. an independent numpy MNA solve | max relative difference **6.6e-12** | two independent solvers, one topology: the netlist is what we think it is |
| IR drop at 1 Ω/pitch | gain droop **−3.59%**, residual after one scalar gain fit **0.84% rms** | most of the error is a calibratable gain, not lost accuracy |
| IR drop at 5 Ω/pitch | gain droop **−15.50%**, residual **3.83% rms / 8.70% worst** | the residual is ~4× smaller than the headline error |
| sky130 access FET at read bias | Ron **844 Ω** = **8.4%** of the 10 kΩ `g_max` cell, **0.08%** of the 1 MΩ `g_min` cell | a per-cell nonlinearity, not a uniform gain: whether a scalar gain removes it depends on the weight distribution |
| array RC pole at 5 Ω and 0.2 fF per pitch | τ₆₃ = **0.69 ps** | the readout amplifier sets read time, not the wire |
| signed differential tile at zero wire R | recovers `(g_max − g_min)·Wᵀx` to **2.0e-15** | the `g_min` offset cancels exactly, as designed |

![Column current error vs. wire resistance per cell pitch, showing that IR drop is strictly one-sided](docs/figures/ir-drop-vs-rwire.png)

![Access-FET Ron compresses the top of the conductance range far harder than the bottom](docs/figures/ron-compression.png)

## Files

| file | role |
| --- | --- |
| `crossbar.py` | `Cell` → `Crossbar` → `Tile` generators, MAC and settling testbenches, error metrics, the sweep in `__main__` |
| `verify_mna.py` | nine machine checks, including the independent numpy MNA solve |
| `ngspice_compat.py` | ngspice ≥ 43 rawfile-header shim for vlsirtools 7.0.0; see below |
| `docs/DESIGN.md` | generator hierarchy, the differential-pair derivation, full parameter reference |
| `docs/REPORT.md` | methodology, verification, results, limitations, reproduction |
| `docs/diagrams/` | schematic diagrams (SVG) for the cell, array, tile, and testbench |
| `docs/figures/` | measured result plots (PNG) |

## Cell configurations

`CellParams.passive=True` is 0T1R — the memristor conductance alone. The default
(`passive=False`) is 1T1R with a series access NMOS, and `CellParams.access` picks the device
model:

| `access` | device | needs a PDK? | calibrated? |
| --- | --- | --- | --- |
| `"generic"` (default) | `.model xbar_nmos_generic nmos level=1 ...` via `h.ExternalModule` | no | **no — placeholder** |
| `"sky130"` | `sky130_fd_pr__nfet_01v8` via `sky130_hdl21.compile()` | yes | yes, tt corner |

**The generic model is a placeholder and is not silicon-calibrated.** It is a level-1 card
with round numbers (`vto=0.5 kp=120u lambda=0.05`) whose only job is to let the 1T1R topology
simulate with no PDK installed — that is, to exercise plumbing. Do not read performance off
it. For scale: its implied Ron at W/L = 1 µm / 180 nm measures **1162.0 Ω** against a level-1
hand estimate of 1153.8 Ω, where the real sky130 `nfet_01v8` in the same testbench comes out
at **818.9 Ω**. The placeholder is 42% pessimistic on Ron. Right order of magnitude, nothing
more.

## The sky130 units gotcha

**This project's public API is SI units everywhere.** `w_acc` and `l_acc` are metres on both
access routes, and the shipped defaults (`w_acc = 1 * µ`, `l_acc = 180 * n`) simulate correctly
as-is. Getting there required working around an upstream `sky130-hdl21` bug, and the workaround
is load-bearing.

The sky130 ngspice library sets `.option scale=1.0u` (via `libs.tech/ngspice/all.spice`), so raw
deck geometry is in **microns**. `sky130-hdl21` 7.0.0 is supposed to convert, in
`Sky130Walker.scale_param` (`sky130_hdl21/pdk_logic.py:351`), but it type-dispatches and silently
skips the conversion for `h.Prefixed`:

```python
if isinstance(orig, h.Prefixed):
    return orig  # FIXME: where's the scaling?
if isinstance(orig, h.Literal):
    return h.Literal(f"({orig.text} * 1e6)")
```

The `FIXME` is upstream's own. A plain `1 * µ` is an `h.Prefixed`, so it would reach ngspice as
`w=1e-06` micron — a 1 pm device — miss every binned `.model`, and fail with nothing but
`could not find a valid modelname`. `si_literal()` converts `Prefixed` → `Literal` on the way
in so the scaling fires, and the deck ends up carrying `w='(1e-06 * 1e6)'` = 1.0 µm.
Consequences:

- Pass SI metres. `w_acc=1.0` means one metre, reaches ngspice as `w=1e+06` micron, and dies
  with the same `could not find a valid modelname`.
- Confirmed by physics, not just by string matching: sweeping `w_acc` over 0.5 µm / 1 µm /
  2 µm moves the measured Ron 1671.4 Ω / 818.9 Ω / 365.1 Ω — a real 1/W trend, which a unit
  slip of 10⁶ could not produce.
- A caveat about the upstream library's own defaults, not about this API: every entry in
  `sky130_hdl21.default_xtor_size` is a `Prefixed` pair (`nfet_01v8` is `0.42 * MICRO`,
  `0.15 * MICRO`), so those defaults take the unscaled path and do *not* simulate against the
  stock ngspice models.

Two checks pin it: `sky130 wiring` asserts the deck literally contains `w='(1e-06 * 1e6)'`,
and `sky130 sim` asserts the measured Ron lands in 300–3000 Ω. A unit slip is a 10⁶× error, so
no plausible device survives that band.

One further sharp edge: `sky130_hdl21.Install` is a process-wide singleton whose
`__post_init__` refuses to be replaced, so `sky130_install()` caches the first PDK it finds for
the life of the process.

## Why `ngspice_compat.py` exists

vlsirtools 7.0.0's `parse_nutbin` reads the ngspice rawfile header *by line count*, assuming
`Title:` / `Date:` / `Plotname:` before `Flags:`. Modern ngspice (confirmed on ngspice-47)
emits an extra `Command:` line, so the parser consumes `Command:` as the plotname and then
trips on `Plotname:` where it expects `Flags:`:

```
ValueError: Invalid flags ['Plotname:', 'Operating', 'Point']
```

The shim rebinds `parse_nutbin` to scan forward to the `Plotname:` line instead of counting,
which works on both old and new ngspice. vlsirtools 7.0.0 is the latest release, so there is
no upstream fix to take instead. Delete this file if a future vlsirtools fixes the parser.

## Verification

`uv run verify_mna.py` runs nine checks and exits nonzero if any fails.

| check | what it asserts |
| --- | --- |
| mna | ngspice column currents vs. a nodal system built directly in numpy |
| op regression | the `.op` error magnitudes above, pinned to 2 dp against drift |
| error metrics | droop is one-sided, and the closed-form best-fit gain matches `lstsq` |
| 1T1R generic | the *default* cell simulates; its Ron matches the level-1 estimate |
| sky130 wiring | `compile` → `nfet_01v8`, `.lib ... tt` from `Install.include`, SI geometry scaled to µm, and a clear actionable error when no PDK is found |
| sky130 sim | the 1T1R `.op` against the real PDK models at default geometry, with Ron in a physical band |
| 1T1R compression | Ron compression is weight-structured: a scalar gain absorbs it for random weights but not for graded ones |
| wire cap | `c_row` / `c_col` build an RC line with the right time constant, linear in `c` |
| signed tile | the column-pair difference recovers `W_signedᵀx` |

The two independent solvers agree to ~1e-12:

```
r= 1.0 ohm/pitch | ngspice vs numpy-MNA: max rel diff 6.58e-12
r= 5.0 ohm/pitch | ngspice vs numpy-MNA: max rel diff 1.01e-12
r=20.0 ohm/pitch | ngspice vs numpy-MNA: max rel diff 1.25e-12
r=   0 ohm/pitch | ngspice vs analytic G.T@x: max rel diff 3.49e-15
```

`sky130 sim` and `1T1R compression` skip themselves, passing, when no model library is
installed, so the suite still passes on a PDK-less machine. Every sky130 number quoted here
was measured against a volare sky130 install under ngspice-47; see
[`docs/REPORT.md`](docs/REPORT.md#reproduction) for exact versions.
