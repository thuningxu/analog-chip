# int8 matmul on the analog crossbar

`int8_matmul.py` runs the workload a compute-in-memory inference accelerator actually
executes: symmetric int8 weights, symmetric int8 activations, an exact int32 accumulator as
the reference, and an int8 requantized output. Underneath it, `fp_matmul.py` is the analog
matmul engine — it encodes real matrices onto `crossbar.Tile`, runs ngspice, and decodes the
sensed currents back to numbers. The circuit itself is `crossbar.py`, unchanged, and the
non-idealities that limit everything below are the ones already characterized in
[`REPORT.md`](REPORT.md); this document does not restate them.

Every number here comes from a run executed on this machine: ngspice-47, hdl21 7.0.0 /
vlsirtools 7.0.0, sky130 via volare, Apple silicon. Commands, sim counts and wall clocks are
in [Cost](#5-cost).

Headline: at zero parasitics the analog path reproduces `A_q @ B_q` **exactly, as integers**.
At 1 Ω/pitch, **94.5%** of int8 outputs are exactly right with per-channel scales and a
per-channel gain calibration, against 71.9% uncalibrated — and no output is ever off by more
than one LSB. Exact int32 accumulation is out of reach at every nonzero parasitic level and
always will be.

---

## 1. Why the int8 *output* is the metric and the accumulator is not

This is the load-bearing design decision, so here is the arithmetic.

**Exact int32 accumulation.** An int8 product is at most `127 * 127 = 16129`, and `N` of them
sum, so the accumulator's full-scale range is `127² · N`. Resolving one integer needs half an
LSB of that:

```
bits needed = log2(2 · 127² · N)      = 20.0 bits at N = 32
                                      = 21.0 bits at N = 64
```

For a *particular* matrix pair the peak is far below the worst case, because random signs
cancel — measured 15.5 bits (per-tensor) and 16.4 bits (per-channel) for this study's
`A (16,32) @ B (32,8)`. Either way, the analog path delivers **6.8 to 11.5 bits** across
0.25–5 Ω/pitch. Exact int32 accumulation is short by 5 to 9 bits and no calibration closes
that: the gap is a factor of 30 to 500 in accuracy, not a trim. A study built on
accumulator-exactness would report "fails everywhere" at every realistic parasitic level and
teach nothing.

**int8 requantized output.** In a real quantized network the int32 accumulator is immediately
requantized to int8 for the next layer. That needs half an LSB out of 127:

```
tolerance = 1 / (2 · 127) = 0.39% of the output's full-scale range   =  8.0 bits
```

8 bits is the same order as the measured analog error, so this is the metric where the answer
is neither trivially yes nor trivially no — it moves with `r_wire`, with array height, with
the cell type, and with the calibration scheme. That is what makes it worth measuring.

The two thresholds are drawn on the accumulator-bits panel of the figure, and the crossing is
visible: per-channel calibrated sits at 8.56 bits at 2 Ω/pitch (89.8% of outputs exact) and
7.26 bits at 5 Ω/pitch (74.2% exact). Once the rms accumulator error passes half an output
LSB, the match rate falls apart. The bit metric predicts the integer metric, which is a
useful consistency check on both.

## 2. Quantization

Symmetric, zero-point-free, which is standard for CIM inference:

```
weights      s_w = max|A| / 127,        A_q = clip(round(A / s_w), -127, 127)
activations  s_x = max|B| / 127,        B_q = clip(round(B / s_x), -127, 127)
reference    C_ref = A_q @ B_q          exact in int64, asserted to fit int32
analog       C_analog = fp_matmul.matmul(A_q, B_q, ...)
output       s_y = max|C_ref| / 127,    y = clip(round(C / s_y), -127, 127)
```

`s_w` comes in two granularities, and the difference is the most interesting result in this
document:

- **per-tensor** — one scale for the whole weight matrix.
- **per-channel** (`per_channel=True`) — one scale per row of `A`, i.e. per output channel.
  On the array that is one scale per **column pair of the tile**, because the encoding puts
  the output dimension on the columns (§4.1).

Clipping is a no-op under symmetric max scaling (`max|X|/s` is exactly 127); it is kept so a
caller supplying their own scale cannot silently overflow. An all-zero matrix quantizes to
zeros with `s = 1/127`, which the `quantization` check pins.

int32 never actually overflows here: `127² · N` reaches `2³¹` only past N ≈ 133,000 rows.
`run_q` asserts it anyway.

## 3. Measured results

![Six-panel int8 study: output match rate and worst LSB deviation versus wire resistance, accumulator bits against the exactness thresholds, array height, 0T1R versus 1T1R, and calibration transfer](figures/int8-accuracy.png)

Regenerate with `uv run scripts/int8_accuracy.py`.

"Exactly correct" below always means the int8 value **equals** the one the integer reference
produces. There is no tolerance in the metric.

### 3.1 The exactness gate

At `r_row = r_col = 0` with `CellParams(passive=True)`, `round(C_analog)` must **be**
`A_q @ B_q`, elementwise. Measured across five shapes × two granularities — non-square,
ragged tiles, all-negative `A`, mixed signs, a zero column of `B`:

| quantity | worst over all ten cases |
| --- | --- |
| accumulator peak | 32835 |
| absolute error, of the 0.5 that rounding allows | **3.64e-11** |
| relative error, of the 1e-12 the float engine is held to | **1.25e-15** |
| int32 exact, elementwise | **yes, all ten** |
| int8 outputs exactly correct | **100%, 0 LSB deviation** |

**A correction to the brief:** the integer gate is *not* strictly stronger than the float
tolerance check — it is much weaker in relative terms. Rounding tolerates an absolute error
of 0.5, which at the measured peak of 16867 is a relative tolerance of 2.96e-05, i.e.
**3e7× looser** than the 1e-12 that `verify_matmul.py` asserts on the float engine. The
integer gate is the better *metric* (binary, no arbitrary constant, and it is what the
workload cares about); the float gate is the tighter *numerical* test and catches subtler
encode/decode bugs. `verify_int8.py` asserts both on every case and prints the slack in each,
so the claim is measured rather than argued.

### 3.2 Wire resistance

`A (16,32) @ B (32,8)`, 16×16 tile (16×32 physical), 0T1R, unipolar two-pass, batched.
2 decks / 32 solves per run, ~2.1 s per configuration.

int8 outputs exactly correct, %:

| `r_wire` [Ω/pitch] | per-tensor raw | per-tensor + one global gain | per-channel raw | **per-channel + per-channel gain** |
| --- | --- | --- | --- | --- |
| 0 | 100.00 | 100.00 | 100.00 | **100.00** |
| 0.25 | 92.97 | 98.44 | 89.06 | **99.22** |
| 0.5 | 86.72 | 96.09 | 82.03 | **96.88** |
| 1 | 74.22 | 90.62 | 71.88 | **94.53** |
| 2 | 60.16 | 78.91 | 51.56 | **89.84** |
| 5 | 32.03 | 62.50 | 26.56 | **74.22** |

Worst output deviation, int8 LSBs — how wrong the wrong ones are:

| `r_wire` | per-tensor raw | per-tensor + global | per-channel raw | per-channel + per-channel |
| --- | --- | --- | --- | --- |
| 0.25 | 1 | 1 | 1 | 1 |
| 1 | 1 | 1 | 2 | **1** |
| 2 | 2 | 1 | 3 | **1** |
| 5 | 5 | 2 | 7 | **1** |

**Per-channel calibration never misses by more than one LSB, all the way to 5 Ω/pitch.** That
matters more than the match rate for inference: a 1-LSB output error is indistinguishable
from ordinary quantization noise, while the 7-LSB errors the uncalibrated per-channel path
produces at 5 Ω/pitch are not.

Accumulator bits recovered (rms), against the 16.4 this pair would need for exact int32:

| `r_wire` | per-channel raw | per-channel + per-channel gain |
| --- | --- | --- |
| 0 | 48.4 | 48.6 |
| 0.25 | 8.8 | 11.5 |
| 0.5 | 7.8 | 10.6 |
| 1 | 6.8 | 9.6 |
| 2 | 5.8 | 8.6 |
| 5 | 4.5 | 7.3 |

### 3.3 Per-channel scales absorb IR droop for free

This is the brief's central hypothesis and it holds. Per-channel weight quantization already
applies one multiplier per output channel in the dequantization path; a per-column IR-drop
gain calibration is **the same degree of freedom**. Folding one into the other therefore costs
no hardware — `s_w[m] / a[m]` is the same instruction as `s_w[m]` — and no extra read.
`int8_matmul.calibrate` is a digital multiply on an accumulator the array already produced,
and the `int8 output` check asserts it launches no new simulation.

At 1 Ω/pitch that turns 71.9% into **94.5%**, and at 5 Ω/pitch 26.6% into **74.2%**. A single
global gain — all a per-tensor network can absorb — gets 90.6% and 62.5%. The per-channel
handle is worth roughly 4 points at 1 Ω/pitch and 12 points at 5 Ω/pitch over the global one,
and it is what caps the worst deviation at 1 LSB.

The gains it fits are all below 1, because IR drop is one-sided, and they genuinely differ
across columns — at 5 Ω/pitch, 0.9407 to 0.9879 against a global 0.9713. That 4.7-point
spread is exactly the part no global gain can reach.

**One result that looks backwards and is not.** Uncalibrated, per-channel scaling is *worse*
than per-tensor at every nonzero `r_wire` (26.6% vs 32.0% at 5 Ω/pitch). The mechanism is
measured, not guessed, and the `calibration` check in `verify_int8.py` asserts it on its own
independent matrix — same `A`, same `B`, only the weight-scale granularity changed, at
5 Ω/pitch:

| | mean \|A_q\| | global droop gain | per-column gain spread | raw int8 match |
| --- | --- | --- | --- | --- |
| per-tensor | 34.53 | 0.9712 | 3.00 pts | 45.83% |
| per-channel | **51.40** | **0.9682** | **3.88 pts** | 28.12% |

Per-channel scaling lifts every row of `A_q` to full scale, so it puts ~50% more weight mass
into the array — more conductance, more current, more IR drop. It loads the array harder *and*
hands over the per-column degrees of freedom to undo it; only the second half shows up in the
calibrated column. The right reading is not "per-tensor is more accurate": each scheme is
measured against its own exact reference, and per-channel represents small-magnitude rows far
better (a 1680× spread in per-channel steps on the deliberately unequal matrix in the
`quantization` check).

### 3.4 Does the calibration transfer? Yes, including across a distribution shift

In-sample calibration numbers are not a deployable claim, so gains fitted on one activation
batch were applied to two others: another Gaussian draw, and a **ReLU'd batch**
(`max(gaussian, 0)`, 50% zeros, non-negative) — what a real layer receives from the ReLU in
front of it, and a completely different total row current, which is what sets IR droop.

int8 outputs exactly correct, %, gains fitted on the Gaussian calibration batch:

| `r_wire` | Gaussian batch raw | → transferred | ReLU batch raw | → transferred | (ReLU refit in-sample) |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 92.97 | **100.00** | 94.53 | **100.00** | 100.00 |
| 0.5 | 85.16 | 97.66 | 88.28 | **98.44** | 98.44 |
| 1 | 68.75 | 96.88 | 73.44 | **96.88** | 97.66 |
| 2 | 47.66 | 92.97 | 47.66 | **92.97** | 92.97 |
| 5 | 28.12 | 81.25 | 27.34 | **78.12** | 82.81 |

**The transfer is essentially free.** Gains fitted on Gaussian activations recover the ReLU
batch to within 0 to 4.7 points of gains refitted on it directly, and at 2 Ω/pitch and below
they are identical. At 5 Ω/pitch on the same-distribution batch the transferred gains score
*higher* than the in-sample refit (81.25% vs 76.56%) — a least-squares gain is not the
match-rate-optimal gain, so in-sample fitting carries no guarantee.

The conclusion is that per-column droop is dominated by the array, not by the activation
statistics, which is what makes a one-time factory calibration a real option. This was worth
checking rather than assuming, and it is a stronger statement than the brief asked for.

### 3.5 Array height: the design limit

Wire resistance alone does not tell a designer how big to build the array. This does. Fixed
0.5 Ω/pitch, tile `N × 8` (N × 16 physical), `A (8,N) @ B (N,8)`, one contraction block so
that **all N rows accumulate in the analog domain**:

| N | per-tensor raw | per-channel + per-channel gain | worst LSB (calibrated) |
| --- | --- | --- | --- |
| 4 | 95.31 | **100.00** | 0 |
| 8 | 95.31 | **100.00** | 0 |
| 16 | 87.50 | **100.00** | 0 |
| 32 | 93.75 | **96.88** | 1 |
| 64 | 64.06 | **85.94** | 1 |

**Calibrated per-channel int8 is exact to N = 16, still 96.9% at N = 32, and breaks at
N = 64.** At 0.5 Ω/pitch the usable analog contraction depth is on the order of 32 rows. Past
that, `fp_matmul`'s tiling moves the accumulation into digital adds, which are exact — so a
deeper contraction is not a correctness problem, it is a cost problem: more tiles, more
solves, more digital accumulate. The design question "how tall should the array be?" has a
measurable answer, and at this parasitic level it is not 128 or 1024.

(The N = 16 to N = 32 non-monotonicity in the per-tensor column is a single random matrix
pair, not a trend. See [Limitations](#7-limitations).)

### 3.6 0T1R vs 1T1R: Ron compression is not a per-column gain

`A (8,16) @ B (16,4)`, 8×8 tile (8×16 physical), per-channel weight scales.

| cell | `r_wire` | raw | + per-channel gain | worst LSB, raw → calibrated |
| --- | --- | --- | --- | --- |
| 0T1R | 0 | **100.00** | 100.00 | 0 → 0 |
| 0T1R | 1 Ω/pitch | 81.25 | **100.00** | 1 → 0 |
| 1T1R generic (placeholder) | 0 | 6.25 | 46.88 | 11 → 2 |
| 1T1R sky130 `nfet_01v8` | 0 | 15.62 | **53.12** | 8 → 1 |

**This is the sharpest result in the study.** A per-column gain removes wire IR droop
*completely* — 81.3% → 100.0%, every output exact — and removes the sky130 access device's
Ron compression only partially: 15.6% → 53.1%, still half the outputs wrong at zero wire
resistance.

The reason is exactly the one the brief predicted. IR droop is, to first order, a per-column
gain, so a per-column multiplier is the right shape to cancel it. Ron compression is a
**per-cell nonlinearity** — the effective conductance is `1/(1/G + Ron)`, which compresses
the top of the weight range ~90× harder than the bottom
([REPORT.md §5](REPORT.md#5-access-device-1t1r-weight-dependent-range-compression), 7.6% at
`g_max` against 0.08% at `g_min`). One number per column can only remove that column's
*mean* compression, and what is left is weight-dependent, not gain-shaped.

Two caveats:

- `A` and `B` are i.i.d. standard normal, the **optimistic** structure for this comparison.
  Per [REPORT.md §5.3](REPORT.md#53-the-consequence-depends-on-weight-structure), unstructured
  weights let each column average over the compression curve; structured weights, which real
  trained networks have, break that averaging. Read 53.1% as a ceiling.
- The generic route is worse than sky130 (46.9% vs 53.1%) because its implied Ron is ~42%
  pessimistic. It is an uncalibrated placeholder and no silicon conclusion follows from it —
  see [README](../README.md#cell-configurations).

**Design implication.** At these settings the access device costs more int8 fidelity than
5 Ω/pitch of wire does, and unlike the wire it is not calibratable away. REPORT.md §5's two
levers apply: widen the access FET (W = 2 µm roughly halves Ron) or lower `g_max`, trading
compression against signal current.

## 4. The analog engine underneath

`fp_matmul.py` is the substrate. It computes `C = A @ B` for arbitrary float64 matrices; int8
is a quantization layer on top, and the engine neither knows nor cares that its operands
happen to be integers. Everything in this section is verified independently by
`verify_matmul.py` and characterized in float terms by
`scripts/matmul_accuracy.py` → [`matmul-accuracy.png`](figures/matmul-accuracy.png).

### 4.1 Encode and decode

Everything rests on the identity the signed `Tile` already provides, derived in
[DESIGN.md](DESIGN.md#tile--signed-weights-on-a-unipolar-array) and verified to 2.0e-15 by the
`signed tile` check in [`verify_mna.py`](../verify_mna.py) — for weights `W ∈ [−1,1]` of shape
(R, Cc) and row voltages `v`:

```
I_diff = (g_max - g_min) * (W.T @ v)
```

Rows of the tile index the **contraction** dimension `n`; columns index the **output**
dimension `m`. So the tile holds `A` transposed:

```
W[n,m] = A_block[m,n] / sA          sA = max|A_block|
v[n]   = B_block[n,k] * vmax / sB   sB = max|B_block[:,k]|,  vmax = 0.2 V
```

Substituting, both normalizations divide straight back out:

```
I_diff[m] = (g_max - g_min) * vmax / (sA * sB) * (A_block @ B_block[:,k])[m]

C_block[:,k] = I_diff * sA * sB / ((g_max - g_min) * vmax)
```

Nothing in the decode is calibrated: `g_max`, `g_min` and `vmax` are design constants and
`sA`, `sB` come from the data. `g_min` never appears — it cancels in the differential pair by
construction, which is the property the `Tile` scheme buys. A zero `sA` or `sB` means a zero
block; the code substitutes 1.0 to keep the division defined and the contribution is still
exactly zero.

That the output dimension lands on the **columns** is what makes §3.3 work: a per-output-channel
scale and a per-column gain calibration are the same object.

### 4.2 Tiling and block scaling

`N` beyond `rows` splits the contraction dimension; each chunk has its own `sA`, so partial
products are **decoded to physical units and accumulated in float64**. They cannot be summed
in the current domain — different blocks carry different scale factors. `M` beyond `cols`
splits the output dimension and the decoded blocks concatenate. The tile count is
`ceil(N/rows) · ceil(M/cols)`; ragged chunks elaborate smaller arrays rather than being
padded. Tile-size invariance is checked directly: the same `(6,9) @ (9,4)` product through one
16×16 tile and through nine 3×2 tiles agrees with numpy to 6.4e-16 and 4.8e-16, and with
itself to 8.6e-16.

Per-block `sA` is a block-floating-point exponent, and it is not free. The `g_min` pedestal is
the mechanism: when a global scale pushes a block's weights towards zero, both legs of every
column pair sit at `g_min` and the answer becomes a small difference of two large, nearly
equal currents. Measured on a matrix whose output blocks span twelve decades:

| block magnitude | per-block `sA` | one global `sA` |
| --- | --- | --- |
| 1 | 51.1 bits | 51.1 bits |
| 1e−4 | 51.3 bits | 43.4 bits |
| 1e−8 | 50.0 bits | 29.1 bits |
| 1e−12 | 49.5 bits | **16.9 bits** |

32.6 bits of headroom on the smallest block, or 3.3 bits per decade of spread — `log2(10)`,
one decade of lost signal per decade of lost scale.

**Three limits on how far to read that, because it is easy to overclaim.**

- **The spread has to be along the output dimension.** It is, deliberately. A magnitude spread
  along the *contraction* dimension shows **no benefit at all**: those blocks' partial products
  are summed, so a block that is 1e−8 down contributes nothing to the total and getting it
  wrong relatively costs nothing absolutely. Separate output rows of `C`, by contrast, each
  have to be right on their own terms. Anyone reproducing this along the contraction dimension
  will measure zero effect and should not conclude block-FP is pointless.
- **The mechanism is the `g_min` pedestal, not quantization.** There is no quantizer anywhere
  on the float path — conductances are continuous float64 — so the loss is entirely the
  cancellation of two nearly equal leg currents. In exact arithmetic a global scale would be
  free, and at `r = 0` with well-conditioned magnitudes it very nearly is.
- **It barely matters for int8, the headline path.** Per-channel int8 quantization already puts
  every row of `A_q` at full scale, so within an output block `sA` is 127 everywhere and the
  block exponent has nothing left to do. Block floating point earns its keep on the float
  engine and on badly scaled real matrices, not on the quantized workload in §3.

### 4.3 Unipolar two-pass inputs

A real row driver cannot pull a row line below the column-side virtual ground, so the default
splits each input column into two non-negative halves and runs both, sharing one `sB` (because
`max|b| = max(max b⁺, max b⁻)`) so the subtraction is valid in decoded units. It costs 2
solves per input column.

`bipolar=True` drives negative row voltages directly in one pass. On the linear passive cell
this is **exact** — the array with every column at virtual ground is a linear resistive
network, so superposition in the row voltages holds identically:

| configuration | bipolar vs unipolar two-pass, max rel. difference |
| --- | --- |
| 0T1R, `r_wire` = 0 | 4.63e-16 |
| 0T1R, `r_wire` = 5 Ω/pitch | 4.77e-15 |
| 1T1R (generic placeholder), `r_wire` = 0 | **6.53e-04** |

The last row is why it is not the default: with an access device in series, a negative row
voltage reverses the FET's source and drain and forward-biases the drain–bulk junction. The
gap is small in absolute terms because the read voltages are small, but eleven orders of
magnitude above the passive result. Use it to halve the solve count on a 0T1R study; do not
use it to characterize 1T1R.

### 4.4 Batching

One ngspice process per input column is the obvious implementation and the wrong one. The K
input columns are electrically independent, so `fp_matmul.batch_sim` emits **one deck holding
one `Tile` instance per column-pass** — `2K` unipolar, `K` bipolar — each with its own driver
set, all solved in a single `.op`. Every instance is the same generator call, so the tile and
every `Cell` subcircuit under it are defined once and instantiated `2K` times. Probes use one
`.save` card per column; `.save all` carries every internal array node instead, measured at
20× the rawfile (1280 signals against 64) on an 8×8 passive deck, and worse on 1T1R which adds
a node per cell. Multiple `.save` cards accumulate in ngspice, and the two strategies were
checked to give bit-identical currents.

Batching works, and it was checked rather than assumed. `batch=False` runs the same problem
one launch per column through `crossbar.run_tile_mac`:

| case | batched | unbatched | agreement | speedup |
| --- | --- | --- | --- | --- |
| 0T1R, `A(8,16) @ B(16,8)`, 8×8 tile, 5 Ω/pitch | 2 decks, 0.22 s | 32 decks, 0.81 s | 3.57e-15 | 3.7x |
| 0T1R, `A(16,64) @ B(64,8)`, 16×16 tile, 1 Ω/pitch | 4 decks, 2.15 s | 64 decks, 4.55 s | 1.22e-14 | 2.1x |
| 1T1R sky130, `A(8,16) @ B(16,4)`, 8×8 tile | 2 decks, 30.9 s | 16 decks, 159.7 s | 6.13e-14 | **5.2x** |

The first row is the `batching` check in `verify_matmul.py`, which prints those wall clocks
and asserts the ratio exceeds 1; they jitter a few percent between runs. It runs at
5 Ω/pitch on purpose — there every instance's answer depends on the whole nodal solve, so
instances leaking into each other through a shared node, or probes read off the wrong source,
could not hide behind a trivially decoupled circuit.

The speedup is largest on sky130, the opposite of what "the nonlinear solve scales
superlinearly, so bigger decks lose" predicts. What dominates a sky130 launch is process
startup plus re-parsing the `.lib`-sectioned model library
([DESIGN.md](DESIGN.md#the-sky130-route-in-detail)), and batching pays that once instead of 16
times. That row is a one-off measurement, not in the checked-in suite, because reproducing it
costs ~3.2 minutes:

```python
import time; from dataclasses import replace
import numpy as np, crossbar as C, fp_matmul as F
rng = np.random.default_rng(1)
A, B = rng.standard_normal((8, 16)), rng.standard_normal((16, 4))
xb = replace(F.IDEAL_XBAR, cell=C.CellParams(access="sky130"))
for batch in (True, False):
    t0 = time.perf_counter()
    F.matmul(A, B, xbar=xb, rows=8, cols=8, batch=batch, rundir=f"/tmp/bc_{batch}")
    print(batch, F.LAST_COST, f"{time.perf_counter() - t0:.1f} s")
```

### 4.5 The float engine's own accuracy

For completeness, the same array measured in float terms rather than int8 — `A (16,64) @
B (64,8)`, 16×16 tile, 0T1R, error relative to numpy float64 and converted to effective
mantissa bits (`−log2` of the norm-relative error):

| `r_wire` [Ω/pitch] | rms rel error | effective bits | after one global gain fit |
| --- | --- | --- | --- |
| 0 | 2.69e-15 | 48.4 | 48.5 |
| 0.5 | 3.31e-03 | 8.2 | 9.6 |
| 1 | 6.59e-03 | 7.3 | 8.6 |
| 5 | 3.21e-02 | 5.0 | 6.3 |

FP32 has a 24-bit mantissa; about 7 survive at 1 Ω/pitch. Details, the sky130 comparison and
the block-FP study are in [`matmul-accuracy.png`](figures/matmul-accuracy.png) and the
`matmul_accuracy` script's stdout. The int8 framing above is the useful one precisely because
"7.3 of 24 mantissa bits" has no pass/fail reading, while "94.5% of int8 outputs exactly
right, worst case 1 LSB" does.

## 5. Cost

Per matmul, with `rows`/`cols` the tile dimensions:

```
tiles            = ceil(N/rows) * ceil(M/cols)
solves           = tiles * K * (2 unipolar, 1 bipolar)
ngspice launches = tiles          (batched, the default)
                 = solves         (batch=False)
```

Verified against `fp_matmul.LAST_COST`: the int8 demo's `A(12,32) @ B(32,6)` on a 16×16 tile
gives `ceil(32/16) * ceil(12/16) = 2` tiles and `2 * 6 * 2 = 24` solves, which is what it
prints. Quantization, requantization and calibration add **zero** simulations — all three are
numpy on arrays the array already produced.

Note what does not appear: `M`, `N` and `K` do not multiply into the solve count the way they
do into a digital matmul's flop count. The contraction happens in the analog domain, so the
cost is `tiles · K` solves regardless of how long the rows are. That is the entire
architectural argument for a crossbar and it survives this encoding intact — with the caveat
from §3.5 that the *usable* contraction depth is bounded by parasitics, so past ~32 rows at
0.5 Ω/pitch the tile count starts growing again.

Measured wall clock, ngspice-47 on Apple silicon:

| command | what | wall |
| --- | --- | --- |
| `uv run int8_matmul.py` | 8 int8 matmuls, `A(12,32) @ B(32,6)`, at 0 / 0.25 / 1 / 5 Ω/pitch × two granularities — 16 decks, 192 solves | **6.5 s** |
| `uv run verify_int8.py` | 4 int8 checks | **5.2 s** |
| `uv run scripts/int8_accuracy.py` | the whole int8 study, including the sky130 panel | **110 s** |
| `uv run fp_matmul.py` | 3 float matmuls at 0 / 1 / 5 Ω/pitch | **1.9 s** |
| `uv run verify_matmul.py` | 4 float-engine checks, 97 decks and 562 tile solves | **3.8 s** |
| `uv run scripts/matmul_accuracy.py` | the float study | **50 s** |

(Wall clocks jitter a few percent between runs; these are from one clean sequential pass.)

Per-deck cost from those runs: a 16×16 tile carrying 16 instances of a 0T1R array solves in
~1.0 s (2 decks in 2.1 s); an 8×8 tile carrying 8 sky130 `nfet_01v8` instances takes ~34 s
(2 decks in 68.3 s). Those are different array sizes, so for a like-for-like figure — a 16×16
tile with 16 instances on both routes, measured once by hand — 0T1R is 0.42 s and sky130 is
**246 s**, about 590×. That is why the study's array sizes differ between panels, and why a
16×16 sky130 tile at K = 16 is not something to reach for.

## 6. What "exact" does and does not mean

At `r_row = r_col = 0` with `CellParams(passive=True)` the analog path reproduces
`A_q @ B_q` exactly as integers, and every requantized int8 output is exactly right. That
result is load-bearing but narrow.

**It is a statement about the encoding and the solver, not about analog computing.** With zero
wire resistance and a linear resistive cell the crossbar *is* an exact linear map — `I = Gᵀv`
with nothing else in it — and ngspice solves that map in float64. The only thing the exactness
gate can fail on is an error in the encode, the decode, the tiling, the accumulation, or the
quantization. That is exactly why it is the gate: it isolates the arithmetic from the physics,
which is what lets a 94.5% match rate at 1 Ω/pitch be attributed to the wire with confidence
rather than to a units slip.

**The achievable fidelity on any real array is set by the non-idealities in
[REPORT.md](REPORT.md), plus two whole categories this project does not model at all:**

- **ADC quantization.** There is no ADC. Column current is read by an ideal 0 V voltage source
  — infinite gain and bandwidth, zero offset, zero noise, zero input impedance. Every match
  rate above assumes the accumulator is digitized perfectly. On real silicon the ADC's
  resolution and its own nonlinearity would very likely dominate everything measured here: an
  8-bit column ADC cannot deliver the 8.0 accumulator bits the int8 output needs, let alone
  the 9.6 measured at 1 Ω/pitch.
- **Device variation.** The cell is a linear resistor at exactly its programmed conductance.
  No cycle-to-cycle or device-to-device variation, no conductance drift or relaxation, no read
  noise, no random telegraph noise, no stuck-at defects, no retention or endurance model, no
  programming dynamics. In published ReRAM arrays this is frequently the *dominant* accuracy
  limit — larger than the IR drop measured here — and unlike IR droop it is not a per-column
  gain, so §3.3's calibration would not absorb it either.

Neither is a small correction to §3. Both would move every number downwards, and the device
sigma in particular would attack the one result that currently looks clean: per-channel
calibration reaching 100% at 1 Ω/pitch.

## 7. Limitations

Beyond §6 and everything in
[REPORT.md §9](REPORT.md#9-limitations-and-threats-to-validity), which applies unchanged:

- **No DAC either.** Row voltages are ideal `Vdc` sources set to exact values. A real driver
  quantizes the input — the input-side twin of the missing ADC. `in_bits` was removed from
  `TileParams` precisely because no circuit backed it.
- **This is not an inference-accuracy study.** One random matrix pair per configuration, one
  seed, i.i.d. Gaussian. A match rate on random data says nothing about top-1 accuracy on a
  real network: what matters there is whether the errors land on the argmax, which depends on
  the layer, the data and the task. Every script takes a `seed` for anyone who wants to check
  the variance; the N = 16 → 32 non-monotonicity in §3.5 is a reminder that single-sample
  numbers wobble by a few points.
- **Weight structure matters and is untested here.** [REPORT.md
  §5.3](REPORT.md#53-the-consequence-depends-on-weight-structure) shows the 1T1R conclusion is
  sensitive to weight *structure*, with a ~6× spread between i.i.d. and graded weights. Real
  trained weights are structured, and §3.6's per-channel calibration result on i.i.d. data is
  the optimistic end of that range. The same caveat applies to §3.3.
- **Symmetric quantization only.** No zero points, no asymmetric ranges, no per-token
  activation scales, no mixed precision, no quantization-aware training. Symmetric int8 is
  standard for CIM weights but real deployments use more than this.
- **`s_w`, `s_x` and `s_y` are computed offline from the full data.** A streaming
  implementation cannot see `max|B_block[:,k]|` before it drives the rows, and `s_y` needs the
  output range before the output exists. Real pipelines use running statistics or static
  calibrated ranges, which are worse.
- **Array sizes are small.** Largest tile is 64×8 and 16×16 logical. §3.5 measures the height
  limit at one wire resistance only, and the sky130 route's cost is what stops this study from
  going further.
- **No transient, energy or throughput.** Every result is a `.op`. REPORT.md §6 has the
  array's own RC pole and the conclusion that the readout amplifier, not the wire, sets read
  time.
- **Calibration transfer was tested across two activation distributions, not many.** §3.4
  covers a Gaussian→ReLU shift, which is the realistic one for a hidden layer, but not
  sparsity extremes, outlier-heavy distributions, or temperature and ageing drift of the array
  itself — and drift is the mechanism most likely to invalidate a factory calibration.

## 8. Reproduction

```bash
uv sync
brew install ngspice                 # verified on ngspice-47

uv run int8_matmul.py                # int8 demo, 0 / 0.25 / 1 / 5 ohm/pitch      ~7 s
uv run verify_int8.py                # 4 checks; int32 exactness is the gate       ~5 s
uv run scripts/int8_accuracy.py      # regenerate int8-accuracy.png             ~110 s

uv run fp_matmul.py                  # the float engine underneath                ~2 s
uv run verify_matmul.py              # 4 checks on the float engine               ~4 s
uv run scripts/matmul_accuracy.py    # regenerate matmul-accuracy.png            ~50 s
```

Both verify scripts end in `VERIFY: PASS` and exit nonzero if any check fails. Both accuracy
studies need a sky130 install for their 1T1R panel and fail up front with `sky130_root()`'s
actionable message if there is none; see [REPORT.md §10](REPORT.md#10-reproduction) for the
volare command.

| file | role |
| --- | --- |
| [`int8_matmul.py`](../int8_matmul.py) | symmetric int8 quantization, requantization, per-channel gain calibration, metrics |
| [`verify_int8.py`](../verify_int8.py) | 4 checks: int32 exactness, quantization, int8 output, calibration |
| [`fp_matmul.py`](../fp_matmul.py) | the analog engine: encode/decode, block-FP tiling, the batched testbench |
| [`verify_matmul.py`](../verify_matmul.py) | 4 checks: exactness, batching, bipolar, block FP |
| [`scripts/int8_accuracy.py`](../scripts/int8_accuracy.py) | the study behind `figures/int8-accuracy.png` |
| [`scripts/matmul_accuracy.py`](../scripts/matmul_accuracy.py) | the study behind `figures/matmul-accuracy.png` |
