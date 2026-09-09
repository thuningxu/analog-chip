"""Floating-point matrix multiply on the analog crossbar.

Computes `C = A @ B` for arbitrary float64 `A` (M,N) and `B` (N,K) by running the real
`crossbar.Tile` in ngspice and decoding the sensed currents back to floating point.
Circuit construction is imported from `crossbar`, never duplicated.

Everything rests on the one identity the signed `Tile` already provides, verified to
2e-15 at zero wire resistance by the `signed tile` check in `verify_mna.py`:

    I_diff = (g_max - g_min) * (W.T @ v)        W in [-1, 1] (R,Cc),  v the row voltages

Encode.  Rows of the tile index the *contraction* dimension n, columns index the
*output* dimension m, so the tile holds A transposed:

    W[n,m] = A_block[m,n] / sA          sA = max|A_block|
    v[n]   = B_block[n,k] * vmax / sB   sB = max|B_block[:,k]|,  vmax = V_MAX

Decode.  Substituting into the identity, both normalizations divide straight out:

    I_diff[m] = (g_max - g_min) * sum_n (A_block[m,n]/sA) * B_block[n,k] * vmax/sB
              = (g_max - g_min) * vmax / (sA * sB) * (A_block @ B_block[:,k])[m]

    C_block[:,k] = I_diff * sA * sB / ((g_max - g_min) * vmax)

`sA` is recomputed **per block**: that is the block-floating-point exponent, and it is
what buys dynamic range on an array whose signal floor is set by `g_min`.  A single
global `sA` pushes small blocks down towards `w = 0`, where both legs of every column
pair sit at `g_min` and the answer is a difference of two nearly equal currents --
`global_scale=True` exists to measure that penalty, not to be used.

Signed inputs.  Row drivers are unipolar in any real design, so the default splits each
input column `b = b+ - b-` with both parts non-negative, drives the two halves
separately, and subtracts the decoded results.  Both halves share one `sB`, so the
subtraction is valid in decoded units.  `bipolar=True` drives negative row voltages
directly in a single pass: valid for the linear passive cell, half the solves, and
physically unrealistic -- and outright wrong for 1T1R, where a negative row voltage
swaps the access device's source and drain.  It is not the default.

Tiling.  `N` beyond `rows` splits the contraction dimension and accumulates the
*decoded* partial products in float64 (blocks carry different `sA`, so they cannot be
summed in the current domain).  `M` beyond `cols` splits and concatenates.

Batching.  The K input columns are electrically independent, so one deck holds one
`Tile` instance per column-pass -- `2K` instances unipolar, `K` bipolar -- each with its
own driver set, all solved in a single `.op`.  The tile subcircuit is defined once; only
the sources differ.  `batch=False` falls back to one ngspice launch per column; the
`batching` check in `verify_matmul.py` asserts the two agree.

See docs/MATMUL.md for measured accuracy, effective bits, and limitations.
"""
import time
from dataclasses import replace

import hdl21 as h
import hdl21.sim as hs
import numpy as np

import crossbar as C

V_MAX = 0.2   # V, top of the row-driver range -- the demo read voltage in REPORT.md
V_SEL = 1.8   # V, row select, matching crossbar.tile_sim

# The *reference* configuration: zero wire R and a passive cell is what makes the decode
# exact, and it is not a physical array -- see docs/MATMUL.md.  `.weights` is a
# placeholder that `matmul` overwrites per block, so it never reaches a deck.
IDEAL_XBAR = C.CrossbarParams(weights=((0.0,),), r_row=0.0, r_col=0.0,
                              cell=C.CellParams(passive=True))

# Sim accounting from the most recent `matmul` call: ngspice launches, tile solves
# (one per column-pass per block), and seconds spent inside those launches.
LAST_COST = {"decks": 0, "solves": 0, "t_sim": 0.0}


def blocks(n: int, size: int) -> list:
    """[(start, stop), ...] covering range(n) in chunks of `size`; the last may be short."""
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def as_matrix(W) -> C.Matrix:
    """numpy array -> the tuple-of-tuples `crossbar.Matrix` a generator parameter wants."""
    return tuple(tuple(float(v) for v in row) for row in W)


# --------------------------------------------------------------------------
# Batched testbench: one Tile instance per input column, one .op for all of them.
# --------------------------------------------------------------------------
def batch_sim(p: C.TileParams, V) -> hs.Sim:
    """`crossbar.tile_sim` replicated `len(V)` times in one deck, one driver set each.

    Every instance is the same generator call, so the tile -- and every `Cell`
    subcircuit under it -- is defined once and instantiated `len(V)` times.  Sense
    sources are named `vsensep{k}_{j}` / `vsensen{k}_{j}` and probed per column with one
    `.save` card each; `.save all` would carry every internal array node, which on the
    sky130 route is 20x the rawfile for nothing.
    """
    rows, cols = len(p.xbar.weights), len(p.xbar.weights[0])
    tile = C.Tile(p)

    tb = h.Module(name="Tb")
    tb.vss = h.Port()
    for k, v in enumerate(V):
        inp = tb.add(h.Signal(width=rows), name=f"inp{k}")
        outp = tb.add(h.Signal(width=cols), name=f"outp{k}")
        outn = tb.add(h.Signal(width=cols), name=f"outn{k}")
        sel = tb.add(h.Signal(width=rows), name=f"sel{k}")
        tb.add(tile(inp=inp, outp=outp, outn=outn, sel=sel, vss=tb.vss), name=f"dut{k}")
        for i in range(rows):
            tb.add(h.Vdc(dc=float(v[i]))(p=inp[i], n=tb.vss), name=f"vin{k}_{i}")
            tb.add(h.Vdc(dc=V_SEL)(p=sel[i], n=tb.vss), name=f"vsel{k}_{i}")
        for j in range(cols):
            tb.add(h.Vdc(dc=0.0)(p=outp[j], n=tb.vss), name=f"vsensep{k}_{j}")
            tb.add(h.Vdc(dc=0.0)(p=outn[j], n=tb.vss), name=f"vsensen{k}_{j}")

    models = C.access_model_attrs(tb, p.xbar.cell)
    saves = [hs.Literal(".save " + " ".join(
        f"i(v.xtop.vvsensep{k}_{j}) i(v.xtop.vvsensen{k}_{j})" for j in range(cols)))
        for k in range(len(V))]
    return hs.Sim(tb=tb, attrs=[*models, hs.Op(), *saves])


def run_batch(p: C.TileParams, V, rundir: str) -> np.ndarray:
    """(len(V), cols) differential currents I+ - I- from one ngspice launch."""
    cols = len(p.xbar.weights[0])
    res = batch_sim(p, V).run(hs.SimOptions(
        simulator=hs.SupportedSimulators.NGSPICE, fmt=hs.ResultFormat.SIM_DATA,
        rundir=rundir))
    d = res.an[0].data
    return np.array([[d[f"i(v.xtop.vvsensep{k}_{j})"] - d[f"i(v.xtop.vvsensen{k}_{j})"]
                      for j in range(cols)] for k in range(len(V))])


def run_serial(p: C.TileParams, V, rundir: str) -> np.ndarray:
    """Same shape as `run_batch`, one launch per column via `crossbar.run_tile_mac`."""
    return np.array([C.run_tile_mac(p, tuple(float(x) for x in v), rundir=f"{rundir}_k{k}")[0]
                     for k, v in enumerate(V)])


# --------------------------------------------------------------------------
# The driver.
# --------------------------------------------------------------------------
def matmul(A, B, xbar: C.CrossbarParams = IDEAL_XBAR, rows: int = 16, cols: int = 16,
           vmax: float = V_MAX, bipolar: bool = False, batch: bool = True,
           global_scale: bool = False, rundir: str = "/tmp/fpmm") -> np.ndarray:
    """`A @ B` through ngspice.  `xbar.weights` is ignored; it is set per block."""
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} do not contract")
    M, N, K = A.shape[0], A.shape[1], B.shape[1]
    delta = xbar.g_max - xbar.g_min
    run = run_batch if batch else run_serial
    s_global = np.abs(A).max() if global_scale else 0.0

    out = np.zeros((M, K))
    LAST_COST.update(decks=0, solves=0, t_sim=0.0)
    for n0, n1 in blocks(N, rows):
        sB = np.abs(B[n0:n1]).max(axis=0)
        # A zero scale means a zero block, whose product is zero either way: substituting
        # 1.0 keeps the division defined and still drives 0 V / programs w = 0.
        v = B[n0:n1] * (vmax / np.where(sB > 0.0, sB, 1.0))              # (Nb, K)
        drive = v.T if bipolar else np.concatenate(
            (np.maximum(v, 0.0).T, np.maximum(-v, 0.0).T))               # (npass*K, Nb)
        for m0, m1 in blocks(M, cols):
            Ab = A[m0:m1, n0:n1]
            sA = s_global if global_scale else np.abs(Ab).max()
            W = Ab.T / (sA if sA > 0.0 else 1.0)                         # (Nb, Mb)
            p = C.TileParams(xbar=replace(xbar, weights=as_matrix(W)))
            t0 = time.perf_counter()
            I = run(p, drive, f"{rundir}/n{n0}_m{m0}")                   # (npass*K, Mb)
            LAST_COST["t_sim"] += time.perf_counter() - t0
            LAST_COST["decks"] += 1 if batch else len(drive)
            LAST_COST["solves"] += len(drive)
            if not bipolar:
                I = I[:K] - I[K:]
            # Decode to physical units *before* accumulating: blocks carry different sA.
            out[m0:m1] += sA * (I * sB[:, None]).T / (delta * vmax)
    return out


def rel_errors(got, ref) -> tuple:
    """(max, rms) error relative to the norm of `ref`, not elementwise.

    Elementwise relative error is meaningless where a dot product nearly cancels, and a
    matrix product has such entries for free.  Both figures are normalized by the
    corresponding norm of the reference: max |dC| / max |C|, and rms(dC) / rms(C).
    """
    d = np.asarray(got, dtype=float) - np.asarray(ref, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return (float(np.abs(d).max() / np.abs(ref).max()),
            float(np.sqrt((d ** 2).mean()) / np.sqrt((ref ** 2).mean())))


def effective_bits(rel: float) -> float:
    """-log2(relative error): how many mantissa bits the analog answer still carries."""
    return float(np.inf) if rel <= 0.0 else float(-np.log2(rel))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, N, K = 12, 24, 6
    A = rng.standard_normal((M, N))
    B = rng.standard_normal((N, K))
    ref = A @ B
    print(f"A {A.shape} @ B {B.shape} on a {16}x{16} tile, unipolar two-pass, batched")

    for r, tag in ((0.0, "reference (r_wire = 0)"), (1.0, "r_wire = 1 ohm/pitch"),
                   (5.0, "r_wire = 5 ohm/pitch")):
        xbar = replace(IDEAL_XBAR, r_row=r, r_col=r)
        t0 = time.perf_counter()
        got = matmul(A, B, xbar=xbar, rundir=f"/tmp/fpmm_demo_r{r:g}")
        wall = time.perf_counter() - t0
        mx, rms = rel_errors(got, ref)
        print(f"  {tag:24s} max rel {mx:.3e}  rms rel {rms:.3e}  "
              f"| {effective_bits(mx):5.1f} bits (max) {effective_bits(rms):5.1f} bits (rms)"
              f"  | {LAST_COST['decks']} decks, {LAST_COST['solves']} solves, "
              f"{wall:.2f} s wall ({LAST_COST['t_sim']:.2f} s in ngspice)")
