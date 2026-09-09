"""Machine checks for the floating-point matmul driver in `fp_matmul.py`.

The gate is check 1. At `r_row = r_col = 0` with `CellParams(passive=True)` the crossbar
is an exactly linear resistive network and the signed `Tile` identity holds to 2e-15
(pinned by `signed tile` in `verify_mna.py`), so the decoded product *must* reproduce
numpy float64 `A @ B` to solver precision. Anything worse there is an encode, decode or
tiling bug, not physics -- nothing else in this file, and nothing in
`scripts/matmul_accuracy.py`, means anything until it passes.

Checks, in order:
    1  exactness   decoded A@B vs numpy float64 at zero parasitics, across non-square
                   shapes, ragged tiles, all-negative A, mixed signs, a zero column of
                   B, and two tile sizes on the same product (tile-size invariance).
    2  batching    K Tile instances in one deck vs one deck per column, at 5 ohm/pitch
                   where the array is not trivial, with the wall clock for both.
    3  bipolar     the one-pass negative-voltage shortcut is exact for the linear
                   passive cell (superposition) and wrong for 1T1R.
    4  block fp    per-block sA against a single global sA, on a matrix whose output
                   blocks span 12 decades.
"""

import time
from dataclasses import replace

import numpy as np

import crossbar as C
import fp_matmul as F


def wire(r: float, **kw) -> C.CrossbarParams:
    """`IDEAL_XBAR` at a chosen wire resistance."""
    return replace(F.IDEAL_XBAR, r_row=r, r_col=r, **kw)


def check_exactness() -> bool:
    """Zero parasitics, passive cells: the decode must reproduce numpy float64."""
    rng = np.random.default_rng(2)
    zero_col = rng.standard_normal((11, 4))
    zero_col[:, 2] = 0.0

    cases = {
        "square, one tile      ": (rng.standard_normal((8, 8)), rng.standard_normal((8, 8)), 16, 16),
        "non-square, ragged    ": (rng.standard_normal((5, 20)), rng.standard_normal((20, 3)), 6, 4),
        "all-negative A        ": (-np.abs(rng.standard_normal((7, 13))), rng.standard_normal((13, 4)), 5, 3),
        "mixed-sign A and B    ": (rng.standard_normal((9, 17)), rng.standard_normal((17, 5)), 4, 4),
        "zero column in B      ": (rng.standard_normal((6, 11)), zero_col, 4, 4),
    }
    ok = True
    for tag, (A, B, rows, cols) in cases.items():
        got = F.matmul(A, B, rows=rows, cols=cols, rundir=f"/tmp/fpmm_ex_{rows}_{cols}_{A.shape[0]}")
        mx, rms = F.rel_errors(got, A @ B)
        ok &= mx < 1e-12
        print(f"  {tag} {A.shape} @ {B.shape} on {rows}x{cols} tile, "
              f"{F.LAST_COST['decks']:2d} decks | max rel {mx:.2e}  rms rel {rms:.2e}"
              f"  {'ok' if mx < 1e-12 else 'FAIL'}")

    # Tile-size invariance: the same product through one tile and through nine.
    A, B = rng.standard_normal((6, 9)), rng.standard_normal((9, 4))
    one = F.matmul(A, B, rows=16, cols=16, rundir="/tmp/fpmm_ex_one")
    many = F.matmul(A, B, rows=3, cols=2, rundir="/tmp/fpmm_ex_many")
    mx_one, _ = F.rel_errors(one, A @ B)
    mx_many, _ = F.rel_errors(many, A @ B)
    mx_pair, _ = F.rel_errors(many, one)
    ok &= mx_one < 1e-12 and mx_many < 1e-12 and mx_pair < 1e-12
    print(f"  tile-size invariance   (6,9)@(9,4): 1 tile max rel {mx_one:.2e}, "
          f"9 tiles max rel {mx_many:.2e}, against each other {mx_pair:.2e}")
    print(f"  worst over all cases must be < 1e-12: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def check_batching() -> bool:
    """One deck with K tile instances must equal K decks with one instance each.

    Run at 5 ohm/pitch, where the array is not trivial and every instance's answer depends
    on the whole nodal solve, so a batching mistake -- instances leaking into each other
    through a shared node, or probes read off the wrong source -- cannot hide.
    """
    rng = np.random.default_rng(4)
    A, B = rng.standard_normal((8, 16)), rng.standard_normal((16, 8))
    ref = A @ B

    out = {}
    for batch in (True, False):
        t0 = time.perf_counter()
        got = F.matmul(A, B, xbar=wire(5.0), rows=8, cols=8, batch=batch,
                       rundir=f"/tmp/fpmm_bat_{batch}")
        out[batch] = (got, time.perf_counter() - t0, F.LAST_COST["decks"],
                      F.LAST_COST["solves"])
        print(f"  batch={str(batch):5s} | {out[batch][2]:2d} decks, {out[batch][3]} solves | "
              f"{out[batch][1]:5.2f} s | max rel vs numpy {F.rel_errors(got, ref)[0]:.3e}")

    mx, _ = F.rel_errors(out[True][0], out[False][0])
    print(f"  batched vs unbatched: max rel {mx:.2e} | "
          f"batching is {out[False][1] / out[True][1]:.1f}x faster here")
    checks = {
        "batched == unbatched to 1e-12": mx < 1e-12,
        "batching collapses 32 decks into 2": out[True][2] == 2 and out[False][2] == 32,
        "same solve count either way": out[True][3] == out[False][3] == 32,
        "both differ from numpy by the same IR drop": abs(
            F.rel_errors(out[True][0], ref)[0] - F.rel_errors(out[False][0], ref)[0]) < 1e-12,
        "batching is not slower": out[False][1] > out[True][1],
    }
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


def check_bipolar() -> bool:
    """The one-pass shortcut is exact on a linear passive cell and wrong on 1T1R.

    A passive crossbar with every column at virtual ground is a linear resistive network,
    so superposition in the row voltages is exact: driving `b+ - b-` in one pass equals
    the difference of two unipolar passes, at *any* wire resistance. Put an access FET in
    series and it stops holding -- a negative row voltage reverses the device's
    source/drain and forward-biases the drain-bulk junction. The generic placeholder model
    is enough to show that, because this is a topology claim, not a performance one.
    """
    rng = np.random.default_rng(5)
    A, B = rng.standard_normal((6, 8)), rng.standard_normal((8, 4))
    ref = A @ B

    out = {}
    for tag, xbar in (("passive, r=0", wire(0.0)), ("passive, r=5", wire(5.0)),
                      ("1T1R generic, r=0", wire(0.0, cell=C.CellParams()))):
        uni = F.matmul(A, B, xbar=xbar, rows=8, cols=6, rundir=f"/tmp/fpmm_bip_u{len(out)}")
        n_uni = F.LAST_COST["solves"]
        bip = F.matmul(A, B, xbar=xbar, rows=8, cols=6, bipolar=True,
                       rundir=f"/tmp/fpmm_bip_b{len(out)}")
        n_bip = F.LAST_COST["solves"]
        gap, _ = F.rel_errors(bip, uni)
        out[tag] = gap
        print(f"  {tag:18s} | unipolar {n_uni} solves, bipolar {n_bip} | "
              f"unipolar vs numpy {F.rel_errors(uni, ref)[0]:.3e}, "
              f"bipolar vs numpy {F.rel_errors(bip, ref)[0]:.3e} | "
              f"bipolar vs unipolar {gap:.3e}")

    checks = {
        "bipolar halves the solve count": n_bip * 2 == n_uni,
        "exact on the passive cell at r=0": out["passive, r=0"] < 1e-12,
        "exact on the passive cell at r=5 (superposition)": out["passive, r=5"] < 1e-12,
        "not exact on 1T1R": out["1T1R generic, r=0"] > 1e-6,
    }
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


BFP_DECADES = (0, 4, 8, 12)   # magnitude of each output block of A, as 10**-d


def block_fp_case(rows=8, cols=4, K=3, seed=3):
    """A whose output blocks span `BFP_DECADES`, so each output tile needs its own sA.

    The spread is along the *output* dimension on purpose. A spread along the contraction
    dimension is invisible in the answer -- the small blocks contribute nothing to the
    sum -- whereas separate output rows of C each have to be right on their own terms.
    """
    rng = np.random.default_rng(seed)
    M, N = cols * len(BFP_DECADES), rows
    A = rng.standard_normal((M, N))
    for b, d in enumerate(BFP_DECADES):
        A[b * cols:(b + 1) * cols] *= 10.0 ** -d
    return A, rng.standard_normal((N, K)), rows, cols


def check_block_fp() -> bool:
    """Per-block sA is the block-floating-point exponent; one global sA throws it away."""
    A, B, rows, cols = block_fp_case()
    ref = A @ B
    per = F.matmul(A, B, rows=rows, cols=cols, rundir="/tmp/fpmm_bfp_per")
    glob = F.matmul(A, B, rows=rows, cols=cols, global_scale=True, rundir="/tmp/fpmm_bfp_glob")

    print("  block   |A| scale   per-block sA          one global sA")
    print("                      max rel     bits      max rel     bits")
    worst_per, worst_glob = 0.0, 0.0
    for b, d in enumerate(BFP_DECADES):
        s = slice(b * cols, (b + 1) * cols)
        mp = F.rel_errors(per[s], ref[s])[0]
        mg = F.rel_errors(glob[s], ref[s])[0]
        worst_per, worst_glob = max(worst_per, mp), max(worst_glob, mg)
        print(f"  {b}       1e-{d:<10d} {mp:.2e}  {F.effective_bits(mp):6.1f}    "
              f"{mg:.2e}  {F.effective_bits(mg):6.1f}")

    checks = {
        "per-block sA is exact on every block": worst_per < 1e-12,
        "one global sA is not": worst_glob > 1e-6,
        "block-FP wins by >1e6 on the worst block": worst_glob / worst_per > 1e6,
    }
    print(f"  worst block: per-block {worst_per:.2e} ({F.effective_bits(worst_per):.1f} bits), "
          f"global {worst_glob:.2e} ({F.effective_bits(worst_glob):.1f} bits) -- "
          f"{F.effective_bits(worst_per) - F.effective_bits(worst_glob):.1f} bits of headroom")
    for name, hit in checks.items():
        print(f"  {'ok  ' if hit else 'FAIL'} {name}")
    return all(checks.values())


CHECKS = [
    ("exactness", check_exactness),
    ("batching", check_batching),
    ("bipolar", check_bipolar),
    ("block fp", check_block_fp),
]


def main() -> int:
    results = {}
    for name, fn in CHECKS:
        print(f"\n== {name} ==")
        results[name] = fn()

    print()
    for name, hit in results.items():
        print(f"{'PASS' if hit else 'FAIL'}  {name}")
    ok = all(results.values())
    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
