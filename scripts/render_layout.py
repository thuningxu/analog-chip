"""Render the drawn sky130 layout to PNG, and copy the GDS somewhere durable.

Two reasons this exists. The layout the rest of the project measures only ever lived in
a scratch rundir, so there was nothing to look at without regenerating it; and a repo
whose headline result is a drawn cell should show the cell.

The GDS is regenerated from `cell_layout`'s own planner rather than copied from a stale
run, so the images cannot drift from the geometry the parasitics were derived on.
Rasterizing goes through the KLayout *application* (the pip module has no renderer) with
sky130's own `sky130A.lyp`, so layer colours and fill patterns are the PDK's rather than
invented.

    uv run scripts/render_layout.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cell_layout as C
import crossbar as X
import layout_oracle as O

OUT = Path(__file__).resolve().parent.parent / "docs" / "layout"

# The KLayout GUI binary. The pip module cannot rasterize -- it has no LayoutView -- so
# this needs the application. Homebrew's cask lands it under /Applications/KLayout/.
KLAYOUT_CANDIDATES = (
    Path.home() / "klayout-local/klayout.app/Contents/MacOS/klayout",
    Path("/Applications/KLayout/klayout.app/Contents/MacOS/klayout"),
    Path("/Applications/klayout.app/Contents/MacOS/klayout"),
)

LYP = Path("libs.tech/klayout/tech/sky130A.lyp")


def klayout_bin() -> Path:
    for p in KLAYOUT_CANDIDATES:
        if p.is_file():
            return p
    raise RuntimeError(
        "KLayout application not found. Tried:\n  "
        + "\n  ".join(str(p) for p in KLAYOUT_CANDIDATES)
        + "\nInstall it with `brew install --cask klayout`. The `klayout` pip module is "
          "not enough: it ships the database layer only, with no renderer."
    )


def lyp_path() -> Path:
    p = X.sky130_root() / LYP
    if not p.is_file():
        raise RuntimeError(f"sky130 layer properties not found at {p}")
    return p


def render(gds: Path, png: Path, w: int, h: int) -> None:
    """Rasterize one GDS through the KLayout application, PDK colours applied."""
    script = (
        "app = RBA::Application.instance\n"
        "mw = app.main_window\n"
        f"mw.load_layout({str(gds)!r}, 1)\n"
        "lv = mw.current_view\n"
        f"lv.load_layer_props({str(lyp_path())!r})\n"
        "lv.max_hier\n"
        "lv.zoom_fit\n"
        # A little air around the die edge; zoom_fit alone clips flush to the bbox.
        "lv.zoom_box(lv.box.enlarged(lv.box.width * 0.04, lv.box.height * 0.04))\n"
        f"lv.save_image({str(png)!r}, {w}, {h})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".rb", delete=False) as fh:
        fh.write(script)
        rb = fh.name
    r = subprocess.run([str(klayout_bin()), "-z", "-nc", "-r", rb],
                       capture_output=True, text=True, timeout=600)
    Path(rb).unlink(missing_ok=True)
    if not png.is_file():
        raise RuntimeError(f"render produced no image for {gds.name}\n"
                           f"stdout: {r.stdout[-800:]}\nstderr: {r.stderr[-800:]}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tech = O.TechConstants.parse()
    layers = O.Layers()
    rules = C.cell_rules()
    dev = C.res_devices(tech)[0]

    # 1 uS is the project default; 20 uS is the measured optimum from cell_layout's
    # g_min sweep. Drawing both is the point -- the area difference is the finding.
    views = []
    for g_min, tag in ((1e-6, "1uS"), (20e-6, "20uS")):
        sp = C.plan_cell(tech, 1.0 / g_min, dev, rules)
        gds = OUT / f"cell-{tag}.gds"
        C.cell_gds(gds, sp, layers)
        views.append((gds, OUT / f"cell-{tag}.png", 1000, 900, sp, g_min))
        bw, bh = sp.block
        print(f"  g_min {g_min*1e6:>4.0f} uS  R {1/g_min/1e3:>7.1f} kohm  "
              f"{sp.stripes:>2d} stripes  cell {bw:.2f} x {bh:.2f} um  "
              f"pitch {sp.pitch:.2f} um  (limit: {sp.pitch_limit}) -> {gds.name}")

    # The array at the default g_min: 4x4 to actually see the structure, and 16x16 for
    # the tiling. Both instance the same unit cell.
    sp1 = C.plan_cell(tech, 1.0 / 1e-6, dev, rules)
    for n, px in ((4, 1100), (16, 1600)):
        gds = OUT / f"array-{n}x{n}.gds"
        C.array_gds(gds, sp1, n, layers, labels=True)
        views.append((gds, OUT / f"array-{n}x{n}.png", px, px, sp1, 1e-6))
        print(f"  array {n}x{n}  {n*sp1.pitch:.1f} um across -> {gds.name}")

    print()
    for gds, png, w, h, *_ in views:
        render(gds, png, w, h)
        print(f"  rendered {png.name:<20s} {png.stat().st_size/1024:>7.1f} KB   "
              f"(gds {gds.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
