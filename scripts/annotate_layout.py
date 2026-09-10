"""Annotate the drawn layout: which polygon is which, at three zoom levels.

Renders each panel through KLayout with an *explicit* zoom box whose aspect matches the
output pixel aspect, so the world->pixel mapping is exact. matplotlib then draws the image
with `extent` set to that same world box, which means every annotation below is placed in
microns and lands on the geometry rather than being positioned by eye.

    uv run scripts/annotate_layout.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cell_layout as C
import layout_oracle as O
from render_layout import klayout_bin, lyp_path

OUT = Path(__file__).resolve().parent.parent / "docs" / "layout"
GDS_ARRAY = OUT / "array-4x4.gds"
GDS_CELL = OUT / "cell-1uS.gds"

INK = "#111111"
BOX = dict(boxstyle="round,pad=0.28", fc="white", ec="#555555", lw=0.7, alpha=0.94)


def render_box(gds: Path, box: tuple, px_w: int) -> tuple:
    """Render `gds` over the world rect `box` = (x0, y0, x1, y1) in um.

    Returns (png_path, box). Pixel height is derived from the box aspect so the mapping
    from microns to pixels is exact and `extent` can be trusted.
    """
    x0, y0, x1, y1 = box
    px_h = int(round(px_w * (y1 - y0) / (x1 - x0)))
    png = Path(tempfile.mkdtemp()) / f"{gds.stem}.png"
    script = (
        "app = RBA::Application.instance\n"
        "mw = app.main_window\n"
        f"mw.load_layout({str(gds)!r}, 1)\n"
        "lv = mw.current_view\n"
        f"lv.load_layer_props({str(lyp_path())!r})\n"
        "lv.max_hier\n"
        f"lv.zoom_box(RBA::DBox.new({x0}, {y0}, {x1}, {y1}))\n"
        f"lv.save_image({str(png)!r}, {px_w}, {px_h})\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".rb", delete=False) as fh:
        fh.write(script)
        rb = fh.name
    subprocess.run([str(klayout_bin()), "-z", "-nc", "-r", rb],
                   capture_output=True, text=True, timeout=600)
    Path(rb).unlink(missing_ok=True)
    if not png.is_file():
        raise RuntimeError(f"no image rendered for {gds}")
    return png, box


def show(ax, gds: Path, box: tuple, px_w: int, title: str):
    png, box = render_box(gds, box, px_w)
    img = plt.imread(png)
    x0, y0, x1, y1 = box
    ax.imshow(img, extent=(x0, x1, y0, y1), origin="upper", interpolation="antialiased")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xlabel("µm", fontsize=8)
    ax.tick_params(labelsize=7)


def tag(ax, xy, xytext, text, color=INK, fs=8.4):
    """A labelled leader line, both ends in microns."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, color=INK, bbox=BOX,
                ha="center", va="center", zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=0.12"))


def main() -> int:
    tech = O.TechConstants.parse()
    sp = C.plan_cell(tech, 1e6, C.res_devices(tech)[0], C.cell_rules())
    p, (bw, bh) = sp.pitch, sp.block
    ox, oy = float(sp.origin[0]), float(sp.origin[1])
    wr, wc = sp.w_row, sp.w_col

    # Stripe centrelines and head centres, mirroring draw_cell's arithmetic so the
    # annotations sit on the real polygons.
    sx = (bw - sp.dev.width) / (sp.stripes - 1)
    x_c = [ox + j * sx + sp.dev.width / 2 for j in range(sp.stripes)]
    y_head_lo = oy + sp.head / 2
    y_isl0, isl_h = C.tap_stack(C.cell_rules(), wr)

    fig = plt.figure(figsize=(16.5, 11.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.40], hspace=0.20, wspace=0.10,
                          left=0.045, right=0.985, top=0.915, bottom=0.055)

    # ---------------------------------------------------------------- panel 1: the array
    ax = fig.add_subplot(gs[0, 0])
    show(ax, GDS_ARRAY, (0, 0, 4 * p, 4 * p), 900,
         "1.  the 4 × 4 array — rails and cells")
    # Row rails are horizontal met1 at the bottom of each pitch; columns are vertical met2
    # up the left of each pitch.
    tag(ax, (3.1 * p, wr / 2), (2.55 * p, 0.42 * p),
        "row rail  (met1, horizontal)\ndriven with input x[i]")
    tag(ax, (wc / 2, 2.6 * p), (0.95 * p, 3.15 * p),
        "column rail  (met2, vertical)\nsummed current out to the TIA")
    ax.add_patch(Rectangle((p + ox, p + oy), bw, bh, fill=False, ec="#0072B2",
                           lw=1.6, zorder=5))
    tag(ax, (p + ox + bw / 2, p + oy + bh), (1.98 * p, 2.62 * p),
        "one cell = one resistor\nat one intersection", color="#0072B2")
    # Pitch dimension line along the top.
    yb = 4 * p - 0.9
    ax.add_patch(FancyArrowPatch((2 * p, yb), (3 * p, yb), arrowstyle="<|-|>",
                                 mutation_scale=9, color="#D55E00", lw=1.4, zorder=6))
    ax.text(2.5 * p, yb - 2.0, f"pitch {p:.2f} µm", fontsize=8.4, color=INK,
            ha="center", va="top", bbox=BOX, zorder=6)

    # ------------------------------------------------------- panel 2: one cell, one pitch
    ax = fig.add_subplot(gs[0, 1])
    show(ax, GDS_CELL, (0, 0, p, p), 900,
         "2.  one cell — a 1 MΩ resistor folded into 16 series stripes")
    tag(ax, (x_c[7], oy + bh / 2), (0.66 * p, 0.56 * p),
        f"{sp.stripes} poly stripes, {sp.dev.width} µm wide\n"
        f"res_xhigh_po at 2000 Ω/□\n(pink outline = the poly_rs marker,\nwhich is what sets"
        " the extracted L)")
    tag(ax, (x_c[13], oy + bh - sp.head / 2), (0.72 * p, 0.90 * p),
        "licon contacts,\nboth ends of every stripe")
    tag(ax, (x_c[2], oy + bh - sp.head / 2), (0.235 * p, 0.905 * p),
        "li1 straps alternate top and bottom —\nthat is the series fold, and it needs\n"
        "no poly corners")
    tag(ax, (p - 1.4, wr / 2), (0.74 * p, 0.155 * p), "row rail (met1)", color="#0072B2")
    tag(ax, (wc / 2, 0.50 * p), (0.245 * p, 0.50 * p), "column rail (met2)",
        color="#CC79A7")

    # ------------------------------ panel 3: both terminals, a wide strip along the bottom
    ax = fig.add_subplot(gs[1, :])
    zy = 3.05
    show(ax, GDS_CELL, (0, 0, p, zy), 1900,
         "3.  the bottom strip: the cell's two terminals, at opposite ends")
    tag(ax, (x_c[0], y_head_lo), (3.05, 2.62),
        "stripe 0 → li1 → mcon → met1 island → via → met2\n"
        "the COLUMN terminal", color="#CC79A7", fs=8.2)
    tag(ax, (wc / 2, y_isl0 + isl_h / 2), (1.35, 1.95),
        "the only via in the cell", color="#009E73", fs=8.2)
    tag(ax, (x_c[-1], wr / 2), (10.5, 2.62),
        "last stripe → li1 → mcon → met1 row rail\nthe ROW terminal",
        color="#0072B2", fs=8.2)
    tag(ax, (0.62, y_isl0 - 0.075), (6.55, 1.42),
        f"island bottom {y_isl0:.2f} µm vs row-rail top {wr:.2f} µm\n"
        f"= {y_isl0 - wr:.2f} µm, against the m1.2 minimum of 0.14 —\n"
        "both are met1, so this is a spacing rule", color="#D55E00", fs=8.0)

    fig.suptitle("sky130 0T1R crossbar layout — what each polygon is   "
                 "(layer colours are the PDK's own sky130A.lyp)",
                 fontsize=13.5, y=0.975)
    fig.text(0.5, 0.945,
             "row rails are horizontal met1, one per row, driven with the input voltage.  "
             "column rails are vertical met2, one per column, held at virtual ground.  "
             "they cross at every cell on different metal layers — no via there — which is "
             "what makes a crossbar legal.",
             fontsize=9.2, color="#333333", ha="center", va="center")
    out = OUT / "annotated.png"
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
