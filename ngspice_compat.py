"""Compatibility shim: ngspice >= 43 rawfile headers vs. vlsirtools 7.0.0.

vlsirtools' `parse_nutbin` assumes the rawfile header is exactly three lines
before `Flags:` --- `Title:`, `Date:`, `Plotname:` --- and reads them by count:

    Title: ...
    Date: ...
    Plotname: Operating Point
    Flags: real

Modern ngspice (checked on ngspice-47) emits an extra `Command:` line:

    Title: ...
    Date: ...
    Command: ngspice-47, Build
    Plotname: Operating Point
    Flags: real

so the line-counting parser consumes `Command:` as the plotname and then trips
on `Plotname:` where it wants `Flags:`, raising
`ValueError: Invalid flags ['Plotname:', 'Operating', 'Point']`.

Fix: scan forward to the `Plotname:` line instead of counting lines. That is
tolerant of any number of leading header lines, so it works on both old and new
ngspice. Everything after `Plotname:` is left to the upstream parser.

`vlsirtools.spice.ngspice.NgSpiceSim.parse_results` looks `parse_nutbin` up as a
module global, so rebinding the module attribute is enough.
"""

from typing import IO, Mapping

from vlsirtools.spice import ngspice as _ng

_PLOTNAME = b"Plotname:"


def _parse_nutbin(f: IO) -> Mapping[str, "_ng.NutBinAnalysis"]:
    """Header-tolerant replacement for `vlsirtools...ngspice.parse_nutbin`."""
    analyses = {}
    while True:
        # Scan forward to the next `Plotname:` line, skipping whatever header
        # lines this ngspice build happens to emit. EOF => no more analyses.
        plotname = None
        for line in iter(f.readline, b""):
            if line.startswith(_PLOTNAME):
                plotname = line.decode("ascii")
                break
        if plotname is None:
            break
        # Upstream keys its results on the raw `Plotname: ...\n` line, and
        # `parse_results` looks them up by that exact string. Pass it through.
        an = _ng.parse_nutbin_analysis(f, plotname)
        analyses[an.analysis_name] = an
    return analyses


def apply() -> None:
    """Install the shim. Idempotent; a no-op if upstream is already fixed."""
    if getattr(_ng.parse_nutbin, "_is_compat_shim", False):
        return
    _parse_nutbin._is_compat_shim = True
    _ng.parse_nutbin = _parse_nutbin
