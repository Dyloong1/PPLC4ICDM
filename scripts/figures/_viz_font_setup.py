"""Shared font setup for every paper figure.

Selects a Times-style serif (Nimbus Roman if available, falling back
through Liberation Serif / DejaVu Serif), and switches matplotlib's
``mathtext`` to STIX so subscripts and superscripts render in a
matching serif.

Import this module before constructing any ``Figure`` in any figure
script in this directory so the whole rcParams stack is set before
any artist is created.
"""

import matplotlib
import matplotlib.font_manager as fm

_PREFERRED_TIMES = [
    "Nimbus Roman",        # URW base-35; ships with Ghostscript on most Linux distros
    "Times New Roman",     # proprietary; only present if explicitly installed
    "Liberation Serif",    # Red Hat's free TNR-compatible alternative
    "DejaVu Serif",        # final matplotlib-bundled fallback
]


def _pick_serif() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _PREFERRED_TIMES:
        if name in available:
            return name
    return "serif"


_FAMILY = _pick_serif()

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [_FAMILY, "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})


def font_family() -> str:
    """Return the actual font family in use (for diagnostic prints)."""
    return _FAMILY
