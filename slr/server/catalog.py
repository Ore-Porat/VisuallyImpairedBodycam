"""The picture catalog for the shopping list web page: aisles and one illustration per item.

Every grocery class in ``config/classes.yaml`` gets a small flat illustration of its typical
package (a can for beans, a carton for milk, ...), drawn as SVG so it needs no image files and
stays sharp at any size. All drawings share one 160 x 160 canvas, one ground shadow, one light
direction and one palette, so the catalog reads as a single set.

The pictures are decoration for sighted helpers: the page always shows the item's name as text,
and the images carry an empty ``alt`` so screen readers do not read them twice.

A class without a drawing (for example from another classes file) gets a neutral shopping bag,
and an aisle not listed here goes into "Other".
"""

from __future__ import annotations

# --------------------------------------------------------------------------- palette

INK = "#2A2622"
CREAM = "#F4EDDF"
PAPER = "#FBF8F2"
METAL = "#CFCFCA"
METAL_DARK = "#A9A9A3"
RED = "#B8382B"
TOMATO = "#D2452F"
LEAF = "#4E7D3A"
MUSTARD = "#E2A33B"
NAVY = "#2D4B7A"
BLUE = "#3D6FA8"
SKY = "#D5E6EE"
BROWN = "#7A3E2B"
COCOA = "#4A2E24"
KRAFT = "#C9A57A"
GOLD = "#D9AE3A"
SAGE = "#5E7F5A"
WINE = "#6E2A35"
PLUM = "#9E3456"
CORN = "#F2C94C"
TEAL = "#2F6E73"

SHADOW = '<ellipse cx="80" cy="139" rx="42" ry="5" fill="#000" opacity=".09"/>'


def _svg(body: str, shadow: str = SHADOW) -> str:
    """Wrap drawing commands in the shared canvas."""
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160">'
            f"{shadow}{body}</svg>")


def _shine(x: float, y: float, h: float, w: float = 5) -> str:
    """The soft vertical highlight on the left of every package (light comes from the left)."""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{w / 2}" fill="#fff" opacity=".28"/>'


def _shade(x: float, y: float, h: float, w: float) -> str:
    """A darker strip on the right edge that gives the package some roundness."""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#000" opacity=".08"/>'


# --------------------------------------------------------------------------- emblems
# Small marks drawn on the label, centred on (cx, cy).

def _bean(cx: float, cy: float) -> str:
    return (f'<g fill="{BROWN}">'
            f'<ellipse cx="{cx - 6}" cy="{cy}" rx="6" ry="9" transform="rotate(-24 {cx - 6} {cy})"/>'
            f'<ellipse cx="{cx + 6}" cy="{cy + 1}" rx="6" ry="9" transform="rotate(20 {cx + 6} {cy + 1})"/>'
            "</g>"
            f'<ellipse cx="{cx - 8}" cy="{cy - 3}" rx="1.6" ry="3" fill="#fff" opacity=".35" '
            f'transform="rotate(-24 {cx - 8} {cy - 3})"/>')


def _cob(cx: float, cy: float) -> str:
    kernels = "".join(
        f'<circle cx="{cx - 4 + 4 * c}" cy="{cy - 10 + 5 * r}" r="1.5" fill="#C99A1E"/>'
        for r in range(5) for c in range(3))
    return (f'<path d="M{cx - 2} {cy + 16} C{cx - 16} {cy + 8} {cx - 15} {cy - 4} {cx - 11} {cy - 10} '
            f'C{cx - 9} {cy} {cx - 5} {cy + 8} {cx - 2} {cy + 16}Z" fill="{LEAF}"/>'
            f'<ellipse cx="{cx}" cy="{cy}" rx="7.5" ry="15" fill="{CORN}"/>{kernels}'
            f'<path d="M{cx + 2} {cy + 16} C{cx + 16} {cy + 8} {cx + 15} {cy - 4} {cx + 11} {cy - 10} '
            f'C{cx + 9} {cy} {cx + 5} {cy + 8} {cx + 2} {cy + 16}Z" fill="{LEAF}"/>')


def _tomato(cx: float, cy: float) -> str:
    return (f'<circle cx="{cx}" cy="{cy + 1}" r="11" fill="{TOMATO}"/>'
            f'<ellipse cx="{cx - 4}" cy="{cy - 3}" rx="2.5" ry="3.5" fill="#fff" opacity=".3"/>'
            f'<path d="M{cx} {cy - 10} l3 -4 l-1 4 l5 -1 l-4 3 l3 3 l-6 -2 l-3 3 l1 -4 l-5 0 l4 -2Z" '
            f'fill="{LEAF}"/>')


def _fish(cx: float, cy: float, color: str = CREAM) -> str:
    return (f'<path d="M{cx - 16} {cy} C{cx - 8} {cy - 10} {cx + 6} {cy - 10} {cx + 12} {cy} '
            f'C{cx + 6} {cy + 10} {cx - 8} {cy + 10} {cx - 16} {cy}Z" fill="{color}"/>'
            f'<path d="M{cx + 10} {cy} L{cx + 20} {cy - 8} L{cx + 19} {cy + 8}Z" fill="{color}"/>'
            f'<circle cx="{cx - 9}" cy="{cy - 2}" r="1.6" fill="{NAVY}"/>')


def _drop(cx: float, cy: float, color: str) -> str:
    return (f'<path d="M{cx} {cy - 11} C{cx + 4} {cy - 4} {cx + 8} {cy} {cx + 8} {cy + 4} '
            f'A8 8 0 0 1 {cx - 8} {cy + 4} C{cx - 8} {cy} {cx - 4} {cy - 4} {cx} {cy - 11}Z" '
            f'fill="{color}"/>')


def _wheat(cx: float, cy: float, color: str) -> str:
    grains = "".join(
        f'<ellipse cx="{cx + s * 4}" cy="{cy - 10 + 6 * i}" rx="2.6" ry="4.6" fill="{color}" '
        f'transform="rotate({s * 28} {cx + s * 4} {cy - 10 + 6 * i})"/>'
        for i in range(3) for s in (-1, 1))
    return (f'<rect x="{cx - .8}" y="{cy - 14}" width="1.6" height="32" rx=".8" fill="{color}"/>'
            f'<ellipse cx="{cx}" cy="{cy - 16}" rx="2.4" ry="4.4" fill="{color}"/>{grains}')


# --------------------------------------------------------------------------- package shapes

def _can(label: str, emblem: str, band: str | None = None, w: float = 62,
         top: float = 50, bottom: float = 136) -> str:
    """A food can seen from the front: metal rims, a paper label and an emblem."""
    x = 80 - w / 2
    h = bottom - top
    band_svg = ""
    if band:
        band_svg = (f'<rect x="{x}" y="{top + 8}" width="{w}" height="6" fill="{band}"/>'
                    f'<rect x="{x}" y="{bottom - 14}" width="{w}" height="6" fill="{band}"/>')
    return (f'<rect x="{x}" y="{top + 4}" width="{w}" height="{h - 8}" fill="{label}"/>'
            f"{band_svg}"
            f'<rect x="{x - 2}" y="{top}" width="{w + 4}" height="9" rx="4.5" fill="{METAL}"/>'
            f'<rect x="{x - 2}" y="{bottom - 9}" width="{w + 4}" height="9" rx="4.5" fill="{METAL}"/>'
            f'<rect x="{x}" y="{top + 2}" width="{w}" height="1.5" fill="{METAL_DARK}"/>'
            f"{_shade(x + w - 9, top + 9, h - 18, 9)}"
            f"{emblem}"
            f"{_shine(x + 6, top + 12, h - 24)}")


def _jar(body: str, lid: str, label: str, emblem: str, w: float = 58,
         top: float = 56, bottom: float = 136, lid_svg: str | None = None) -> str:
    """A glass jar: a rounded body, a screw lid and an oval label."""
    x = 80 - w / 2
    h = bottom - top
    lid_draw = lid_svg or (
        f'<rect x="{x + 4}" y="{top - 12}" width="{w - 8}" height="15" rx="3" fill="{lid}"/>'
        f'<rect x="{x + 4}" y="{top - 3}" width="{w - 8}" height="2" fill="#000" opacity=".12"/>')
    return (f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="12" fill="{body}"/>'
            f"{lid_draw}"
            f'<ellipse cx="80" cy="{top + h * .55}" rx="{w * .36}" ry="{h * .26}" fill="{label}"/>'
            f"{emblem}"
            f'<rect x="{x + w - 9}" y="{top + 10}" width="6" height="{h - 22}" rx="3" fill="#000" opacity=".08"/>'
            f"{_shine(x + 6, top + 10, h - 22)}")


def _bottle(body: str, cap: str, label: str, emblem: str, w: float = 40,
            neck: float = 14, top: float = 22, shoulder: float = 62, bottom: float = 136,
            label_top: float | None = None, label_bottom: float | None = None) -> str:
    """A bottle with a narrow neck, sloped shoulders, a cap and a label band."""
    x = 80 - w / 2
    nx = 80 - neck / 2
    lt = label_top if label_top is not None else shoulder + 14
    lb = label_bottom if label_bottom is not None else bottom - 14
    path = (f"M{nx} {top + 10} L{nx} {shoulder - 16} C{nx} {shoulder - 6} {x} {shoulder - 6} {x} {shoulder + 6} "
            f"L{x} {bottom - 6} Q{x} {bottom} {x + 6} {bottom} L{x + w - 6} {bottom} "
            f"Q{x + w} {bottom} {x + w} {bottom - 6} L{x + w} {shoulder + 6} "
            f"C{x + w} {shoulder - 6} {nx + neck} {shoulder - 6} {nx + neck} {shoulder - 16} "
            f"L{nx + neck} {top + 10}Z")
    return (f'<path d="{path}" fill="{body}"/>'
            f'<rect x="{nx - 1.5}" y="{top}" width="{neck + 3}" height="13" rx="2.5" fill="{cap}"/>'
            f'<rect x="{x}" y="{lt}" width="{w}" height="{lb - lt}" fill="{label}"/>'
            f"{emblem}"
            f"{_shade(x + w - 7, shoulder + 4, bottom - shoulder - 10, 7)}"
            f"{_shine(x + 5, shoulder + 6, bottom - shoulder - 18, 4.5)}")


def _box(face: str, x: float, y: float, w: float, h: float, inner: str = "",
         top_strip: str | None = None) -> str:
    """A printed carton seen square on, with a darker lid strip."""
    strip = (f'<rect x="{x}" y="{y}" width="{w}" height="8" fill="{top_strip}"/>'
             if top_strip else "")
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2.5" fill="{face}"/>{strip}{inner}'
            f"{_shade(x + w - 8, y, h, 8)}"
            f"{_shine(x + 5, y + 12, h - 22)}")


def _paper_bag(face: str, band: str, emblem: str) -> str:
    """A flat-bottomed paper sack with a folded top (flour, sugar)."""
    return (f'<path d="M48 52 L112 52 L114 136 L46 136Z" fill="{face}"/>'
            f'<path d="M48 40 L112 40 L112 54 L48 54Z" fill="{face}"/>'
            f'<path d="M48 52 L112 52 L112 56 L48 56Z" fill="#000" opacity=".10"/>'
            f'<rect x="47" y="112" width="66" height="10" fill="{band}"/>'
            f"{emblem}"
            f'<path d="M104 56 L112 56 L114 136 L105 136Z" fill="#000" opacity=".07"/>'
            f"{_shine(53, 62, 66)}")


def _pouch(face: str, crimp: str, emblem: str, top: float = 34, bottom: float = 136) -> str:
    """A pillow pack with crimped seals top and bottom (chips, candy)."""
    teeth_top = "".join(f"L{46 + 4 * i + 2} {top - 4} L{46 + 4 * (i + 1)} {top} " for i in range(17))
    teeth_bot = "".join(f"L{114 - 4 * i - 2} {bottom + 4} L{114 - 4 * (i + 1)} {bottom} "
                        for i in range(17))
    outline = (f"M46 {top} {teeth_top}L114 {top} L114 {top + 10} C120 {top + 40} 120 {bottom - 40} "
               f"114 {bottom - 10} L114 {bottom} {teeth_bot}L46 {bottom} L46 {bottom - 10} "
               f"C40 {bottom - 40} 40 {top + 40} 46 {top + 10}Z")
    return (f'<path d="{outline}" fill="{face}"/>'
            f'<rect x="46" y="{top}" width="68" height="9" fill="{crimp}"/>'
            f'<rect x="46" y="{bottom - 9}" width="68" height="9" fill="{crimp}"/>'
            f"{emblem}"
            f'<path d="M106 {top + 10} C112 {top + 40} 112 {bottom - 40} 106 {bottom - 10}Z" '
            f'fill="#000" opacity=".08"/>'
            f"{_shine(52, top + 20, bottom - top - 40)}")


def _standup(face: str, emblem: str, top: float = 36, bottom: float = 136) -> str:
    """A stand-up pouch with a zip and a gusseted base (coffee, nuts)."""
    return (f'<path d="M50 {top} L110 {top} L113 {bottom - 4} Q80 {bottom + 2} 47 {bottom - 4}Z" '
            f'fill="{face}"/>'
            f'<rect x="50" y="{top}" width="60" height="7" fill="#000" opacity=".14"/>'
            f'<rect x="50" y="{top + 11}" width="60" height="1.5" fill="#000" opacity=".16"/>'
            f"{emblem}"
            f'<path d="M102 {top + 13} L110 {top + 13} L113 {bottom - 4} L104 {bottom - 2}Z" '
            f'fill="#000" opacity=".07"/>'
            f"{_shine(55, top + 22, bottom - top - 40)}")


# --------------------------------------------------------------------------- the drawings

def _art() -> dict[str, str]:
    a: dict[str, str] = {}

    a["beans"] = _svg(_can(CREAM, _bean(80, 93), band=BROWN))
    a["corn"] = _svg(_can("#2F5D3A", _cob(80, 93), band=CORN))
    a["tomato_sauce"] = _svg(_can(RED, f'<circle cx="80" cy="93" r="17" fill="{CREAM}"/>{_tomato(80, 93)}',
                                  band=CREAM))
    a["fish"] = _svg(
        f'<rect x="28" y="96" width="104" height="40" rx="11" fill="{NAVY}"/>'
        f'<rect x="28" y="88" width="104" height="16" rx="8" fill="{METAL}"/>'
        f'<rect x="32" y="98" width="96" height="2" fill="#000" opacity=".12"/>'
        f'<circle cx="44" cy="96" r="6.5" fill="none" stroke="{METAL_DARK}" stroke-width="2.4"/>'
        f"{_fish(80, 119)}"
        f'{_shade(120, 104, 30, 8)}{_shine(35, 106, 22, 4.5)}')

    # Red can, white top half: the classic condensed soup can, with a gold medallion.
    a["chicken_noodle_soup"] = _svg(_can(
        PAPER,
        f'<rect x="49" y="59" width="62" height="35" fill="{RED}"/>'
        f'<path d="M58 72 C64 66 70 78 76 70 S88 66 92 72 S100 76 102 70" stroke="{PAPER}" '
        'stroke-width="3" fill="none" stroke-linecap="round"/>'
        f'<circle cx="80" cy="96" r="9" fill="{GOLD}"/><circle cx="80" cy="96" r="5.5" fill="none" '
        'stroke="#fff" stroke-width="1.2" opacity=".7"/>'
        f'<path d="M62 116 L98 116" stroke="{RED}" stroke-width="4" stroke-linecap="round"/>'))

    # Red can with a white wave.
    a["coca_cola_can"] = _svg(
        '<path d="M60 36 L100 36 L104 44 L104 130 L100 136 L60 136 L56 130 L56 44Z" fill="#C8463A"/>'
        f'<rect x="60" y="26" width="40" height="11" rx="4" fill="{METAL}"/>'
        f'<path d="M56 44 L60 37 L100 37 L104 44Z" fill="{METAL_DARK}"/>'
        f'<path d="M56 130 L104 130 L100 136 L60 136Z" fill="{METAL}"/>'
        f'<path d="M56 98 C70 84 88 108 104 90 L104 104 C88 120 70 98 56 112Z" fill="{CREAM}"/>'
        f'<g fill="{CREAM}" opacity=".85"><circle cx="70" cy="64" r="3"/><circle cx="80" cy="56" r="2"/>'
        '<circle cx="88" cy="70" r="2.5"/></g>'
        f'{_shade(96, 44, 86, 8)}{_shine(62, 48, 74, 4.5)}')
    a["soda"] = _svg(_can(
        TEAL,
        f'<g fill="{CREAM}" opacity=".9"><circle cx="70" cy="84" r="4"/><circle cx="84" cy="76" r="3"/>'
        '<circle cx="90" cy="96" r="5"/><circle cx="74" cy="104" r="2.5"/><circle cx="86" cy="112" r="3"/></g>',
        w=48, top=40))
    a["water"] = _svg(_bottle(
        SKY, BLUE, BLUE, _drop(80, 100, CREAM), w=38, neck=16, top=18, shoulder=56,
        label_top=88, label_bottom=114)
        + "".join(f'<rect x="61" y="{y}" width="38" height="1.5" fill="#fff" opacity=".7"/>'
                  for y in (70, 78, 122, 128)))
    a["juice"] = _svg(_bottle(
        "#EE9A2E", LEAF, CREAM,
        '<circle cx="80" cy="102" r="11" fill="#EE9A2E"/><circle cx="80" cy="102" r="8" fill="#F7C46B"/>'
        '<g stroke="#EE9A2E" stroke-width="1.4"><path d="M80 94 V110"/><path d="M72 102 H88"/>'
        '<path d="M74.5 96.5 L85.5 107.5"/><path d="M85.5 96.5 L74.5 107.5"/></g>',
        w=44, neck=18, top=30, shoulder=64))
    a["oil"] = _svg(_bottle(GOLD, INK, CREAM, _drop(80, 104, GOLD), w=36, neck=12, top=14,
                            shoulder=66, label_top=86, label_bottom=122))
    a["vinegar"] = _svg(_bottle(
        WINE, KRAFT, CREAM,
        f'<g fill="{WINE}"><circle cx="76" cy="98" r="4"/><circle cx="84" cy="98" r="4"/>'
        '<circle cx="80" cy="105" r="4"/><circle cx="72" cy="105" r="3.5"/><circle cx="88" cy="105" r="3.5"/>'
        '<circle cx="80" cy="112" r="3.5"/></g>'
        f'<path d="M80 94 C80 90 83 88 86 88" stroke="{LEAF}" stroke-width="1.8" fill="none"/>',
        w=32, neck=11, top=14, shoulder=60, label_top=86, label_bottom=122))
    a["milk"] = _svg(
        f'<path d="M52 58 L66 38 L94 38 L108 58Z" fill="#E7E3DA"/>'
        f'<rect x="64" y="28" width="32" height="11" fill="{PAPER}"/>'
        f'<rect x="52" y="58" width="56" height="78" fill="{PAPER}"/>'
        f'<rect x="52" y="104" width="56" height="32" fill="{BLUE}"/>'
        f'<path d="M52 104 C66 96 92 112 108 102 L108 108 C92 118 66 102 52 110Z" fill="{BLUE}"/>'
        f"{_drop(80, 80, BLUE)}"
        f'{_shade(100, 58, 78, 8)}{_shine(57, 66, 60, 4.5)}')

    a["cereal"] = _svg(_box(
        MUSTARD, 44, 32, 72, 104,
        f'<rect x="52" y="48" width="56" height="12" rx="6" fill="{CREAM}"/>'
        f'<g fill="#C9752B"><circle cx="70" cy="91" r="5"/><circle cx="81" cy="87" r="5.5"/>'
        '<circle cx="91" cy="92" r="4.5"/><circle cx="76" cy="95" r="4"/></g>'
        f'<path d="M58 96 L102 96 A22 20 0 0 1 58 96Z" fill="{PAPER}"/>'
        f'<path d="M58 96 L102 96" stroke="{BLUE}" stroke-width="3"/>',
        top_strip="#C9842A"))
    a["cheerios"] = _svg(_box(
        CORN, 44, 32, 72, 104,
        f'<rect x="52" y="44" width="56" height="16" rx="3" fill="{NAVY}"/>'
        + "".join(f'<circle cx="{x}" cy="{y}" r="5" fill="none" stroke="#C9842A" stroke-width="3.2"/>'
                  for x, y in ((66, 86), (80, 82), (94, 87), (72, 98), (88, 99)))
        + f'<path d="M58 104 L102 104 A22 18 0 0 1 58 104Z" fill="{PAPER}"/>',
        top_strip="#D9A62A"))
    a["tea"] = _svg(_box(
        SAGE, 38, 72, 84, 64,
        f'<path d="M80 88 C94 92 96 108 80 122 C66 108 66 92 80 88Z" fill="{CREAM}"/>'
        f'<path d="M80 92 V118" stroke="{SAGE}" stroke-width="1.6"/>',
        top_strip="#4D6B49")
        + f'<path d="M110 72 C112 60 118 56 124 58" stroke="{INK}" stroke-width="1.2" fill="none"/>'
          f'<rect x="119" y="56" width="14" height="16" rx="1.5" fill="{CREAM}" transform="rotate(12 126 64)"/>')
    a["pasta"] = _svg(_box(
        NAVY, 44, 40, 72, 96,
        f'<rect x="54" y="62" width="52" height="46" rx="8" fill="{CREAM}"/>'
        + "".join(f'<rect x="{x}" y="{y}" width="8" height="18" rx="2" fill="#E3B54E" '
                  f'transform="rotate({r} {x + 4} {y + 9})"/>'
                  for x, y, r in ((62, 70, 35), (76, 66, -20), (88, 74, 50), (68, 84, -40), (84, 86, 15)))
        + f'<rect x="54" y="116" width="52" height="5" rx="2.5" fill="{CREAM}" opacity=".6"/>',
        top_strip="#22395E"))
    # A stack of dark sandwich cookies with cream filling.
    a["oreo"] = _svg("".join(
        f'<rect x="50" y="{y + 8}" width="60" height="8" rx="2" fill="#FFF9EE"/>'
        f'<rect x="50" y="{y + 14}" width="60" height="2" fill="#000" opacity=".08"/>'
        f'<rect x="46" y="{y}" width="68" height="10" rx="5" fill="#5E3C2F"/>'
        f'<rect x="46" y="{y + 14}" width="68" height="10" rx="5" fill="#5E3C2F"/>'
        f'<rect x="54" y="{y + 3}" width="20" height="2.2" rx="1.1" fill="#fff" opacity=".2"/>'
        for y in (112, 86, 60)))
    a["cookies"] = _svg("".join(
        f'<ellipse cx="80" cy="{y}" rx="30" ry="9" fill="#C98A47"/>'
        f'<ellipse cx="80" cy="{y - 2}" rx="30" ry="8" fill="#DDA35E"/>'
        + "".join(f'<circle cx="{80 + dx}" cy="{y - 2 + dy}" r="2" fill="{COCOA}"/>'
                  for dx, dy in ((-14, -2), (-4, 3), (8, -3), (16, 2)))
        for y in (126, 108, 90, 72)))
    a["cake"] = _svg(
        f'<ellipse cx="80" cy="134" rx="46" ry="6" fill="{PAPER}"/>'
        '<rect x="42" y="78" width="76" height="54" rx="4" fill="#E9C79A"/>'
        '<rect x="42" y="102" width="76" height="6" fill="#B5643C"/>'
        '<path d="M40 78 Q40 66 52 66 L108 66 Q120 66 120 78 L120 86 Q114 94 108 86 Q102 96 94 86 '
        'Q86 98 78 86 Q70 96 62 86 Q55 94 48 86 Q44 92 40 86Z" fill="#F6EFE6"/>'
        f'<circle cx="80" cy="58" r="8" fill="{RED}"/>'
        f'<path d="M80 50 C82 42 88 38 92 38" stroke="{LEAF}" stroke-width="2" fill="none"/>'
        '<ellipse cx="77" cy="55" rx="2" ry="2.6" fill="#fff" opacity=".35"/>'
        f'{_shade(110, 86, 46, 8)}')
    a["candy"] = _svg(
        f'<path d="M58 96 L30 78 L36 96 L30 114Z" fill="{PLUM}"/>'
        f'<path d="M102 96 L130 78 L124 96 L130 114Z" fill="{PLUM}"/>'
        '<path d="M58 96 L30 78 L33 87Z" fill="#fff" opacity=".2"/>'
        f'<circle cx="80" cy="96" r="24" fill="{PLUM}"/>'
        f'<path d="M62 86 C72 74 92 76 98 90" stroke="{CREAM}" stroke-width="5" fill="none" '
        'stroke-linecap="round"/>'
        f'<path d="M62 102 C68 116 90 120 98 106" stroke="{CREAM}" stroke-width="5" fill="none" '
        'stroke-linecap="round" opacity=".7"/>'
        '<ellipse cx="72" cy="86" rx="4" ry="6" fill="#fff" opacity=".22"/>',
        shadow='<ellipse cx="80" cy="134" rx="34" ry="4" fill="#000" opacity=".09"/>')
    a["chocolate"] = _svg(
        f'<rect x="48" y="38" width="64" height="58" rx="3" fill="{COCOA}"/>'
        + "".join(f'<rect x="{52 + 20 * c}" y="{42 + 18 * r}" width="16" height="14" rx="2" fill="#5E3B2E"/>'
                  for r in range(3) for c in range(3))
        + '<path d="M44 82 L116 74 L116 136 L44 136Z" fill="#E8E2D6"/>'
          f'<path d="M44 88 L116 80 L116 136 L44 136Z" fill="{WINE}"/>'
          f'<rect x="56" y="104" width="48" height="16" rx="8" fill="{CREAM}"/>'
          f'{_shade(108, 80, 56, 8)}{_shine(50, 94, 34, 4.5)}')
    a["chips"] = _svg(_pouch(
        MUSTARD, RED,
        f'<path d="M58 72 L102 72 L102 82 L58 82Z" fill="{RED}"/>'
        '<path d="M66 104 C66 92 92 88 96 100 C100 112 76 122 70 114 C67 110 66 108 66 104Z" fill="#F6D27C"/>'
        '<path d="M70 104 C74 98 86 96 90 102" stroke="#D9A23A" stroke-width="1.6" fill="none"/>'))
    a["lays"] = _svg(_pouch(
        "#F4C531", "#D93B2B",
        '<circle cx="80" cy="86" r="21" fill="#D93B2B"/>'
        '<path d="M60 84 Q80 70 100 84 L100 90 Q80 78 60 90Z" fill="#F4C531"/>'
        '<path d="M68 112 C68 104 88 101 91 109 C94 117 76 124 72 118 C69 116 68 114 68 112Z" fill="#F8DC84"/>'))
    a["nuts"] = _svg(_standup(
        KRAFT,
        '<rect x="60" y="66" width="40" height="40" rx="20" fill="#F1E6D2"/>'
        + "".join(f'<ellipse cx="{x}" cy="{y}" rx="5" ry="7.5" fill="#A0623A" '
                  f'transform="rotate({r} {x} {y})"/>'
                  for x, y, r in ((72, 80, -30), (86, 78, 25), (78, 94, 70), (90, 92, -10)))
        + f'<rect x="62" y="114" width="36" height="6" rx="3" fill="{BROWN}"/>'))
    a["coffee"] = _svg(_standup(
        "#24504A",
        f'<circle cx="80" cy="92" r="20" fill="{KRAFT}"/>'
        f'<ellipse cx="80" cy="92" rx="8" ry="11.5" fill="{COCOA}" transform="rotate(25 80 92)"/>'
        f'<path d="M77 82 C82 88 78 96 83 102" stroke="{KRAFT}" stroke-width="1.6" fill="none"/>'
        '<circle cx="80" cy="60" r="3.5" fill="#000" opacity=".3"/>'))
    a["flour"] = _svg(_paper_bag(CREAM, "#B5643C", _wheat(80, 86, "#C9A057")))
    a["sugar"] = _svg(_paper_bag(
        PAPER, BLUE,
        f'<path d="M80 72 L96 80 L80 88 L64 80Z" fill="#E6EEF5"/>'
        f'<path d="M64 80 L80 88 L80 106 L64 98Z" fill="#C9D9E8"/>'
        f'<path d="M96 80 L80 88 L80 106 L96 98Z" fill="#A9C1D9"/>'))
    a["rice"] = _svg(
        '<path d="M64 46 C56 54 44 70 44 104 Q44 136 80 136 Q116 136 116 104 C116 70 104 54 96 46Z" '
        'fill="#EDE5D3"/>'
        '<path d="M66 34 L94 34 L96 46 L64 46Z" fill="#EDE5D3"/>'
        f'<rect x="62" y="44" width="36" height="6" rx="3" fill="{RED}"/>'
        f'<path d="M46 92 L114 92 L115 108 L45 108Z" fill="{RED}"/>'
        + "".join(f'<ellipse cx="{x}" cy="{y}" rx="2.2" ry="4.5" fill="#fff" '
                  f'transform="rotate({r} {x} {y})"/>'
                  for x, y, r in ((74, 100, -30), (80, 98, 0), (86, 100, 30)))
        + '<path d="M106 62 C114 76 116 92 116 104 Q116 128 102 134 C108 120 110 90 106 62Z" '
          'fill="#000" opacity=".07"/>'
        + _shine(52, 76, 40))
    a["honey"] = _svg(_jar(
        "#E3A132", "#6B4A2B", CREAM,
        f'<path d="M80 88 L90 94 L90 106 L80 112 L70 106 L70 94Z" fill="#E3A132"/>'
        '<path d="M80 93 L86 96.5 L86 103.5 L80 107 L74 103.5 L74 96.5Z" fill="#F4C766"/>',
        w=64, top=60))
    a["jam"] = _svg(_jar(
        "#A3283A", "", CREAM,
        f'<circle cx="80" cy="100" r="9" fill="{RED}"/>'
        '<g fill="#F4D58A"><circle cx="77" cy="98" r="1"/><circle cx="83" cy="99" r="1"/>'
        '<circle cx="80" cy="104" r="1"/></g>'
        f'<path d="M74 91 L80 94 L86 91 L82 89 L80 92 L78 89Z" fill="{LEAF}"/>',
        w=58, top=62,
        lid_svg=(
            '<path d="M44 58 L116 58 L110 72 Q80 78 50 72Z" fill="#F7F1E6"/>'
            + "".join(f'<rect x="{46 + 10 * i}" y="58" width="5" height="12" fill="{RED}" opacity=".85"/>'
                      for i in range(7))
            + '<rect x="48" y="66" width="64" height="2.5" fill="#E8DFCF"/>')))
    a["spices"] = _svg(
        f'<rect x="44" y="64" width="34" height="72" rx="6" fill="#C24F2A"/>'
        f'<rect x="44" y="50" width="34" height="16" rx="3" fill="{PAPER}"/>'
        f'<g fill="#B9B3A6"><circle cx="54" cy="57" r="1.4"/><circle cx="61" cy="57" r="1.4"/>'
        '<circle cx="68" cy="57" r="1.4"/></g>'
        f'<rect x="44" y="92" width="34" height="24" fill="{CREAM}"/>'
        f'{_shine(48, 70, 58, 4)}'
        f'<rect x="84" y="80" width="32" height="56" rx="6" fill="#7A8B3A"/>'
        f'<rect x="84" y="68" width="32" height="14" rx="3" fill="{INK}"/>'
        f'<rect x="84" y="102" width="32" height="20" fill="{CREAM}"/>'
        f'{_shine(88, 86, 44, 4)}{_shade(109, 82, 54, 7)}')
    return a


ART: dict[str, str] = _art()

FALLBACK_ART = _svg(
    f'<path d="M46 56 L114 56 L118 136 L42 136Z" fill="{KRAFT}"/>'
    f'<path d="M62 58 C62 34 98 34 98 58" stroke="{BROWN}" stroke-width="5" fill="none"/>'
    f'<path d="M106 56 L114 56 L118 136 L108 136Z" fill="#000" opacity=".08"/>'
    f"{_shine(50, 66, 58)}")


# --------------------------------------------------------------------------- aisles

# (aisle title, item ids in display order). Items not listed here go into "Other".
AISLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Pantry", ("flour", "sugar", "rice", "pasta", "oil", "vinegar", "spices", "honey")),
    ("Cans & jars", ("chicken_noodle_soup", "beans", "corn", "tomato_sauce", "fish", "jam")),
    ("Breakfast", ("cheerios", "cereal", "coffee", "tea")),
    ("Drinks", ("coca_cola_can", "soda", "water", "juice", "milk")),
    ("Snacks & sweets", ("lays", "chips", "oreo", "cookies", "chocolate", "candy", "nuts", "cake")),
)


def art_for(item: str) -> str:
    """The SVG drawing for a class id (a neutral bag if there is no drawing for it)."""
    return ART.get(item, FALLBACK_ART)


def aisles_for(known: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Group the known class ids into aisles, keeping only aisles that have items.

    known: class id -> spoken name (the shopping list's valid items).
    """
    placed: set[str] = set()
    out: list[tuple[str, list[str]]] = []
    for title, ids in AISLES:
        present = [i for i in ids if i in known]
        placed.update(present)
        if present:
            out.append((title, present))
    rest = sorted((i for i in known if i not in placed), key=lambda i: known[i])
    if rest:
        out.append(("Other", rest))
    return out
