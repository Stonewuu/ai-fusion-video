#!/usr/bin/env python3
"""Generate the approved Rongguang Assistant logo as deterministic SVG.

The geometry is formula-driven and raster-free:

1. One leaf is the outer circle minus a displaced construction circle.
2. The construction-circle radius is ``sqrt(3) * offset``.
3. Rotating the leaf by 120 degrees creates the other two leaves; the shared
   negative space is a Reuleaux triangle whose extended arcs are the seams.
4. Nested clips complete the cyclic cyan > coral > dark > cyan overlap.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


CANONICAL_SIZE = 1254
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "ai-fusion-video-web"
    / "public"
    / "logo-candidates"
    / "rongguang-assistant-chat-glow.svg"
)


def fmt(value: float) -> str:
    """Format a coordinate compactly with deterministic millipixel precision."""
    rounded = round(value, 3)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def circle_path(cx: float, cy: float, radius: float) -> str:
    """Return a closed clockwise circle composed of two SVG elliptical arcs."""
    top = cy - radius
    bottom = cy + radius
    return (
        f"M{fmt(cx)} {fmt(top)} "
        f"A{fmt(radius)} {fmt(radius)} 0 1 1 {fmt(cx)} {fmt(bottom)} "
        f"A{fmt(radius)} {fmt(radius)} 0 1 1 {fmt(cx)} {fmt(top)}Z"
    )


def build_svg(size: int, include_background: bool) -> str:
    """Build the approved logo at ``size`` square CSS pixels."""
    scale = size / CANONICAL_SIZE

    def s(value: float) -> str:
        return fmt(value * scale)

    cx = 627.0 * scale
    cy = 595.0 * scale
    outer_radius = 386.0 * scale
    reuleaux_vertex_radius = 250.0 * scale
    construction_offset = 120.0 * scale
    construction_radius = math.sqrt(
        construction_offset**2
        + construction_offset * reuleaux_vertex_radius
        + reuleaux_vertex_radius**2
    )
    construction_cy = cy + construction_offset

    leaf = " ".join(
        (
            circle_path(cx, cy, outer_radius),
            circle_path(cx, construction_cy, construction_radius),
        )
    )

    # Only clip geometry is expanded by half a canonical pixel to prevent
    # antialiasing seams where the same coral fill is redrawn over an overlap.
    clip_leaf = " ".join(
        (
            circle_path(cx, cy, 386.5 * scale),
            circle_path(
                cx,
                construction_cy,
                construction_radius - 0.5 * scale,
            ),
        )
    )

    tail = (
        f"M{s(360)} {s(800)} "
        f"C{s(355)} {s(850)} {s(320)} {s(900)} {s(270)} {s(938)} "
        f"C{s(254)} {s(948)} {s(243)} {s(958)} {s(242)} {s(967)} "
        f"C{s(241)} {s(975)} {s(249)} {s(979)} {s(264)} {s(980)} "
        f"C{s(338)} {s(981)} {s(426)} {s(976)} {s(505)} {s(956)} "
        f"C{s(530)} {s(949)} {s(550)} {s(938)} {s(566)} {s(925)} "
        f"C{s(505)} {s(870)} {s(430)} {s(825)} {s(360)} {s(800)}Z"
    )

    sparkle = (
        f"M{s(638)} {s(445)} "
        f"C{s(629)} {s(445)} {s(631)} {s(483)} {s(618)} {s(513)} "
        f"C{s(600)} {s(552)} {s(559)} {s(573)} {s(499)} {s(580)} "
        f"C{s(488)} {s(581)} {s(488)} {s(592)} {s(496)} {s(596)} "
        f"C{s(556)} {s(608)} {s(607)} {s(641)} {s(624)} {s(704)} "
        f"C{s(627)} {s(716)} {s(627)} {s(734)} {s(638)} {s(734)} "
        f"C{s(648)} {s(734)} {s(648)} {s(716)} {s(651)} {s(704)} "
        f"C{s(668)} {s(641)} {s(719)} {s(608)} {s(779)} {s(596)} "
        f"C{s(788)} {s(594)} {s(788)} {s(582)} {s(778)} {s(580)} "
        f"C{s(717)} {s(573)} {s(676)} {s(552)} {s(658)} {s(513)} "
        f"C{s(645)} {s(483)} {s(647)} {s(445)} {s(638)} {s(445)}Z"
    )

    background = ""
    if include_background:
        background = (
            f'  <rect width="{size}" height="{size}" '
            'fill="url(#background)"/>\n\n'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-labelledby="title desc">
  <title id="title">融光助手 Logo</title>
  <desc id="desc">三片相隔一百二十度并循环覆盖的渐变叶片组成圆形莫比乌斯对话气泡，构造圆交集形成鲁洛克斯三角形负空间。</desc>

  <defs>
    <radialGradient id="background" cx="{s(627)}" cy="{s(539)}" r="{s(980)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#fffaf2"/>
      <stop offset="0.58" stop-color="#fef9ef"/>
      <stop offset="1" stop-color="#fcf7eb"/>
    </radialGradient>
    <linearGradient id="cyanLeaf" x1="{s(350)}" y1="{s(260)}" x2="{s(720)}" y2="{s(980)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#27b3c7"/>
      <stop offset="0.5" stop-color="#20abc2"/>
      <stop offset="1" stop-color="#25adc2"/>
    </linearGradient>
    <linearGradient id="darkLeaf" x1="{s(900)}" y1="{s(470)}" x2="{s(580)}" y2="{s(970)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0291b1"/>
      <stop offset="0.5" stop-color="#0589ad"/>
      <stop offset="1" stop-color="#0879a1"/>
    </linearGradient>
    <linearGradient id="coralLeaf" x1="{s(530)}" y1="{s(230)}" x2="{s(1010)}" y2="{s(780)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ff8a6a"/>
      <stop offset="0.5" stop-color="#fe8467"/>
      <stop offset="1" stop-color="#fc8165"/>
    </linearGradient>
    <linearGradient id="tailGradient" x1="{s(505)}" y1="{s(895)}" x2="{s(325)}" y2="{s(1015)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0785a8"/>
      <stop offset="0.42" stop-color="#0b91ae"/>
      <stop offset="0.74" stop-color="#1ca7bc"/>
      <stop offset="1" stop-color="#28b2c4"/>
    </linearGradient>
    <radialGradient id="lightSpirit" cx="{s(620)}" cy="{s(565)}" r="{s(180)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ffc65a"/>
      <stop offset="0.58" stop-color="#fdbe54"/>
      <stop offset="1" stop-color="#fbbc52"/>
    </radialGradient>
    <linearGradient id="eye" x1="0" y1="{s(575)}" x2="0" y2="{s(609)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0e3768"/>
      <stop offset="1" stop-color="#082b56"/>
    </linearGradient>

    <clipPath id="outerCircle" clipPathUnits="userSpaceOnUse">
      <circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(outer_radius)}"/>
    </clipPath>
    <path id="leaf" fill-rule="evenodd" clip-rule="evenodd" d="{leaf}"/>
    <path id="leafClipExpanded" fill-rule="evenodd" clip-rule="evenodd" d="{clip_leaf}"/>
    <clipPath id="coralLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(60 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="darkLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(180 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="cyanLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(300 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="upperRightOverlap" clipPathUnits="userSpaceOnUse">
      <rect x="{fmt(cx)}" y="0" width="{fmt(size - cx)}" height="{fmt(cy)}"/>
    </clipPath>
    <clipPath id="upperLeftOverlap" clipPathUnits="userSpaceOnUse">
      <rect x="0" y="0" width="{fmt(cx)}" height="{fmt(cy)}"/>
    </clipPath>
  </defs>

{background}  <path d="{tail}" fill="url(#tailGradient)"/>

  <g clip-path="url(#outerCircle)">
    <use href="#leaf" transform="rotate(60 {fmt(cx)} {fmt(cy)})" fill="url(#coralLeaf)"/>
    <use href="#leaf" transform="rotate(300 {fmt(cx)} {fmt(cy)})" fill="url(#cyanLeaf)"/>
    <use href="#leaf" transform="rotate(180 {fmt(cx)} {fmt(cy)})" fill="url(#darkLeaf)"/>
    <g clip-path="url(#coralLeafClip)">
      <g clip-path="url(#darkLeafClip)">
        <use href="#leaf" transform="rotate(60 {fmt(cx)} {fmt(cy)})" fill="url(#coralLeaf)"/>
      </g>
    </g>
    <g clip-path="url(#coralLeafClip)">
      <g clip-path="url(#darkLeafClip)">
        <g clip-path="url(#cyanLeafClip)">
          <g clip-path="url(#upperRightOverlap)">
            <use href="#leaf" transform="rotate(300 {fmt(cx)} {fmt(cy)})" fill="url(#cyanLeaf)"/>
          </g>
          <g clip-path="url(#upperLeftOverlap)">
            <use href="#leaf" transform="rotate(180 {fmt(cx)} {fmt(cy)})" fill="url(#darkLeaf)"/>
          </g>
        </g>
      </g>
    </g>
  </g>

  <path d="{sparkle}" fill="url(#lightSpirit)"/>
  <ellipse cx="{s(610)}" cy="{s(592)}" rx="{s(11)}" ry="{s(17)}" fill="url(#eye)"/>
  <ellipse cx="{s(669.5)}" cy="{s(592)}" rx="{s(10.5)}" ry="{s(17)}" fill="url(#eye)"/>
</svg>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the formula-driven Rongguang Assistant logo SVG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"SVG output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=CANONICAL_SIZE,
        help=f"Canvas size in CSS pixels (default: {CANONICAL_SIZE}).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Omit the warm off-white background rectangle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 128:
        raise SystemExit("--size must be at least 128")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_svg(args.size, include_background=not args.transparent),
        encoding="utf-8",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
