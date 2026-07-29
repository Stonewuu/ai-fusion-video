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

    def rotate_point(x: float, y: float, degrees: float) -> tuple[float, float]:
        angle = math.radians(degrees)
        dx = x * scale - cx
        dy = y * scale - cy
        return (
            cx + dx * math.cos(angle) - dy * math.sin(angle),
            cy + dx * math.sin(angle) + dy * math.cos(angle),
        )

    coral_seam_start = rotate_point(530, 230, 60)
    coral_seam_end = rotate_point(1010, 780, 60)

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

    # Keep the visible bubble tail inside the outer circle's bounding box
    # (x >= 241, y <= 981). The closing curve stays inside the circle and is
    # covered by the fan, leaving one broad, slightly concave exposed edge.
    tail = (
        f"M{s(325)} {s(680)} "
        f"C{s(315)} {s(745)} {s(310)} {s(850)} {s(250)} {s(942)} "
        f"C{s(243)} {s(949)} {s(241)} {s(956)} {s(244)} {s(963)} "
        f"C{s(248)} {s(974)} {s(265)} {s(979)} {s(284)} {s(979)} "
        f"C{s(365)} {s(981)} {s(445)} {s(978)} {s(520)} {s(956)} "
        f"C{s(470)} {s(850)} {s(400)} {s(740)} {s(325)} {s(680)}Z"
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
    <linearGradient id="topLeaf" x1="{s(350)}" y1="{s(260)}" x2="{s(720)}" y2="{s(980)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#78afff"/>
      <stop offset="0.5" stop-color="#5c93f5"/>
      <stop offset="1" stop-color="#447ae5"/>
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
    <linearGradient id="coralSeam" x1="{fmt(coral_seam_start[0])}" y1="{fmt(coral_seam_start[1])}" x2="{fmt(coral_seam_end[0])}" y2="{fmt(coral_seam_end[1])}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#ff8a6a"/>
      <stop offset="0.5" stop-color="#fe8467"/>
      <stop offset="1" stop-color="#fc8165"/>
    </linearGradient>
    <linearGradient id="tailGradient" x1="{s(320)}" y1="{s(800)}" x2="{s(241)}" y2="{s(970)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0291b1"/>
      <stop offset="0.55" stop-color="#0291b1"/>
      <stop offset="0.8" stop-color="#159bb3"/>
      <stop offset="1" stop-color="#27b3c7"/>
    </linearGradient>
    <linearGradient id="eye" x1="0" y1="{s(575)}" x2="0" y2="{s(609)}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#0e3768"/>
      <stop offset="1" stop-color="#082b56"/>
    </linearGradient>

    <clipPath id="outerCircle" clipPathUnits="userSpaceOnUse">
      <circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(outer_radius)}"/>
    </clipPath>
    <path id="leaf" fill-rule="evenodd" clip-rule="evenodd" d="{leaf}"/>
    <path id="leafClipExpanded" fill-rule="evenodd" clip-rule="evenodd" d="{clip_leaf}"/>
    <circle id="darkConstructionCircle" cx="{fmt(cx)}" cy="{fmt(cy - construction_offset)}" r="{fmt(construction_radius)}"/>
    <clipPath id="coralLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(60 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="darkLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(180 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="topLeafClip" clipPathUnits="userSpaceOnUse">
      <use href="#leafClipExpanded" transform="rotate(300 {fmt(cx)} {fmt(cy)})"/>
    </clipPath>
    <clipPath id="upperRightOverlap" clipPathUnits="userSpaceOnUse">
      <rect x="{fmt(cx)}" y="0" width="{fmt(size - cx)}" height="{fmt(cy)}"/>
    </clipPath>
    <clipPath id="upperLeftOverlap" clipPathUnits="userSpaceOnUse">
      <rect x="0" y="0" width="{fmt(cx)}" height="{fmt(cy)}"/>
    </clipPath>
    <clipPath id="rightHalf" clipPathUnits="userSpaceOnUse">
      <rect x="{fmt(cx)}" y="0" width="{fmt(size - cx)}" height="{fmt(size)}"/>
    </clipPath>
    <mask id="outsideTopLeaf" x="0" y="0" width="{fmt(size)}" height="{fmt(size)}" maskUnits="userSpaceOnUse" maskContentUnits="userSpaceOnUse" style="mask-type:luminance">
      <rect width="{fmt(size)}" height="{fmt(size)}" fill="white"/>
      <use href="#leaf" transform="rotate(300 {fmt(cx)} {fmt(cy)})" fill="black"/>
    </mask>
  </defs>

{background}  <path d="{tail}" fill="url(#tailGradient)"/>

  <g clip-path="url(#outerCircle)">
    <use href="#leaf" transform="rotate(60 {fmt(cx)} {fmt(cy)})" fill="url(#coralLeaf)"/>
    <use href="#leaf" transform="rotate(300 {fmt(cx)} {fmt(cy)})" fill="url(#topLeaf)"/>
    <use href="#leaf" transform="rotate(180 {fmt(cx)} {fmt(cy)})" fill="url(#darkLeaf)"/>
    <g clip-path="url(#coralLeafClip)">
      <g clip-path="url(#darkLeafClip)">
        <use href="#leaf" transform="rotate(60 {fmt(cx)} {fmt(cy)})" fill="url(#coralLeaf)"/>
      </g>
    </g>
    <!-- Repair the coral clip fringe before restoring the cyclic top layers. -->
    <g clip-path="url(#coralLeafClip)" mask="url(#outsideTopLeaf)">
      <g clip-path="url(#rightHalf)">
        <use href="#darkConstructionCircle" fill="none" stroke="url(#coralSeam)" stroke-width="{s(32)}"/>
      </g>
    </g>
    <g clip-path="url(#coralLeafClip)">
      <g clip-path="url(#darkLeafClip)">
        <g clip-path="url(#topLeafClip)">
          <g clip-path="url(#upperRightOverlap)">
            <use href="#leaf" transform="rotate(300 {fmt(cx)} {fmt(cy)})" fill="url(#topLeaf)"/>
          </g>
          <g clip-path="url(#upperLeftOverlap)">
            <use href="#leaf" transform="rotate(180 {fmt(cx)} {fmt(cy)})" fill="url(#darkLeaf)"/>
          </g>
        </g>
      </g>
    </g>
  </g>

  <ellipse cx="{s(558)}" cy="{s(591)}" rx="{s(54)}" ry="{s(82)}" fill="url(#eye)"/>
  <ellipse cx="{s(718)}" cy="{s(591)}" rx="{s(54)}" ry="{s(82)}" fill="url(#eye)"/>
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
        "--background",
        action="store_true",
        help="Include the warm off-white background rectangle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 128:
        raise SystemExit("--size must be at least 128")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_svg(args.size, include_background=args.background),
        encoding="utf-8",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
