"""
generate_curved_pipe_mesh.py
-----------------------------
Generates a hollow pipe STL mesh (binary format) for Gazebo.

Supports three shapes, controlled by --shape:

  arc     (default)
      Toroidal section (elbow) curving in the horizontal XY plane.
      Controlled by --pipe_length and --bend_angle_deg.

  l_bend
      Two straight legs joined by a 90° corner arc:
          Leg 1:  straight along +X for --leg1 metres
          Corner: 90° CCW arc of radius --corner_radius in the XY plane
          Leg 2:  straight along +Y for --leg2 metres
      Controlled by --leg1, --leg2, --corner_radius.

The mesh is output in WORLD coordinates so the Gazebo model needs an
identity pose (<pose>0 0 0  0 0 0</pose>).

Four surfaces per shape:
  1. Inner cylinder   — inward normals  (what the camera sees)
  2. Outer cylinder   — outward normals
  3. Entry end cap    — at s=0
  4. Exit  end cap    — at s=pipe_length

Usage
-----
  python3 generate_curved_pipe_mesh.py
  python3 generate_curved_pipe_mesh.py --shape arc --bend_angle_deg 90 --pipe_length 4.5
  python3 generate_curved_pipe_mesh.py --shape l_bend --leg1 2.0 --leg2 2.0 --corner_radius 0.30
  python3 generate_curved_pipe_mesh.py --output /path/to/out.stl

Dependencies: numpy only.
"""

import argparse
import math
import os
import struct

import numpy as np


# ------------------------------------------------------------------ #
# STL helpers
# ------------------------------------------------------------------ #

def _unit_normal(v0, v1, v2):
    v0, v1, v2 = np.asarray(v0, float), np.asarray(v1, float), np.asarray(v2, float)
    n = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n)
    return (n / mag).tolist() if mag > 1e-12 else [0.0, 0.0, 1.0]


def _add_tri(tris, v0, v1, v2):
    tris.append((_unit_normal(v0, v1, v2), v0, v1, v2))


def _add_quad(tris, v00, v01, v10, v11):
    """
    Quad with CCW winding from the normal side:
      v00--v10
       |  / |
      v01--v11
    """
    _add_tri(tris, v00, v10, v01)
    _add_tri(tris, v10, v11, v01)


def _write_binary_stl(tris, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = b"Curved pipe mesh - generate_curved_pipe_mesh.py"
    header = header[:80].ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(tris)))
        for normal, v0, v1, v2 in tris:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


# ------------------------------------------------------------------ #
# Cross-section helpers — arc/toroid
# ------------------------------------------------------------------ #

def _cross_section_ring(phi, r, n_ring, R_bend):
    """
    Return n_ring (x,y,z) points on the pipe cross-section circle
    at arc position corresponding to subtended angle phi (arc shape).
    """
    cx = R_bend * math.sin(phi)
    cy = R_bend * (1.0 - math.cos(phi))

    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    # e_h = T × e_v  where T=(cos_phi, sin_phi, 0), e_v=(0,0,1)
    # e_h = (sin_phi, -cos_phi, 0)
    eh = (sin_phi, -cos_phi, 0.0)
    ev = (0.0, 0.0, 1.0)

    pts = []
    for k in range(n_ring):
        theta = 2.0 * math.pi * k / n_ring
        ct, st = math.cos(theta), math.sin(theta)
        x = cx + r * (ct * ev[0] + st * eh[0])
        y = cy + r * (ct * ev[1] + st * eh[1])
        z =      r * (ct * ev[2] + st * eh[2])
        pts.append((x, y, z))
    return pts


def _straight_ring(x_pos, r, n_ring):
    """Cross-section ring at x = x_pos on a straight pipe along X."""
    pts = []
    for k in range(n_ring):
        theta = 2.0 * math.pi * k / n_ring
        pts.append((x_pos, r * math.sin(theta), r * math.cos(theta)))
    return pts


# ------------------------------------------------------------------ #
# Cross-section helpers — L-bend
# ------------------------------------------------------------------ #

def _l_bend_frenet(s, leg1, corner_radius):
    """
    Return (cx, cy, tx, ty) — centreline position and heading tangent —
    at arc-length s along an L-bend path.

    Path:
      Leg 1:  s ∈ [0, leg1]                      → along +X
      Corner: s ∈ [leg1, leg1 + corner_radius*π/2] → 90° CCW arc in XY
      Leg 2:  s ∈ [arc_end, arc_end + leg2]       → along +Y
    """
    corner_arc_len = corner_radius * math.pi / 2.0
    arc_end_s      = leg1 + corner_arc_len

    if s <= leg1:
        return s, 0.0, 1.0, 0.0

    if s <= arc_end_s:
        phi  = (s - leg1) / corner_radius
        cx   = leg1 + corner_radius * math.sin(phi)
        cy   = corner_radius * (1.0 - math.cos(phi))
        return cx, cy, math.cos(phi), math.sin(phi)

    d = s - arc_end_s
    return leg1 + corner_radius, corner_radius + d, 0.0, 1.0


def _l_bend_ring(s, r, n_ring, leg1, corner_radius):
    """
    Return n_ring (x,y,z) points on the pipe cross-section at arc position s
    along an L-bend path.

    Frenet frame:
      e_v = (0, 0, 1)              — world vertical
      e_h = T × e_v = (ty, -tx, 0) — right perpendicular to heading
      P(θ) = C + r·(cos(θ)·e_v + sin(θ)·e_h)
    """
    cx, cy, tx, ty = _l_bend_frenet(s, leg1, corner_radius)
    eh_x = ty
    eh_y = -tx

    pts = []
    for k in range(n_ring):
        theta = 2.0 * math.pi * k / n_ring
        ct, st = math.cos(theta), math.sin(theta)
        x = cx + r * (ct * 0.0 + st * eh_x)
        y = cy + r * (ct * 0.0 + st * eh_y)
        z =      r * (ct * 1.0 + st * 0.0)
        pts.append((x, y, z))
    return pts


# ------------------------------------------------------------------ #
# Mesh builders
# ------------------------------------------------------------------ #

def _build_hollow_pipe_from_rings(inner_rings, outer_rings, n_ring, n_arc):
    """
    Shared lateral + end-cap logic for any ring sequence.
    inner_rings / outer_rings: lists of n_arc+1 ring point lists.
    """
    tris = []

    # ---- Lateral surfaces ---------------------------------------- #
    for seg in range(n_arc):
        ri0, ri1 = inner_rings[seg], inner_rings[seg + 1]
        ro0, ro1 = outer_rings[seg], outer_rings[seg + 1]

        for k in range(n_ring):
            kn = (k + 1) % n_ring

            # Inner surface — inward normals (CW from outside = CCW from inside)
            _add_quad(tris, ri1[k], ri0[k], ri1[kn], ri0[kn])

            # Outer surface — outward normals (CCW from outside)
            _add_quad(tris, ro0[k], ro1[k], ro0[kn], ro1[kn])

    # ---- Entry end cap ------------------------------------------- #
    ri_e, ro_e = inner_rings[0], outer_rings[0]
    for k in range(n_ring):
        kn = (k + 1) % n_ring
        _add_tri(tris, ro_e[k],  ri_e[k],  ro_e[kn])
        _add_tri(tris, ri_e[k],  ri_e[kn], ro_e[kn])

    # ---- Exit end cap -------------------------------------------- #
    ri_x, ro_x = inner_rings[-1], outer_rings[-1]
    for k in range(n_ring):
        kn = (k + 1) % n_ring
        _add_tri(tris, ro_x[kn], ri_x[kn], ro_x[k])
        _add_tri(tris, ri_x[kn], ri_x[k],  ro_x[k])

    return tris


def build_curved_hollow_pipe(inner_r, outer_r, pipe_length, bend_angle_deg,
                              n_arc, n_ring):
    """
    Hollow curved pipe — circular arc (toroid) in the horizontal XY plane.
    If bend_angle_deg == 0 builds a straight pipe along world X.
    """
    bend_angle_rad = math.radians(bend_angle_deg)
    R_bend         = pipe_length / bend_angle_rad if bend_angle_deg > 0.0 else 0.0
    phi_values     = [bend_angle_rad * i / n_arc for i in range(n_arc + 1)]
    s_values       = [pipe_length    * i / n_arc for i in range(n_arc + 1)]

    inner_rings, outer_rings = [], []
    for phi, s in zip(phi_values, s_values):
        if bend_angle_deg > 0.0:
            inner_rings.append(_cross_section_ring(phi, inner_r, n_ring, R_bend))
            outer_rings.append(_cross_section_ring(phi, outer_r, n_ring, R_bend))
        else:
            inner_rings.append(_straight_ring(s, inner_r, n_ring))
            outer_rings.append(_straight_ring(s, outer_r, n_ring))

    return _build_hollow_pipe_from_rings(inner_rings, outer_rings, n_ring, n_arc)


def build_l_bend_hollow_pipe(inner_r, outer_r, leg1, leg2, corner_radius,
                              n_arc, n_ring):
    """
    Hollow L-bend pipe:
        Leg 1: leg1 metres straight along +X
        Corner: 90° CCW arc of radius corner_radius in the XY plane
        Leg 2: leg2 metres straight along +Y

    Total arc length = leg1 + corner_radius*pi/2 + leg2
    """
    total_len = leg1 + corner_radius * math.pi / 2.0 + leg2
    s_values  = [total_len * i / n_arc for i in range(n_arc + 1)]

    inner_rings, outer_rings = [], []
    for s in s_values:
        inner_rings.append(_l_bend_ring(s, inner_r, n_ring, leg1, corner_radius))
        outer_rings.append(_l_bend_ring(s, outer_r, n_ring, leg1, corner_radius))

    return _build_hollow_pipe_from_rings(inner_rings, outer_rings, n_ring, n_arc)


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

def _parse_args():
    p = argparse.ArgumentParser(description="Generate hollow pipe STL.")
    p.add_argument("--shape",          type=str,   default="arc",
                   choices=["arc", "l_bend"],
                   help="Pipe shape: 'arc' (toroid) or 'l_bend' (90° L-corner)")
    # Shared
    p.add_argument("--inner_radius",   type=float, default=0.15)
    p.add_argument("--wall_thickness", type=float, default=0.03)
    p.add_argument("--n_arc_segs",     type=int,   default=64,
                   help="Number of arc segments (resolution along pipe)")
    p.add_argument("--n_ring_segs",    type=int,   default=32,
                   help="Number of ring segments (cross-section resolution)")
    p.add_argument("--output", type=str,
                   default=os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "..", "meshes", "pipe_hollow.stl",
                   ))
    # Arc-specific
    p.add_argument("--pipe_length",    type=float, default=4.5,
                   help="Total arc length in metres (arc shape only)")
    p.add_argument("--bend_angle_deg", type=float, default=90.0,
                   help="Total turn in degrees (arc shape only)")
    # L-bend-specific
    p.add_argument("--leg1",           type=float, default=2.0,
                   help="First straight section length [m] (l_bend only)")
    p.add_argument("--leg2",           type=float, default=2.0,
                   help="Second straight section length [m] (l_bend only)")
    p.add_argument("--corner_radius",  type=float, default=0.30,
                   help="Corner arc radius [m] (l_bend only)")
    return p.parse_args()


def main():
    args    = _parse_args()
    inner_r = args.inner_radius
    outer_r = inner_r + args.wall_thickness

    print("Generating pipe mesh…")
    print(f"  Shape          : {args.shape}")
    print(f"  Inner radius   : {inner_r} m")
    print(f"  Outer radius   : {outer_r} m")

    if args.shape == "arc":
        total_len = args.pipe_length
        print(f"  Pipe length    : {total_len} m")
        print(f"  Bend angle     : {args.bend_angle_deg}°")
        if args.bend_angle_deg > 0.0:
            R = args.pipe_length / math.radians(args.bend_angle_deg)
            print(f"  Bend radius    : {R:.3f} m")
        else:
            print(f"  Pipe shape     : straight (along world X)")
        tris = build_curved_hollow_pipe(
            inner_r, outer_r,
            args.pipe_length, args.bend_angle_deg,
            args.n_arc_segs, args.n_ring_segs,
        )
    else:  # l_bend
        corner_arc = args.corner_radius * math.pi / 2.0
        total_len  = args.leg1 + corner_arc + args.leg2
        print(f"  Leg 1          : {args.leg1} m  (along +X)")
        print(f"  Corner arc     : {corner_arc:.3f} m  (R={args.corner_radius} m, 90° CCW)")
        print(f"  Leg 2          : {args.leg2} m  (along +Y)")
        print(f"  Total length   : {total_len:.3f} m")
        tris = build_l_bend_hollow_pipe(
            inner_r, outer_r,
            args.leg1, args.leg2, args.corner_radius,
            args.n_arc_segs, args.n_ring_segs,
        )

    print(f"  Arc segments   : {args.n_arc_segs}")
    print(f"  Ring segments  : {args.n_ring_segs}")

    out = os.path.abspath(args.output)
    _write_binary_stl(tris, out)

    size_kb = os.path.getsize(out) / 1024
    print(f"Done: {len(tris)} triangles → {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
