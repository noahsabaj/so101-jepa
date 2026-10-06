"""Prepare the parts of the insertion test: the SO-101 base (housing) and an STS3215 servo.

All three simulators use the same output (bakeoff/parts/):
- base_*.obj: convex pieces of the base (CoACD), in the base frame of the official MJCF.
- servo_*.obj: convex pieces of the servo (no horn), in the servo frame.
- parts.json: the seated servo pose in the base frame, the free insertion direction found by a
  sweep test, and the CAD clearance around the servo body.

Run: uv run python bakeoff/geometry.py
"""

import json
import os
from pathlib import Path

import coacd
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ASSETS = Path("assets/so101/assets")
OUT = Path("bakeoff/parts")

# Poses from assets/so101/so101_new_calib_camera.xml, body "base" (MuJoCo quat order: w x y z).
BASE_MESH_POSE = ([-0.00636471, 0.0, -0.0024], [0.5, 0.5, 0.5, 0.5])
SERVO_POSE = ([0.0263353, 0.0, 0.0437], [1.0, 0.0, 0.0, 0.0])
# CoACD concavity threshold for the base: 0.03 gave 66 pieces that reach up to 0.59 mm into the
# pocket, more than the clearances under test. Set from the comparison in parts.json.
BASE_THRESHOLD = float(os.environ.get("BASE_THRESHOLD", "0.01"))


def pose_matrix(pos, quat_wxyz):
    m = np.eye(4)
    w, x, y, z = quat_wxyz
    m[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    m[:3, 3] = pos
    return m


def decompose(mesh, threshold):
    """Convex pieces of a watertight mesh with CoACD (threshold: concavity in [0.01, 1]). Each piece
    has at most 64 vertices: PhysX simplifies hulls above 255 vertices and can make them larger."""
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                            decimate=True, max_ch_vertex=64)
    return [trimesh.Trimesh(v, f).convex_hull for v, f in parts]


def surface_points(mesh, n, inset, seed):
    """Surface points moved `inset` m inward: CAD faces that touch at zero clearance (or 0.05 mm
    of tessellation overlap) do not count as collisions."""
    points, face_idx = mesh.sample(n, return_index=True, seed=seed)
    return points - mesh.face_normals[face_idx] * inset


def free_directions(base, servo, travel=0.06, samples=4000, inset=1.5e-4):
    """Directions (+-x, +-y, +-z of the base frame) along which the seated servo leaves the base
    with no (inset) surface point inside it."""
    points = surface_points(servo, samples, inset, seed=0)
    free = []
    for axis, name in enumerate("xyz"):
        for sign in (1, -1):
            hits = 0
            for d in np.linspace(0.001, travel, 30):
                shift = np.zeros(3)
                shift[axis] = sign * d
                hits += int(base.contains(points + shift).sum())
            if hits == 0:
                free.append(("+" if sign > 0 else "-") + name)
    return free


def side_gaps(base, servo, max_gap=0.01):
    """Gap (mm) from each servo face direction (+-x, +-y, +-z) to the base: the 5th percentile of
    ray distances from servo surface points along their outward normals (rays that find no base
    within max_gap are ignored). 0 means the faces touch."""
    points, face_idx = servo.sample(40000, return_index=True, seed=1)
    normals = servo.face_normals[face_idx]
    gaps = {}
    for axis, name in enumerate("xyz"):
        for sign in (1, -1):
            sel = normals[:, axis] * sign > 0.95
            origins = points[sel] - normals[sel] * 2e-4  # start just inside the servo
            loc, ray_idx, _ = base.ray.intersects_location(origins, normals[sel], multiple_hits=False)
            d = np.linalg.norm(loc - origins[ray_idx], axis=1) - 2e-4
            d = d[d < max_gap]
            key = ("+" if sign > 0 else "-") + name
            gaps[key] = round(float(np.percentile(d, 5)) * 1000, 3) if len(d) else None
    return gaps


def intrusion_mm(pieces, servo_seated, n=30000):
    """How far the convex pieces reach into the seated servo's space (max and 99th percentile,
    mm). The exact CAD mesh gives 0.05 mm (zero-clearance faces); more is decomposition error."""
    points = surface_points(servo_seated, n, 0.0, seed=4)
    depth = np.zeros(len(points))
    for piece in pieces:
        inside = piece.contains(points)
        if inside.any():
            _, d, _ = trimesh.proximity.closest_point(piece, points[inside])
            depth[inside] = np.maximum(depth[inside], d)
    hit = depth[depth > 0]
    return round(float(depth.max()) * 1000, 3), round(float(np.percentile(hit, 99)) * 1000, 3) if len(hit) else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.obj"):
        old.unlink()
    base = trimesh.load(ASSETS / "base_so101_v2.stl")
    base.apply_transform(pose_matrix(*BASE_MESH_POSE))
    servo = trimesh.load(ASSETS / "sts3215_03a_no_horn_v1.stl")
    seated = pose_matrix(*SERVO_POSE)
    servo_in_base = servo.copy().apply_transform(seated)

    overlap = int(base.contains(surface_points(servo_in_base, 4000, 1.5e-4, seed=2)).sum())
    free = free_directions(base, servo_in_base)
    gaps = side_gaps(base, servo_in_base)

    # Only the pocket matters for this test: crop the base to the servo's box plus 8 mm.
    lo, hi = servo_in_base.bounds[0] - 0.008, servo_in_base.bounds[1] + 0.008
    box = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
    base_crop = base.intersection(box)
    base_crop.export(OUT / "base_crop.obj")
    servo.export(OUT / "servo.obj")

    base_parts = decompose(base_crop, threshold=BASE_THRESHOLD)
    servo_parts = decompose(servo, threshold=0.05)
    for i, p in enumerate(base_parts):
        p.export(OUT / f"base_{i:02d}.obj")
    for i, p in enumerate(servo_parts):
        p.export(OUT / f"servo_{i:02d}.obj")

    info = {
        "seated_servo_pos": SERVO_POSE[0],
        "seated_servo_quat_wxyz": SERVO_POSE[1],
        "free_directions": free,
        "seated_overlap_points": overlap,
        "side_gaps_mm": gaps,
        "base_pieces": len(base_parts),
        "base_coacd_threshold": BASE_THRESHOLD,
        "pocket_intrusion_mm_max_p99": intrusion_mm(base_parts, servo_in_base),
        "servo_pieces": len(servo_parts),
        "servo_mass_kg": 0.055,
    }
    (OUT / "parts.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
