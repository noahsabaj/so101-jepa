"""Static check of the insertion path. The servo moves from 20 mm out to seated along the free
direction (+x of the base frame). At each depth, how far do the shapes overlap?

- pieces: servo convex pieces vs base convex pieces (what the simulators see),
- servo mesh vs base pieces (the error of the base split),
- servo pieces vs base mesh (the error of the servo split),
- servo mesh vs base mesh (the CAD itself; 0.05 mm is tessellation at zero-clearance faces).

The servo is made smaller across x by the clearance, as in the insertion tests.

Run: PARTS=bakeoff/parts uv run --with rtree python bakeoff/path_check.py [CLEARANCE_MM ...]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

PARTS = Path(os.environ.get("PARTS", "bakeoff/parts"))
DEPTHS_MM = np.arange(20.0, -0.01, -0.5)  # distance from seated
POINTS = 20000


def load(prefix):
    return [trimesh.load(f) for f in sorted(PARTS.glob(f"{prefix}_[0-9]*.obj"))]


def planes(piece):
    """Face planes (unit normal, offset) of a convex piece: inside where all n.p - d < 0."""
    return piece.face_normals, np.einsum("ij,ij->i", piece.face_normals, piece.triangles[:, 0])


def overlap_pieces(points, pieces):
    """Deepest point inside any convex piece (m): the distance to the nearest face plane."""
    depth = 0.0
    for n, d, lo, hi in pieces:
        sel = np.all((points >= lo) & (points <= hi), axis=1)
        if not sel.any():
            continue
        s = points[sel] @ n.T - d  # (points, faces)
        inside = s.max(axis=1) < 0
        if inside.any():
            depth = max(depth, float((-s[inside].max(axis=1)).max()))
    return depth


def overlap_mesh(points, mesh):
    inside = mesh.contains(points)
    if not inside.any():
        return 0.0
    _, d, _ = trimesh.proximity.closest_point(mesh, points[inside])
    return float(d.max())


def surface(meshes, n, seed):
    mesh = trimesh.util.concatenate(meshes)
    return mesh.sample(n, seed=seed)


def main():
    info = json.loads((PARTS / "parts.json").read_text())
    seated = np.array(info["seated_servo_pos"])
    servo, base = trimesh.load(PARTS / "servo.obj"), trimesh.load(PARTS / "base_crop.obj")
    base_pieces = load("base")
    servo_pieces = load("servo")
    base_planes = [(*planes(p), p.bounds[0] - 1e-6, p.bounds[1] + 1e-6) for p in base_pieces]
    ext, centre = servo.extents, servo.bounds.mean(axis=0)
    report = {"parts": str(PARTS), "base_pieces": len(base_pieces), "servo_pieces": len(servo_pieces), "clearances": {}}
    for c_mm in [float(a) for a in sys.argv[1:]] or [0.2]:
        c = c_mm / 1000
        scale = np.array([1.0, (ext[1] - 2 * c) / ext[1], (ext[2] - 2 * c) / ext[2]])
        pts_mesh = (surface([servo], POINTS, 1) - centre) * scale + centre
        pts_pieces = (surface(servo_pieces, POINTS, 2) - centre) * scale + centre
        rows = []
        for depth in DEPTHS_MM:
            shift = seated + [depth / 1000, 0.0, 0.0]
            a, b = pts_pieces + shift, pts_mesh + shift
            rows.append({
                "out_mm": round(float(depth), 1),
                "pieces": round(1000 * overlap_pieces(a, base_planes), 3),
                "servo_mesh_vs_base_pieces": round(1000 * overlap_pieces(b, base_planes), 3),
                "servo_pieces_vs_base_mesh": round(1000 * overlap_mesh(a, base), 3),
                "meshes": round(1000 * overlap_mesh(b, base), 3),
            })
            print(c_mm, rows[-1], flush=True)
        report["clearances"][str(c_mm)] = rows
    out = PARTS / "path_check.json"
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
