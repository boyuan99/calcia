"""
Export the dendrite growth so the BROWSER can render it, instead of a video.

WHY THIS AND NOT A MOVIE
    A pre-rendered film is a recording; the page just plays it back. Shipping
    the geometry instead is both purer and cheaper:

        22 s of 1600x900 H.264   ~9.4 MB, decoded every frame, not interactive
        the geometry it came from ~0.4 MB, uploaded to the GPU once, live

    Once the segments are in a vertex buffer, the entire growth animation is a
    single uniform: `uGrow`, the fractional growth iteration. Nothing is rebuilt
    per frame -- the vertex shader decides which segments exist yet, extends the
    one that is currently growing, and fades the rest by age. Per frame the
    browser writes one float and issues one draw call.

    That is why "no 3D in the browser" was the wrong constraint to apply here.
    Four neurons is ~9k line segments. What must be avoided is the DENSE case --
    a whole volume, thousands of somata, vessel meshes, marching cubes.

WHAT IS EXPORTED
    Exactly the scene web/render_growth_film.py grows (it imports build_scene
    from it), so the still poster and the live hero are the same tissue:

      segments   parent->child dendrite edges, with Rall radius at both ends
                 and the growth iteration the child was born on
      attractors the growth field, with the iteration each was consumed
      somata     centre + radius (the GP surface is a 15 um blob on screen;
                 a shader sphere is indistinguishable and 4000x smaller)
      wisps      decimated background axon polylines

    Coordinates are centred and scaled so the scene fits a unit sphere; the
    micron scale is kept in the JSON for anything that needs to state real size.

Run:
    conda run -n calcia python web/export_growth_web.py
    conda run -n calcia python web/export_growth_web.py --seed 3 --n-neurons 5
"""

import argparse
import json
import os

import numpy as np

import render_growth_film as film

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "site", "assets", "growth")

# Same palette the film uses, as 0-255 so the JSON stays readable.
PALETTE = [[int(round(c * 255)) for c in rgb] for rgb in film.PALETTE]


def pack(arrays):
    """Concatenate float32 arrays into one blob, returning offsets in FLOATS."""
    out, offsets, cursor = [], {}, 0
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        offsets[name] = {"offset": cursor, "length": int(a.size)}
        out.append(a)
        cursor += int(a.size)
    blob = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    return blob, offsets


def build_segments(scene):
    """Parent -> child edges, with both radii and the child's birth iteration."""
    parent = scene["node_parent"]
    child = np.flatnonzero(parent >= 0)
    p0 = scene["node_pos"][parent[child]]
    p1 = scene["node_pos"][child]
    r0 = scene["node_radius"][parent[child]]
    r1 = scene["node_radius"][child]
    birth = scene["node_iter"][child].astype(np.float32)
    nid = (scene["node_nid"][child] - 1).astype(np.float32)
    return p0, p1, r0, r1, birth, nid


def build_wisps(scene, stride):
    """Background axons as loose segments, decimated -- they are atmosphere."""
    p0, p1 = [], []
    for walk in scene["axons"]:
        w = walk[::stride]
        if len(w) < 2:
            continue
        p0.append(w[:-1])
        p1.append(w[1:])
    if not p0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(p0), np.vstack(p1)


def soma_spheres(scene):
    """Centre + radius per soma, from the real mesh's own extent."""
    centres, radii = [], []
    for verts, _faces in scene["somata"]:
        c = verts.mean(axis=0)
        centres.append(c)
        radii.append(float(np.percentile(np.linalg.norm(verts - c, axis=1), 60)))
    return np.array(centres), np.array(radii)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export dendrite growth geometry for the in-browser hero.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    p.add_argument("--wisp-stride", dest="wisp_stride", type=int, default=4,
                   help="Keep every Nth point of each background axon.")
    p.add_argument("--duration", type=float, default=22.0,
                   help="Loop length in seconds the page should play it at.")
    film.add_scene_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print(f"Growth geometry -> {out_dir}")
    print("=" * 68)

    scene = film.build_scene(args)

    p0, p1, r0, r1, birth, nid = build_segments(scene)
    w0, w1 = build_wisps(scene, args.wisp_stride)
    soma_c, soma_r = soma_spheres(scene)
    attr = scene["attractors"]

    # Attractor consumption, resolved once here rather than per frame in the
    # browser: an attractor dies at the earliest iteration any node lands within
    # the kill radius. The film re-derives this with a KD-tree every frame; the
    # page cannot afford that, and it does not have to -- the answer is static.
    from scipy.spatial import cKDTree
    order = np.argsort(scene["node_iter"], kind="stable")
    tree = cKDTree(scene["node_pos"][order])
    hits = tree.query_ball_point(attr, r=scene["kill_um"])
    kill = np.full(len(attr), -1.0, dtype=np.float32)
    node_iter_sorted = scene["node_iter"][order].astype(np.float32)
    for i, idx in enumerate(hits):
        if idx:
            kill[i] = node_iter_sorted[np.min(idx)]
    print(f"[export] {int((kill >= 0).sum())}/{len(attr)} attractors consumed")

    # Centre and normalise to a unit sphere so the renderer needs no scale
    # constants; the micron figures stay in the JSON for anything that quotes
    # real sizes.
    centre = scene["centre"]
    radius = scene["radius"]

    def norm(a):
        return (np.asarray(a, dtype=np.float64) - centre) / radius

    blob, offsets = pack({
        "seg_p0": norm(p0), "seg_p1": norm(p1),
        "seg_r0": r0 / radius, "seg_r1": r1 / radius,
        "seg_birth": birth, "seg_nid": nid,
        "attr_pos": norm(attr), "attr_kill": kill,
        "soma_pos": norm(soma_c), "soma_r": soma_r / radius,
        "wisp_p0": norm(w0), "wisp_p1": norm(w1),
    })
    bin_path = os.path.join(out_dir, "growth.bin")
    blob.tofile(bin_path)

    meta = {
        "format": "float32-le",
        "bin": "growth.bin",
        "arrays": offsets,
        "counts": {
            "segments": int(len(p0)),
            "attractors": int(len(attr)),
            "somata": int(len(soma_c)),
            "wisps": int(len(w0)),
        },
        "n_iters": int(scene["n_iters"]),
        "palette": PALETTE[:len(soma_c)],
        "duration_s": args.duration,
        "beats": {k: {"start": a, "end": b} for k, (a, b) in film.BEATS.items()},
        "scene": {
            "n_neurons": scene["n_neurons"],
            "n_dendrite_nodes": int(len(scene["node_pos"])),
            "n_growth_iterations": int(scene["n_iters"]),
            "n_attractors": int(len(attr)),
            "volume_um": list(scene["vol_sz"]),
            "radius_um": round(float(radius), 2),
            "seed": args.seed,
            "strategy": "space_colonization",
        },
        "poster": ("growth_poster.jpg"
                   if os.path.exists(os.path.join(out_dir, "growth_poster.jpg"))
                   else None),
    }
    with open(os.path.join(out_dir, "growth.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    kb = os.path.getsize(bin_path) / 1024.0
    print(f"[export] {len(p0)} segments, {len(w0)} wisp segments, "
          f"{len(attr)} attractors, {len(soma_c)} somata")
    print(f"  growth.bin   {kb:8.1f} KB   ({blob.size} float32)")
    print(f"  growth.json  {os.path.getsize(os.path.join(out_dir, 'growth.json')) / 1024.0:8.1f} KB")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
