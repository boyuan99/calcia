# -*- coding: utf-8 -*-
"""Fold freshly-generated FLAT series runs into the organised tree
two_color_series/volumes/seedNNNN/{gcamp,tdt,stub}. Idempotent: run after each
generation batch so the corpus stays clean. Only touches top-level deep
thin-vessel series runs; never archives or deletes."""
import glob, json, os, re, shutil

OUT = os.path.join(os.path.dirname(__file__), "output")
VOLS = os.path.join(OUT, "two_color_series", "volumes")


def meta(d):
    try:
        return json.load(open(os.path.join(d, "metadata.json")))
    except Exception:
        return {}


def canonical(pat, stub, seed):
    """Latest run matched to `stub`, preferring activity seed == volume seed."""
    best = None
    for d in glob.glob(os.path.join(OUT, pat)):        # top-level only
        m = meta(d)
        if m.get("matched_run") != stub or m.get("nt", 0) < 200:
            continue
        score = (1 if int(m.get("seed", -1)) == seed else 0, os.path.getmtime(d))
        if best is None or score > best[0]:
            best = (score, d)
    return best[1] if best else None


def main():
    moved = 0
    for stub in glob.glob(os.path.join(OUT, "deepthinves_s*_flat_stub")):
        m = re.search(r"deepthinves_s(\d+)_", os.path.basename(stub))
        if not m:
            continue
        seed = int(m.group(1))
        vdir = os.path.join(VOLS, f"seed{seed:04d}")
        stubname = os.path.basename(stub)
        pairs = [("stub", stub),
                 ("gcamp", canonical("gcamp_realistic_*", stubname, seed)),
                 ("tdt", canonical("striatum_tdt_static_*", stubname, seed))]
        for sub, src in pairs:
            if not src or not os.path.exists(src):
                continue
            dst = os.path.join(vdir, sub)
            if os.path.exists(dst):
                continue                                # already folded
            os.makedirs(vdir, exist_ok=True)
            shutil.move(src, dst)
            moved += 1
            print(f"  folded seed{seed:04d}/{sub}  <- {os.path.basename(src)}")
    print(f"folded {moved} dirs into {os.path.relpath(VOLS, OUT)}")


if __name__ == "__main__":
    main()
