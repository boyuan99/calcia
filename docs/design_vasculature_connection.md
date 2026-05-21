# Vasculature Design Note: Why "Grow + Dijkstra-Connect"?

This note documents a design question raised about the blood-vessel generation
pipeline in [calcia/volume/vasculature.py](../calcia/volume/vasculature.py),
and the rationale for keeping the current NAOMi-style algorithm versus
potential alternatives.

---

## The Question

> Vessels are already produced by a growth process (random walks from source
> nodes at the volume boundary). Why then do we need a separate Dijkstra pass
> to "connect" the nodes? Isn't this redundant and unnecessarily complex?

See `grow_major_vessels` in
[vasculature.py:494-586](../calcia/volume/vasculature.py#L494-L586)
and `connect_vessel_nodes` in
[vasculature.py:589-647](../calcia/volume/vasculature.py#L589-L647).

---

## Why the Current Two-Stage Approach Is Defensible

Real cerebral vasculature is **not a tree** — it contains a large number of
anastomoses (loop/mesh connections). The current pipeline models this by
separating concerns:

| Stage | Produces | What it models |
|-------|----------|----------------|
| Growth (random walk from sources) | Node **geometry** | Tortuosity / curvature of individual vessel branches |
| Dijkstra (weighted shortest path between node pairs) | Node **topology** | Anastomotic network structure (which branches fuse) |

If we did growth only, we would get a disconnected forest — one tree per
source node — with no cross-branch fusion. That does not match the anatomical
reality of the cortical vessel network.

The Dijkstra pass specifically:
1. Builds an `N × N` distance matrix between all nodes
2. Perturbs it with `randWeightScale * rand * dist` — so paths are not
   pure geodesics but have biologically plausible jitter
3. Masks entries beyond `3 × lensc` to `inf` — only local connections are
   allowed
4. Runs Dijkstra from every source node and links each node to the closest
   source tree via `parents` back-tracking

So the "grow" and "connect" stages are doing genuinely different modelling
jobs — the second stage is not cleanup, it is where the network topology is
actually generated.

---

## Alternatives Worth Considering (for a Future v2)

If the project ever drops the MATLAB-parity constraint, three alternative
designs would be worth evaluating:

### 1. Space Colonization (Runions 2007)
Seed attractor points inside the volume; growing branch tips advance toward
the nearest attractor and merge when they approach another branch. Produces
form **and** topology in a single pass, with natural anastomoses. This is the
approach used by SimVascular and much of the modern botanical / vascular
modelling literature.

- **Pro**: One-pass, biologically principled, loops emerge naturally
- **Con**: Different parameter space from NAOMi — requires retuning + revalidation

### 2. In-Growth Connection (Local Merge)
Keep the current growth loop, but at each step check whether the advancing
tip is within `mindist` of any existing node. If so, fuse directly instead
of continuing. Eliminates the post-hoc `N²` Dijkstra entirely.

- **Pro**: Simple change, reduces complexity from O(N² log N) to O(N log N)
- **Con**: Topology is path-dependent (order of growth matters), harder to
  control aggregate connection statistics

### 3. Delaunay / Voronoi Pruning
Scatter seed points, run Delaunay triangulation, then prune edges by a
distance + random weight criterion to yield the vessel graph.

- **Pro**: Very fast, fully geometric
- **Con**: Less biological realism in branch curvature — you would still need
  a separate step to add tortuosity

---

## Recommendation: Keep the Current Algorithm

**Do not change the vasculature algorithm in calcia v1.**

Two reasons:

1. **MATLAB parity is a project invariant.** The memory file records the rule
   "Always compare with MATLAB after running calcia". Replacing the growth +
   Dijkstra algorithm would invalidate every regression comparison, including
   downstream effects on neuron placement (which avoids vessels) and PSF
   occlusion statistics in Phase 2.

2. **NAOMi is a published benchmark.** Diverging from its vessel statistics
   would mean calcia is no longer a faithful port, which undermines the
   point of the port.

If a calcia v2 is ever undertaken without the MATLAB-parity constraint,
**Space Colonization is the recommended replacement** — it is the cleanest
single-pass formulation and produces anatomically realistic networks directly.

---

## Summary

- The two-stage approach is not redundant; the stages model different things
  (geometry vs. topology).
- "Space colonization" is the canonical modern alternative and would be the
  right choice for a clean-slate rewrite.
- For now, the MATLAB-parity constraint pins us to the current design — and
  that is the correct engineering call for a reproducibility-focused port.
