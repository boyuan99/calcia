# web/ — the public showcase

A static page for putting calcia on a website. Pure frontend: HTML, CSS, and
JavaScript with **no framework, no library, no CDN, and no server**. The 3D is
really rendered in the browser — it is not a video of a render.

The constraint that shaped it is that a visitor's laptop must not be made to
work. That does not mean "don't render 3D"; it means **don't make the CPU
animate anything, and never put a dense scene on screen.**

```
web/
  export_growth_web.py    growth geometry  ->  growth.bin + growth.json  (what the site uses)
  bundle_data.py          inlines every payload -> assets/data.js  (so file:// works)
  render_growth_film.py   the same scene as an offline film + poster     (optional)
  build_site.py           finished runs    ->  web video + tiny JSON payloads
  site/
    index.html            the page (bilingual EN / 中文, no build step)
    styles.css
    growth3d.js           ~330 lines of WebGL 1 — the whole 3D renderer
    app.js
    assets/               GENERATED — git-ignored, rebuilt by the scripts above
```

## Build

```bash
# 1. the growth geometry the browser renders   (~15 s, 333 KB)
conda run -n calcia python web/export_growth_web.py

# 2. imaging footage, ground truth, traces, PSF, diversity corpus   (~30 s)
conda run -n calcia python web/build_site.py

# 3. inline those payloads so the page needs no server at all   (~1 s)
conda run -n calcia python web/bundle_data.py
```

Then just open `web/site/index.html`. Double-clicking it works — there is no
server, no build, no watcher, nothing running.

**Python is a build tool here, not a runtime.** It generates the assets once and
is never involved again; the deployed site is static files that your existing
web host serves. Skip step 3 if you prefer the browser to cache each payload
separately — then the page fetches them, which needs the folder served over
HTTP (`python -m http.server`, `npx serve`, nginx, GitHub Pages, anything).

Optional, and not needed by the page:

```bash
# the offline film + the poster the page falls back to without WebGL  (~5 min)
conda run -n calcia python web/render_growth_film.py            # --smoke for a 1-min check
```

`?t=0.42` on the page URL freezes the growth at that point in the loop — useful
for grabbing a still, and it makes a moment in the growth linkable.

## Why the 3D is cheap

Growth is **not animated on the CPU**. Every dendrite segment is uploaded once
with the growth iteration it was born on baked into its vertices. Per frame the
vertex shader decides which segments exist yet, extends the single one that is
currently growing, and fades the rest by age.

So the per-frame work is: write one float (`uGrow`), issue four draw calls.
Nothing is rebuilt, no array is walked, no geometry is touched.

|  | live geometry | the equivalent film |
|---|---|---|
| download | **333 KB** | 9.4 MB |
| per frame | 1 uniform + 4 draw calls | decode 1600×900, every frame |
| interactive | orbit, scrub, pause anywhere | no |

Blending is additive with the depth test off. That is both the right look —
fluorescence really does add — and the reason no depth sorting is ever needed.
Depth is carried by fog instead of by occlusion.

Guard rails in `growth3d.js`: device pixel ratio capped at 1.5, total backing
store capped at ~2.2 Mpx however large the element gets, the render loop stops
entirely when the hero scrolls out of view or the tab is hidden, and
`prefers-reduced-motion` holds a still instead of looping. No WebGL at all falls
back to the rendered poster.

## What is on screen is the real algorithm

`space_colonization` (the pipeline's default basal-dendrite strategy) scatters
attractor points through the tissue and grows every neuron's tree toward them
simultaneously from one shared pool: an attractor is consumed by whichever tip
reaches it first, so the trees partition space by competition with no explicit
avoidance. The algorithm builds an explicit node/parent forest and then
rasterizes it into voxels, discarding the forest.

`grow_neuron_dendrites(..., capture_growth=True)` keeps it
(`calcia/volume/dendrites.py`, `DendriteGrowthGraph`). That is the only change
this showcase required in the library, it is opt-in, and `neur_num` is
bit-identical with it on or off. Node order is a valid growth order for free:
nodes are created parent-before-child, so `node_parent[i] < i` always holds.

| on screen | what it is |
|---|---|
| drifting dust | the live attractor pool |
| dust vanishing | each attractor's real consumption iteration, resolved at export |
| branch girth | Rall's law on real subtree tip counts, unrounded |
| thin tips thickening | branch radius scaled by node age |
| white specks | the growth front: nodes born this iteration |

Two things are staging rather than physics, and are labelled as such in the
code: the per-neuron **reveal** is staggered (`--stagger-iters`) even though the
forest really does grow in lockstep, and the dendritic field radius is tightened
from the 150 µm library default (`--field-um`) because at portrait scale the
trees otherwise fuse into one hairball. Somata are drawn as shader spheres; the
real GP-displaced surfaces are 15 µm blobs at this scale and are exported only
as centre + radius.

## The imaging assets

`build_site.py` re-reads `movies.npz` from finished runs — it never re-simulates
— applies the same 0.5/99.5 percentile window `calcia/io/render.py` uses, adds a
display gamma, and pipes raw frames straight into ffmpeg.

Those clips stay `<video>` elements. A video tag is frontend; the objection this
page answers is *pre-rendering 3D and calling it a 3D demo*, not the existence of
footage. Shipping 152 frames of 200×200 as pixel data instead would be ~25 MB of
texture memory per channel for no gain.

This still matters more than it sounds: every animation this project had
produced was a GIF, they average 30–80 MB for six seconds, and a GIF is decoded
frame by frame on the CPU with no hardware path. The 42 MB diversity overview
becomes 1.4 MB of H.264.

### The soma overlay is verified, not assumed

µm → movie pixel is `x_um * vres / sfrac - scan_buff / sfrac`
(`calcia/benchmark/gt.py`). Dropping the `scan_buff` term misregisters everything
by half the buffer.

The row/col assignment is genuinely ambiguous in this codebase — `viz_ladders3d`
uses `row=x`, `benchmark/matching` searches the swap — and on a dense field
temporal standard deviation cannot tell them apart (both score ~1.1× the median).
So the builder correlates each claimed pixel's time course against that neuron's
**own known trace**, which can: a transposed mapping correlates a cell with
someone else's activity and collapses to zero.

On the shipped hero run: `row=x` → median r = **+0.121**, `row=y` → **−0.003**.
That number is printed at build time and shown on the page.

## Client cost

| section | what runs in the browser |
|---|---|
| growth scene | one static vertex buffer; 1 uniform + 4 draw calls per frame |
| compare slider | a CSS clip rectangle |
| ground truth | a few hundred canvas circles, redrawn only on input |
| traces | static canvas + a one-pixel playhead |
| anything off-screen | render loop stopped; videos paused, unbuffered, undecoded |

Off-screen gating is an `IntersectionObserver` and it is the single most
important line on the page: six autoplaying videos will heat a laptop even though
each decode is individually cheap.

## Deploying

`web/site/` is a plain static directory — copy it anywhere. Nothing runs
server-side; there is no API, no database, no process. `assets/` is git-ignored
because it is build output; either run the scripts on the deploy host or copy the
folder across. Total ~19 MB, of which the growth scene is 333 KB.

Each section in `index.html` is a self-contained block with its own `init*()` in
`app.js`, so individual pieces lift into a host page cleanly. All styles are
scoped under `.calcia-showcase`.

## What is deliberately not here

No data from `data/real/` — those are recordings from live animals. Nothing on
this page is derived from them, and the builder does not read that directory.
