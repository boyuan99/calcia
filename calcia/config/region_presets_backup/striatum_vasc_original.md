# Striatum vasculature presets — thinned (active) vs de-thinned (variant)

The THINNED preset is currently ACTIVE in `_STRIATUM_VASC` (fast, tractable
generation). A DE-THINNED variant renders the dark vessels the real tdTomato data
shows and is verified at small FOV, but makes 1.7 mm-FOV Phase-1 generation
IMPRACTICAL (>6.5 h, vessel Dijkstra does not scale). Use the de-thinned values
only for small-FOV vessel'd runs, until the pathfinding is sped up. Both value
sets are recorded here so either can be pasted into `_STRIATUM_VASC`.

## THINNED `_STRIATUM_VASC` (ACTIVE — vessels faint, generation fast)
```python
_STRIATUM_VASC = {
    "depth_surf": 0.0,
    # Thinner, sparser vessels: in 1P widefield the strong out-of-focus haze
    # fills THIN vessel voids so real striatum vessels are nearly invisible
    # (measured: real mean-image has ~0 strong dark structures, sim had median
    # ~15 um up to 65 um dark voids). radius 9->4 um (diameter ~8 um) + sparser
    # so the haze fills them and they fade like the real samples.
    "vesSize": (2.0, 2.0, 1.0),
    "vesFreq": (600.0, 600.0, 150.0),
    "distsc": 6.0,
}
```

## DE-THINNED variant (renders dark vessels; small-FOV only)
```python
    "vesSize": (10.0, 6.0, 2.0),
    "vesFreq": (200.0, 250.0, 60.0),
    "distsc": 4.0,
```
The thinning premise ("real striatum vessels are nearly invisible") is
CONTRADICTED by the real tdTomato recordings (`data/real/tdt-bfp/*_tdt_mc.h5`),
which show clear dark blood vessels + voids as the dominant texture. The
de-thinned values restore vessel size/density toward the cortex baseline
(`VascParams` defaults `vesSize=(15,9,2)`, `vesFreq=(125,200,50)`, `distsc=4`)
while keeping striatum topology (`depth_surf=0.0`, isotropic penetrators), so the
scan produces dark vasculature via hemoglobin absorption — physically, not as an
overlay. Verified on a 500 um volume (1.06% vessel voxels, dark vessels visible).
**Caveat:** at 1.7 mm FOV the denser vessel Dijkstra ran >6.5 h without finishing
— speeding up the vessel pathfinding is the prerequisite for full-FOV vessels.

Cortex baseline for reference (`VascParams` dataclass defaults):
`vesSize=(15.0, 9.0, 2.0)`, `vesFreq=(125.0, 200.0, 50.0)`, `distsc=4.0`.
