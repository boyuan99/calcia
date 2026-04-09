"""Compare MATLAB NAOMi Phase 1 output with calcia Python output.

Loads MATLAB statistics exported by run_phase1_for_comparison.m (debug mode)
and Python output from calcia's output.npz, then produces a comprehensive
comparison report with pass/warn/fail indicators.

Usage:
    conda run -n calcia python comparison_tools/compare_phase1.py
    conda run -n calcia python comparison_tools/compare_phase1.py --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io

# Add calcia to path if needed
CALCIA_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CALCIA_ROOT))

from calcia import import_pipeline_output

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MATLAB_STATS_PATH = Path(r"C:\Users\boyuan\Documents\GitHub\naomi_sim"
                         r"\comparison_tools\matlab_phase1_stats.mat")
PYTHON_OUTPUT_PATH = CALCIA_ROOT / "examples" / "output" / "output.npz"


# ---------------------------------------------------------------------------
# Tolerance thresholds: (relative_tol, absolute_tol)
# ---------------------------------------------------------------------------
TOLERANCES = {
    "exact":           (0.02, 5),       # near-exact (neuron count, etc.)
    "structural":      (0.10, None),    # structural metrics (soma, dendrites)
    "statistical":     (0.25, None),    # statistical metrics (fluorescence)
    "fill_fraction":   (None, 0.05),    # absolute 5% for fill fractions
    "component_count": (0.15, None),    # component counts (bg, axon)
    "spatial":         (0.10, None),    # spatial distributions
}


def _safe_int(val):
    return int(val.item()) if hasattr(val, 'item') else int(val)


def _safe_float(val):
    return float(val.item()) if hasattr(val, 'item') else float(val)


def _safe_attr(obj, name, default=None):
    """Safely get attribute, return default if missing."""
    return getattr(obj, name, default) if hasattr(obj, name) else default


def _fmt_val(val):
    """Format a value for display."""
    if val is None:
        return f"{'---':>14s}"
    if isinstance(val, float):
        if abs(val) < 0.01 or abs(val) > 1e6:
            return f"{val:>14.4e}"
        return f"{val:>14,.4f}"
    if isinstance(val, int):
        return f"{val:>14,d}"
    return f"{str(val):>14s}"


def compare_metric(label, m_val, p_val, tol_key="structural"):
    """Compare one metric. Returns (status, formatted_line)."""
    if m_val is None or p_val is None:
        return "SKIP", f"  {label:45s}  {_fmt_val(m_val)}  {_fmt_val(p_val)}  {'---':>8s}  [ --- ]"

    rel_tol, abs_tol = TOLERANCES.get(tol_key, (0.20, None))

    # Compute differences
    if m_val != 0:
        rel_diff = abs(p_val - m_val) / abs(m_val)
    elif p_val != 0:
        rel_diff = 1.0
    else:
        rel_diff = 0.0
    abs_diff = abs(p_val - m_val)

    # Determine status
    status = "PASS"
    if rel_tol is not None and rel_diff > rel_tol:
        status = "WARN" if rel_diff < rel_tol * 2 else "FAIL"
    if abs_tol is not None and abs_diff > abs_tol:
        if status != "FAIL":
            status = "WARN" if abs_diff < abs_tol * 2 else "FAIL"

    diff_str = f"{rel_diff:+.1%}" if m_val != 0 else "N/A"
    tag = {"PASS": "  OK ", "WARN": " WARN", "FAIL": " FAIL"}[status]

    return status, f"  {label:45s}  {_fmt_val(m_val)}  {_fmt_val(p_val)}  {diff_str:>8s}  [{tag}]"


def load_matlab_stats(path):
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return mat["stats"]


def extract_python_stats(output):
    """Extract comprehensive statistics from calcia NeuralVolumeOutput."""
    from scipy.spatial.distance import pdist

    p = output.params.get("vol_params")
    s = {}

    grid = output.neur_vol.shape
    total = int(np.prod(grid))
    s["grid_shape"] = grid
    s["total_voxels"] = total
    s["N_neur"] = int(p.N_neur)
    s["N_den"] = int(p.N_den) if hasattr(p, "N_den") else 0
    s["N_bg"] = int(p.N_bg) if hasattr(p, "N_bg") else 0

    # --- Neuron positions ---
    N = s["N_neur"]
    s["n_locs"] = output.locs.shape[0]
    locs_neur = output.locs[:N]
    if len(locs_neur) > 1:
        dists = pdist(locs_neur)
        s["mean_inter_dist"] = float(np.mean(dists))
        s["min_inter_dist"] = float(np.min(dists))
        s["median_inter_dist"] = float(np.median(dists))

    # --- Vessel ---
    # neur_ves is full-depth (vol_depth + vol_sz[2]) in voxels, not imaging-only.
    # Use the full-depth total for the fraction denominator.
    if output.neur_ves is not None:
        s["vessel_voxels"] = int(np.sum(output.neur_ves > 0))
        full_depth_total = int(np.prod(output.neur_ves.shape))
        s["vessel_fraction"] = s["vessel_voxels"] / full_depth_total
    else:
        s["vessel_voxels"] = 0
        s["vessel_fraction"] = 0.0

    # --- Soma (from gp_soma) ---
    soma_sizes = np.array([len(ss[0]) for ss in output.gp_soma])
    s["n_soma_voxels"] = int(soma_sizes.sum())
    s["soma_sizes"] = soma_sizes
    s["mean_soma_size"] = float(soma_sizes.mean()) if len(soma_sizes) > 0 else 0.0
    s["std_soma_size"] = float(soma_sizes.std()) if len(soma_sizes) > 0 else 0.0

    # --- Nucleus (from gp_nuc) ---
    nuc_sizes = np.array([len(n[0]) for n in output.gp_nuc])
    nuc_values = np.array([n[1] for n in output.gp_nuc])
    s["n_gp_nuc"] = len(output.gp_nuc)
    s["nuc_sizes"] = nuc_sizes
    s["nuc_values"] = nuc_values
    s["mean_nuc_size"] = float(nuc_sizes.mean()) if len(nuc_sizes) > 0 else 0.0

    # --- gp_soma ---
    gp_soma_sizes = np.array([len(ss[0]) for ss in output.gp_soma])
    s["n_gp_soma"] = len(output.gp_soma)
    s["gp_soma_sizes"] = gp_soma_sizes
    s["mean_gp_soma_size"] = float(gp_soma_sizes.mean()) if len(gp_soma_sizes) > 0 else 0.0

    # --- neur_num fill breakdown ---
    nn = output.neur_num
    n_gp_vals = len(output.gp_vals)
    s["n_gp_vals"] = n_gp_vals
    s["neur_num_nonzero"] = int(np.sum(nn > 0))
    s["neur_num_max"] = int(nn.max())

    N_den = s["N_den"]
    s["fill_neuron_only"] = int(np.sum((nn >= 1) & (nn <= N)))
    s["fill_apical"] = int(np.sum((nn > N) & (nn <= N + N_den)))
    s["fill_bg_dendrite"] = int(np.sum((nn > N + N_den) & (nn <= n_gp_vals)))
    s["fill_all_components"] = int(np.sum(nn > 0))
    s["fill_empty"] = int(np.sum(nn == 0))
    s["frac_neuron_only"] = s["fill_neuron_only"] / total
    s["frac_apical"] = s["fill_apical"] / total
    s["frac_bg_dendrite"] = s["fill_bg_dendrite"] / total
    s["frac_all_components"] = s["fill_all_components"] / total
    s["frac_empty"] = s["fill_empty"] / total

    # Per-neuron voxel counts
    per_neuron = np.array([int(np.sum(nn == k)) for k in range(1, N + 1)])
    s["per_neuron_voxels"] = per_neuron
    s["mean_neuron_voxels"] = float(per_neuron.mean())

    # --- Fluorescence ---
    nv = output.neur_vol
    nv_nz = nv[nv > 0]
    s["fluor_n_nonzero"] = len(nv_nz)
    s["fluor_fraction_nonzero"] = len(nv_nz) / total
    if len(nv_nz) > 0:
        s["fluor_mean"] = float(nv_nz.mean())
        s["fluor_std"] = float(nv_nz.std())
        s["fluor_min"] = float(nv_nz.min())
        s["fluor_max"] = float(nv_nz.max())
        s["fluor_median"] = float(np.median(nv_nz))
        s["fluor_p25"] = float(np.percentile(nv_nz, 25))
        s["fluor_p75"] = float(np.percentile(nv_nz, 75))
        s["fluor_p95"] = float(np.percentile(nv_nz, 95))
        s["fluor_p99"] = float(np.percentile(nv_nz, 99))
    s["fluor_nz_values"] = nv_nz  # keep for plotting

    # --- gp_vals ---
    gv_sizes = np.array([len(g.indices) for g in output.gp_vals])
    s["gp_vals_sizes"] = gv_sizes
    s["gp_vals_mean_size"] = float(gv_sizes.mean()) if len(gv_sizes) > 0 else 0.0
    s["gp_vals_std_size"] = float(gv_sizes.std()) if len(gv_sizes) > 0 else 0.0
    # Break down by type
    s["gp_vals_neuron_sizes"] = gv_sizes[:N]
    s["gp_vals_apical_sizes"] = gv_sizes[N:N + N_den] if N + N_den <= len(gv_sizes) else np.array([])
    n_bg_comp = len(gv_sizes) - N - N_den
    s["gp_vals_bg_sizes"] = gv_sizes[N + N_den:] if n_bg_comp > 0 else np.array([])
    s["n_bg_components"] = max(0, n_bg_comp)

    # --- Background ---
    s["n_bg_placed"] = s["n_locs"] - N

    # --- Axons ---
    s["n_gp_bgvals"] = len(output.gp_bgvals)
    bgv_sizes = np.array([len(b[0]) for b in output.gp_bgvals])
    s["gp_bgvals_sizes"] = bgv_sizes
    s["gp_bgvals_mean_size"] = float(bgv_sizes.mean()) if len(bgv_sizes) > 0 else 0.0

    # --- bg_proc ---
    s["n_bg_proc"] = len(output.bg_proc)
    bp_sizes = np.array([len(bp.indices) for bp in output.bg_proc])
    s["bg_proc_sizes"] = bp_sizes
    s["bg_proc_mean_size"] = float(bp_sizes.mean()) if len(bp_sizes) > 0 else 0.0

    # --- neur_num_AD ---
    s["neur_num_AD_nonzero"] = int(np.sum(output.neur_num_ad > 0))
    unique_ad = np.unique(output.neur_num_ad[output.neur_num_ad > 0])
    s["n_apical_components"] = len(unique_ad)

    return s


def print_section(title):
    print()
    print("-" * 110)
    print(f"  {title}")
    print("-" * 110)
    print(f"  {'Metric':45s}  {'MATLAB':>14s}  {'Python':>14s}  {'Diff':>8s}  Status")
    print(f"  {'-'*45}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*6}")


def main():
    parser = argparse.ArgumentParser(description="Compare MATLAB vs Python Phase 1")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--matlab", type=str, default=str(MATLAB_STATS_PATH))
    parser.add_argument("--python", type=str, default=str(PYTHON_OUTPUT_PATH))
    args = parser.parse_args()

    print("Loading MATLAB stats...")
    m = load_matlab_stats(args.matlab)
    is_debug = _safe_attr(m, "source", "") == "matlab_debug"

    print("Loading Python output...")
    py_out = import_pipeline_output(args.python)
    py = extract_python_stats(py_out)

    m_grid = tuple(int(x) for x in np.atleast_1d(m.grid_shape))

    print()
    print("=" * 110)
    print("  MATLAB NAOMi vs calcia Python — Phase 1 Comprehensive Comparison")
    print("=" * 110)
    print(f"  MATLAB: seed={_safe_int(m.seed)}, grid={m_grid}, "
          f"source={'debug run' if is_debug else 'checkpoint export'}")
    print(f"  Python: seed=42, grid={py['grid_shape']}")
    print(f"  Note: Different seeds + different RNG => statistical comparison only")

    all_statuses = []

    def add(label, m_val, p_val, tol_key="structural"):
        status, line = compare_metric(label, m_val, p_val, tol_key)
        if status != "SKIP":
            all_statuses.append(status)
        print(line)

    # ================================================================
    # Section 1: Volume & Neuron Placement
    # ================================================================
    print_section("1. Volume & Neuron Placement")
    add("Total voxels", _safe_int(m.total_voxels), py["total_voxels"], "exact")
    add("Neurons placed", _safe_int(m.n_neurons_placed), py["N_neur"], "exact")
    add("N_den (apical dendrite components)", _safe_int(m.N_den), py["N_den"], "exact")
    add("N_bg (target bg dendrites)", _safe_int(m.N_bg), py["N_bg"], "exact")

    # ================================================================
    # Section 2: Vessels
    # ================================================================
    print_section("2. Blood Vessels (Step 1)")
    # Both vessel_voxels counts are full-depth (vol_depth+vol_sz[2])*vres along z.
    # Vessel generation is stochastic and uses different seeds, so use loose tolerance.
    # Note: vessel_fraction in old MATLAB .mat files used imaging-only denominator;
    #       Python now correctly uses full-depth denominator. Skip fraction comparison
    #       if MATLAB mat was generated before the denominator fix.
    add("Vessel voxels (full-depth)", _safe_int(m.vessel_voxels), py["vessel_voxels"], "component_count")

    # ================================================================
    # Section 3: Soma & Nucleus
    # ================================================================
    print_section("3. Soma & Nucleus (Step 3)")
    m_soma = _safe_int(m.n_soma_voxels) if _safe_attr(m, "has_neur_soma", _safe_attr(m, "n_soma_voxels")) else None
    add("Total soma voxels", m_soma, py["n_soma_voxels"], "structural")
    add("Mean soma size (voxels)", _safe_float(m.mean_soma_size), py["mean_soma_size"], "structural")
    add("Std soma size", _safe_attr(m, "std_soma_size"), py["std_soma_size"], "statistical")
    add("gp_nuc entries", _safe_int(m.n_gp_nuc), py["n_gp_nuc"], "exact")
    add("Mean nucleus size (voxels)", _safe_float(m.mean_nuc_size), py["mean_nuc_size"], "structural")
    add("gp_soma entries", _safe_int(m.n_gp_soma), py["n_gp_soma"], "exact")
    add("Mean gp_soma size", _safe_float(m.mean_gp_soma_size), py["mean_gp_soma_size"], "structural")

    # ================================================================
    # Section 4: Voxel Fill Breakdown (neur_num)
    # ================================================================
    print_section("4. Voxel Fill Breakdown (neur_num)")
    if _safe_attr(m, "has_neur_num", False):
        add("Neuron voxels (1..N_neur)", _safe_int(m.fill_neuron_only), py["fill_neuron_only"], "structural")
        add("Apical dendrite voxels", _safe_int(m.fill_apical), py["fill_apical"], "structural")
        add("BG dendrite voxels", _safe_int(m.fill_bg_dendrite), py["fill_bg_dendrite"], "component_count")
        add("Total occupied voxels", _safe_int(m.fill_all_components), py["fill_all_components"], "structural")
        add("Empty voxels", _safe_int(m.fill_empty), py["fill_empty"], "structural")
        add("Fraction: neuron", _safe_float(m.frac_neuron_only), py["frac_neuron_only"], "fill_fraction")
        add("Fraction: apical", _safe_float(m.frac_apical), py["frac_apical"], "fill_fraction")
        add("Fraction: bg dendrite", _safe_float(m.frac_bg_dendrite), py["frac_bg_dendrite"], "fill_fraction")
        add("Fraction: empty", _safe_float(m.frac_empty), py["frac_empty"], "fill_fraction")
        add("neur_num max", _safe_int(m.neur_num_max), py["neur_num_max"], "structural")
        add("Mean voxels per neuron", _safe_float(m.mean_neuron_voxels), py["mean_neuron_voxels"], "structural")
    else:
        print("  (neur_num not available in MATLAB data — run with debug_opt=true)")
        add("neur_num nonzero (total)", None, py["fill_all_components"])

    # ================================================================
    # Section 5: Apical Dendrites
    # ================================================================
    print_section("5. Apical Dendrites (neur_num_AD)")
    m_ad_nz = _safe_attr(m, "neur_num_AD_nonzero")
    m_ad_comp = _safe_attr(m, "n_apical_components")
    add("neur_num_AD nonzero", _safe_int(m_ad_nz) if m_ad_nz is not None else None,
        py["neur_num_AD_nonzero"], "structural")
    add("Apical components (unique IDs)", _safe_int(m_ad_comp) if m_ad_comp is not None else None,
        py["n_apical_components"], "structural")

    # ================================================================
    # Section 6: Fluorescence (Steps 6-7)
    # ================================================================
    print_section("6. Fluorescence Distribution")
    add("Non-zero fluorescence voxels", _safe_int(m.fluor_n_nonzero), py["fluor_n_nonzero"], "structural")
    add("Fluorescence fraction (non-zero)", _safe_float(m.fluor_fraction_nonzero), py["fluor_fraction_nonzero"], "fill_fraction")
    add("Fluorescence mean", _safe_float(m.fluor_mean), py.get("fluor_mean"), "statistical")
    add("Fluorescence std", _safe_float(m.fluor_std), py.get("fluor_std"), "statistical")
    add("Fluorescence min", _safe_float(m.fluor_min), py.get("fluor_min"), "statistical")
    add("Fluorescence max", _safe_float(m.fluor_max), py.get("fluor_max"), "statistical")
    add("Fluorescence median", _safe_float(m.fluor_median), py.get("fluor_median"), "statistical")
    add("Fluorescence p25", _safe_attr(m, "fluor_p25"), py.get("fluor_p25"), "statistical")
    add("Fluorescence p75", _safe_attr(m, "fluor_p75"), py.get("fluor_p75"), "statistical")
    add("Fluorescence p95", _safe_attr(m, "fluor_p95"), py.get("fluor_p95"), "statistical")
    add("Fluorescence p99", _safe_attr(m, "fluor_p99"), py.get("fluor_p99"), "statistical")

    # ================================================================
    # Section 7: Component Counts (gp_vals)
    # ================================================================
    print_section("7. Component Data (gp_vals)")
    add("gp_vals total components", _safe_int(m.n_gp_vals), py["n_gp_vals"], "structural")
    add("gp_vals mean size (voxels)", _safe_float(m.gp_vals_mean_size), py["gp_vals_mean_size"], "structural")
    add("gp_vals std size", _safe_attr(m, "gp_vals_std_size"), py.get("gp_vals_std_size"), "statistical")
    add("BG dendrite components in gp_vals", _safe_attr(m, "n_bg_components"), py.get("n_bg_components"), "component_count")

    # ================================================================
    # Section 8: Background & Axons (Step 7)
    # ================================================================
    print_section("8. Background & Axons (Step 7)")
    add("BG dendrites placed (locs - N_neur)", _safe_int(m.n_bg_placed), py["n_bg_placed"], "component_count")
    m_bgv = _safe_int(m.n_gp_bgvals) if _safe_attr(m, "has_gp_bgvals", _safe_attr(m, "n_gp_bgvals", 0) > 0) else _safe_int(m.n_gp_bgvals)
    add("gp_bgvals (axon processes)", m_bgv, py["n_gp_bgvals"], "component_count")
    m_bgv_mean = _safe_attr(m, "gp_bgvals_mean_size")
    add("gp_bgvals mean size", _safe_float(m_bgv_mean) if m_bgv_mean is not None else None,
        py["gp_bgvals_mean_size"], "structural")
    add("bg_proc (sorted processes)", _safe_int(m.n_bg_proc), py["n_bg_proc"], "component_count")
    add("bg_proc mean size", _safe_float(m.bg_proc_mean_size), py["bg_proc_mean_size"], "structural")

    # ================================================================
    # Section 9: Spatial Distribution
    # ================================================================
    print_section("9. Spatial Distribution")
    add("Mean inter-neuron distance (um)", _safe_float(m.mean_inter_dist), py.get("mean_inter_dist"), "spatial")
    add("Min inter-neuron distance (um)", _safe_float(m.min_inter_dist), py.get("min_inter_dist"), "spatial")
    add("Median inter-neuron distance (um)", _safe_attr(m, "median_inter_dist"), py.get("median_inter_dist"), "spatial")

    # ================================================================
    # Summary
    # ================================================================
    n_pass = all_statuses.count("PASS")
    n_warn = all_statuses.count("WARN")
    n_fail = all_statuses.count("FAIL")

    print()
    print("=" * 110)
    print(f"  SUMMARY: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL  "
          f"(out of {len(all_statuses)} metrics)")
    if n_fail > 0:
        print("  *** FAILURES detected — investigate discrepancies ***")
    elif n_warn > 0:
        print("  Some warnings — may be expected due to different RNG seeds")
    else:
        print("  All metrics within tolerance!")
    print("=" * 110)

    if args.plot:
        _generate_plots(m, py, py_out)


def _generate_plots(m, py, py_output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle("MATLAB NAOMi vs calcia Python — Phase 1 Comparison", fontsize=14, y=0.98)

    m_seed = _safe_int(m.seed)

    # 1. Fluorescence histogram
    ax = axes[0, 0]
    if hasattr(m, "fluor_hist_counts"):
        m_counts = np.atleast_1d(m.fluor_hist_counts).astype(float)
        m_edges = np.atleast_1d(m.fluor_hist_edges)
        m_centers = (m_edges[:-1] + m_edges[1:]) / 2
        ax.bar(m_centers, m_counts / m_counts.sum(),
               width=(m_edges[1] - m_edges[0]) * 0.8,
               alpha=0.5, label=f"MATLAB (seed={m_seed})", color="tab:blue")
    nv_nz = py.get("fluor_nz_values", np.array([]))
    if len(nv_nz) > 0:
        py_counts, py_edges = np.histogram(nv_nz, bins=100)
        py_centers = (py_edges[:-1] + py_edges[1:]) / 2
        ax.bar(py_centers, py_counts / py_counts.sum(),
               width=(py_edges[1] - py_edges[0]) * 0.8,
               alpha=0.5, label="Python (seed=42)", color="tab:orange")
    ax.set_xlabel("Fluorescence value")
    ax.set_ylabel("Density")
    ax.set_title("Fluorescence Distribution")
    ax.legend()

    # 2. Voxel fill breakdown (stacked bar)
    ax = axes[0, 1]
    has_nn = _safe_attr(m, "has_neur_num", False)
    if has_nn:
        categories = ["Neuron\n(soma+dend)", "Apical\ndendrite", "BG\ndendrite", "Empty"]
        m_vals = [_safe_int(m.fill_neuron_only), _safe_int(m.fill_apical),
                  _safe_int(m.fill_bg_dendrite), _safe_int(m.fill_empty)]
        p_vals_plot = [py["fill_neuron_only"], py["fill_apical"],
                       py["fill_bg_dendrite"], py["fill_empty"]]
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w / 2, m_vals, w, label="MATLAB", color="tab:blue", alpha=0.7)
        ax.bar(x + w / 2, p_vals_plot, w, label="Python", color="tab:orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
    else:
        categories = ["Non-zero", "Empty"]
        m_nz = _safe_int(m.fluor_n_nonzero)
        m_empty = _safe_int(m.total_voxels) - m_nz
        p_nz = py["fluor_n_nonzero"]
        p_empty = py["total_voxels"] - p_nz
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, [m_nz, m_empty], w, label="MATLAB", color="tab:blue", alpha=0.7)
        ax.bar(x + w / 2, [p_nz, p_empty], w, label="Python", color="tab:orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
    ax.set_ylabel("Voxels")
    ax.set_title("Voxel Fill Breakdown")
    ax.legend()

    # 3. Per-neuron soma size distribution
    ax = axes[1, 0]
    m_soma = np.atleast_1d(m.soma_sizes) if hasattr(m, "soma_sizes") else np.array([])
    p_soma = py["soma_sizes"]
    if len(m_soma) > 0:
        ax.hist(m_soma, bins=30, alpha=0.5, label="MATLAB", color="tab:blue", density=True)
    if len(p_soma) > 0:
        ax.hist(p_soma, bins=30, alpha=0.5, label="Python", color="tab:orange", density=True)
    ax.set_xlabel("Soma size (voxels)")
    ax.set_ylabel("Density")
    ax.set_title("Soma Size Distribution")
    ax.legend()

    # 4. Per-neuron total voxels (soma + dendrite)
    ax = axes[1, 1]
    if has_nn and hasattr(m, "per_neuron_voxels"):
        m_pn = np.atleast_1d(m.per_neuron_voxels)
        p_pn = py["per_neuron_voxels"]
        ax.hist(m_pn, bins=30, alpha=0.5, label="MATLAB", color="tab:blue", density=True)
        ax.hist(p_pn, bins=30, alpha=0.5, label="Python", color="tab:orange", density=True)
        ax.set_xlabel("Voxels per neuron (soma + dendrites)")
        ax.set_ylabel("Density")
        ax.set_title("Per-Neuron Volume Distribution")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "neur_num not available\n(run MATLAB with debug_opt=true)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Per-Neuron Volume Distribution")

    # 5. gp_vals component sizes by type
    ax = axes[2, 0]
    m_gv = np.atleast_1d(m.gp_vals_sizes) if hasattr(m, "gp_vals_sizes") else np.array([])
    p_gv = py["gp_vals_sizes"]
    if len(m_gv) > 0:
        ax.hist(m_gv, bins=50, alpha=0.5, label=f"MATLAB ({len(m_gv)})", color="tab:blue", density=True)
    if len(p_gv) > 0:
        ax.hist(p_gv, bins=50, alpha=0.5, label=f"Python ({len(p_gv)})", color="tab:orange", density=True)
    ax.set_xlabel("Component size (voxels)")
    ax.set_ylabel("Density")
    ax.set_title("gp_vals Component Size Distribution")
    ax.legend()

    # 6. Nucleus size distribution
    ax = axes[2, 1]
    m_nuc = np.atleast_1d(m.nuc_sizes) if hasattr(m, "nuc_sizes") else np.array([])
    p_nuc = py["nuc_sizes"]
    if len(m_nuc) > 0:
        ax.hist(m_nuc, bins=30, alpha=0.5, label="MATLAB", color="tab:blue", density=True)
    if len(p_nuc) > 0:
        ax.hist(p_nuc, bins=30, alpha=0.5, label="Python", color="tab:orange", density=True)
    ax.set_xlabel("Nucleus size (voxels)")
    ax.set_ylabel("Density")
    ax.set_title("Nucleus Size Distribution")
    ax.legend()

    plt.tight_layout()
    out_path = CALCIA_ROOT / "comparison_tools" / "comparison_phase1.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\nPlot saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
