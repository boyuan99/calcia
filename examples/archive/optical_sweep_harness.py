"""Load-once optical-domain sweep on the deep-500 volume. Composite OFF: probe the
PHYSICAL PSF-breadth levers (obj_na, scatter_um, scatter_length_um_wf) and see which
optical method best matches real GCaMP (dff~0.20, texture_cv~0.14) WITHOUT the
post-scan composite. Volume + traces logic via the tested run_gcamp; nt small."""
import os, sys, glob
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = r"C:\Users\boyuan\Documents\GitHub\calcia\examples"
sys.path.insert(0, HERE)
import demo_gcamp_realistic_matched as G  # noqa
import _striatum_common as C  # noqa

OUT = os.path.join(HERE, "output")
STUB = "deepthinves_volume_500um_d180_stub"
NT = 24
SIM_FOV_UM, REAL_FOV_UM = 500.0, 1700.0

# SINGLE-Gaussian scatter (one width -> holes OR washed cells) vs TWO-SCALE PSF
# (sharp core + wide halo -> both), the real 1p PSF shape. All flat, composite OFF.
CONFIGS = [
    ("sharp ref",             dict(scatter_um=0, flat=True)),
    ("single scatter24",      dict(scatter_um=24, flat=True)),
    ("2scale h18 w0.5",       dict(halo_um=18, halo_weight=0.5, flat=True)),
    ("2scale h18 w0.7",       dict(halo_um=18, halo_weight=0.7, flat=True)),
    ("2scale h15 w0.6",       dict(halo_um=15, halo_weight=0.6, flat=True)),
    ("2scale h22 w0.65",      dict(halo_um=22, halo_weight=0.65, flat=True)),
]

def dff99(mov, bias):
    m = np.transpose(mov, (1,2,0)).astype(np.float32); sig = np.clip(m-bias,0,None)
    f0 = np.percentile(sig,10,axis=2,keepdims=True); mk = f0.squeeze() >= np.percentile(f0,60)
    return float(np.percentile(((sig-f0)/(f0+1e-6))[mk],99))
def texcv(img):
    H,W=img.shape; c=img[int(H*0.2):int(H*0.8),int(W*0.2):int(W*0.8)].astype(np.float32)
    flat=c/(gaussian_filter(c,40)+1e-6); return float(flat.std()/(flat.mean()+1e-9))
def holes(frame):
    f=frame.astype(np.float32); base=gaussian_filter(f,25)
    dark=(f<0.55*base)&(base>np.percentile(base,40)); lab,_=label(dark)
    s=np.bincount(lab.ravel())[1:]; return int(((s>20)&(s<1500)).sum())

print("loading deep-500 once...")
vol_out, vol_params, meta = G.load_matched_volume(STUB)
C.fill_nuclei(vol_out)
seed = int(meta["seed"]); illum_cfg = C.IllumConfig(**meta["config"]["illum"])
focal = meta.get("focal_depth_um")

results = []
for label_, kw in CONFIGS:
    print(f"\n### {label_}  {kw}")
    ic = None if kw.get("flat") else illum_cfg
    noisy, clean, *_ = G.run_gcamp(
        vol_out, vol_params, NT, seed, 0.02, kw.get("scatter_um", 0),
        "physio", seed+3, ic, focal_um=focal,
        neuropil_smooth_um=0,                    # COMPOSITE OFF
        obj_na=kw.get("obj_na"), scatter_length_um_wf=kw.get("scatter_length_um_wf"),
        halo_um=kw.get("halo_um", 18.0), halo_weight=kw.get("halo_weight", 0.0))
    noisy = np.transpose(noisy, (2, 0, 1))       # run_gcamp returns (H,W,T) -> (T,H,W)
    bias = C.StriatumConfig().build_cam().bias
    bright = noisy[np.argmax(noisy.reshape(noisy.shape[0],-1).mean(1))]
    r = dict(label=label_, dff=dff99(noisy,bias), cv=texcv(noisy.mean(0)),
             holes=holes(bright), bright=bright, mean=noisy.mean(0))
    results.append(r)
    print(f"    dff_p99={r['dff']:.3f}  texture_cv={r['cv']:.3f}  holes={r['holes']}")

# real reference, scale-matched
rf = sorted(glob.glob(os.path.join(os.path.dirname(HERE),"data","real","striatum_raw_samples_15","*.tif")))[0]
real = tifffile.imread(rf).astype(np.float32); Hr=real.shape[1]
crop=int(round(SIM_FOV_UM/(REAL_FOV_UM/Hr))); c=Hr//2; h=crop//2
real_c=real[:,c-h:c+h,c-h:c+h]; simH=results[0]["mean"].shape[0]
real_dff=dff99(real_c,np.percentile(real_c,2)); real_cv=texcv(real_c.mean(0))

print("\n==== SUMMARY (real dff~0.20 cv~0.14) ====")
print(f"{'method':>20} {'dff_p99':>8} {'tex_cv':>7} {'holes':>6}")
print(f"{'REAL':>20} {real_dff:>8.3f} {real_cv:>7.3f} {'-':>6}")
for r in results:
    print(f"{r['label']:>20} {r['dff']:>8.3f} {r['cv']:>7.3f} {r['holes']:>6}")

def nz(a,lo=1,hi=99):
    p0,p1=np.percentile(a,[lo,hi]); return np.clip((a-p0)/(p1-p0+1e-9),0,1)
n=len(results)+1
fig,ax=plt.subplots(2,n,figsize=(3.2*n,6.8))
ax[0,0].imshow(nz(zoom(real_c[np.argmax(real_c.reshape(real_c.shape[0],-1).mean(1))],simH/crop,order=1)),cmap="gray")
ax[0,0].set_title(f"REAL {SIM_FOV_UM:.0f}um\ndff{real_dff:.2f} cv{real_cv:.2f}")
ax[1,0].imshow(nz(zoom(real_c.mean(0),simH/crop,order=1)),cmap="gray"); ax[1,0].set_title("REAL mean")
for i,r in enumerate(results,1):
    ax[0,i].imshow(nz(r["bright"]),cmap="gray")
    ax[0,i].set_title(f"{r['label']}\ndff{r['dff']:.2f} cv{r['cv']:.2f} h{r['holes']}",fontsize=8)
    ax[1,i].imshow(nz(r["mean"]),cmap="gray"); ax[1,i].set_title(f"{r['label']} mean",fontsize=8)
for a in ax.ravel(): a.axis("off")
fig.suptitle("TWO-SCALE PSF (core+halo) vs single scatter vs real GCaMP — scale-matched; brightest(top)/mean(bottom)")
fig.tight_layout()
outp=os.path.join(OUT,"_twoscale_gcamp.png"); fig.savefig(outp,dpi=78); plt.close()
print("saved:",outp)
