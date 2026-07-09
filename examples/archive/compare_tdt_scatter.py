"""Batch-compare figure for the tdt physical-scatter sweep (composite OFF, flat):
real tdt zoom-in (scale-matched) vs each scatter config. spatial_cv (real ~0.10)
+ floor_frac (real ~0.75) + holes annotated."""
import glob, json, os
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, zoom, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\boyuan\Documents\GitHub\calcia"
OUT = os.path.join(ROOT, "examples", "output")
STUB = "deepthinves_volume_500um_d180_stub"
WANT = [8.0, 16.0, 24.0]
SIM_FOV_UM, REAL_FOV_UM = 500.0, 1700.0

def spatial_cv(img):
    H,W=img.shape; c=img[int(H*0.2):int(H*0.8),int(W*0.2):int(W*0.8)].astype(np.float32)
    flat=c/(gaussian_filter(c,40)+1e-6); return float(flat.std()/(flat.mean()+1e-9))
def floor_frac(img):
    f=img.astype(np.float32); return float((f < 0.5*np.median(f)).mean())  # dark-floor fraction proxy
def holes(frame):
    f=frame.astype(np.float32); base=gaussian_filter(f,25)
    dark=(f<0.55*base)&(base>np.percentile(base,40)); lab,_=label(dark)
    s=np.bincount(lab.ravel())[1:]; return int(((s>20)&(s<1500)).sum())

# newest tdt run per wanted scatter (this sweep: composite off, flat)
best={}
for d in sorted(glob.glob(os.path.join(OUT,"striatum_tdt_static_500um_*")), key=os.path.getmtime):
    try: meta=json.load(open(os.path.join(d,"metadata.json")))
    except Exception: continue
    if meta.get("matched_run")!=STUB: continue
    sc=meta.get("scatter_um")
    if sc in WANT: best[sc]=d

rows=[]
for sc in WANT:
    if sc not in best: print(f"MISSING tdt scatter={sc}"); continue
    try: mov=np.load(os.path.join(best[sc],"movies.npz"))["mov_noisy"].astype(np.float32)
    except Exception as e: print(f"CORRUPT scatter={sc}: {e}"); continue
    mean=mov.mean(0); fr=mov[mov.shape[0]//2]
    rows.append(dict(sc=sc, cv=spatial_cv(mean), ff=floor_frac(mean), holes=holes(fr),
                     frame=fr, mean=mean))

# real tdt, scale-matched
rf=sorted(glob.glob(os.path.join(ROOT,"data","real","tdt-bfp","*tdt*.h5")))[0]
with h5py.File(rf,"r") as h: real=h["images"][:].astype(np.float32)
Hr=real.shape[1]; crop=int(round(SIM_FOV_UM/(REAL_FOV_UM/Hr))); c=Hr//2; hh=crop//2
real_c=real[:,c-hh:c+hh,c-hh:c+hh]; simH=rows[0]["mean"].shape[0]
rmean=zoom(real_c.mean(0),simH/crop,order=1); rframe=zoom(real_c[real_c.shape[0]//2],simH/crop,order=1)
rcv=spatial_cv(real_c.mean(0)); rff=floor_frac(real_c.mean(0))

print(f"\n{'scatter':>8} {'spatial_cv':>11} {'floor_frac':>11} {'holes':>6}   (real cv~0.10)")
print(f"{'REAL':>8} {rcv:>11.3f} {rff:>11.3f} {'-':>6}")
for r in rows: print(f"{r['sc']:>8} {r['cv']:>11.3f} {r['ff']:>11.3f} {r['holes']:>6}")

def nz(a,lo=1,hi=99):
    p0,p1=np.percentile(a,[lo,hi]); return np.clip((a-p0)/(p1-p0+1e-9),0,1)
n=len(rows)+1
fig,ax=plt.subplots(2,n,figsize=(3.4*n,7))
ax[0,0].imshow(nz(rframe),cmap="gray"); ax[0,0].set_title(f"REAL tdt {SIM_FOV_UM:.0f}um\ncv{rcv:.2f} ff{rff:.2f}")
ax[1,0].imshow(nz(rmean),cmap="gray"); ax[1,0].set_title("REAL mean")
for i,r in enumerate(rows,1):
    ax[0,i].imshow(nz(r["frame"]),cmap="gray")
    ax[0,i].set_title(f"scatter={r['sc']:.0f}\ncv{r['cv']:.2f} ff{r['ff']:.2f} h{r['holes']}")
    ax[1,i].imshow(nz(r["mean"]),cmap="gray"); ax[1,i].set_title(f"scatter={r['sc']:.0f} mean")
for a in ax.ravel(): a.axis("off")
fig.suptitle("tdt physical-scatter sweep (composite OFF, flat) vs real — scale-matched; frame(top)/mean(bottom)")
fig.tight_layout()
outp=os.path.join(OUT,"_tdt_scatter_compare.png"); fig.savefig(outp,dpi=80); plt.close()
print("saved:",outp)
