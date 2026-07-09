"""Clean scatter progression 0->4->6->8->12 (all VIGNETTE, composite OFF, nt=200)
vs real GCaMP, scale-matched. Find the sweet spot between too-sharp (0) and
too-blurry (12)."""
import glob, json, os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\boyuan\Documents\GitHub\calcia"
OUT = os.path.join(ROOT, "examples", "output")
VIGN_STUB = "deepthinves_volume_500um_d180_stub"
WANT = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
SIM_FOV_UM, REAL_FOV_UM = 500.0, 1700.0

def dff99(mov, bias):
    m = np.transpose(mov,(1,2,0)).astype(np.float32); sig=np.clip(m-bias,0,None)
    f0=np.percentile(sig,10,axis=2,keepdims=True); mk=f0.squeeze()>=np.percentile(f0,60)
    return float(np.percentile(((sig-f0)/(f0+1e-6))[mk],99))
def texcv(img):
    H,W=img.shape; c=img[int(H*0.2):int(H*0.8),int(W*0.2):int(W*0.8)].astype(np.float32)
    flat=c/(gaussian_filter(c,40)+1e-6); return float(flat.std()/(flat.mean()+1e-9))
def holes(frame):
    f=frame.astype(np.float32); base=gaussian_filter(f,25)
    dark=(f<0.55*base)&(base>np.percentile(base,40)); lab,_=label(dark)
    s=np.bincount(lab.ravel())[1:]; return int(((s>20)&(s<1500)).sum())

# pick newest vignette composite-OFF run per wanted scatter
best = {}
for d in sorted(glob.glob(os.path.join(OUT,"gcamp_realistic_500um_*")), key=os.path.getmtime):
    try: meta=json.load(open(os.path.join(d,"metadata.json")))
    except Exception: continue
    if meta.get("matched_run")!=VIGN_STUB: continue
    sc=meta.get("scatter_um")
    if sc in WANT: best[sc]=d          # newest wins (sorted asc mtime)

rows=[]
for sc in WANT:
    if sc not in best: print(f"MISSING scatter={sc}"); continue
    try:
        mov=np.load(os.path.join(best[sc],"movies.npz"))["mov_noisy"].astype(np.float32)
    except Exception as e:
        print(f"CORRUPT scatter={sc} ({os.path.basename(best[sc])}): {e}"); continue
    bright=mov[np.argmax(mov.reshape(mov.shape[0],-1).mean(1))]; bias=float(np.percentile(mov,2))
    rows.append(dict(sc=sc, dff=dff99(mov,bias), cv=texcv(mov.mean(0)),
                     holes=holes(bright), bright=bright, mean=mov.mean(0)))

rf=sorted(glob.glob(os.path.join(ROOT,"data","real","striatum_raw_samples_15","*.tif")))[0]
real=tifffile.imread(rf).astype(np.float32); Hr=real.shape[1]
crop=int(round(SIM_FOV_UM/(REAL_FOV_UM/Hr))); c=Hr//2; h=crop//2
real_c=real[:,c-h:c+h,c-h:c+h]; simH=rows[0]["mean"].shape[0]
rdff=dff99(real_c,np.percentile(real_c,2)); rcv=texcv(real_c.mean(0))

print(f"\n{'scatter':>8} {'dff_p99':>8} {'tex_cv':>7} {'holes':>6}")
print(f"{'REAL':>8} {rdff:>8.3f} {rcv:>7.3f} {'-':>6}")
for r in rows: print(f"{r['sc']:>8} {r['dff']:>8.3f} {r['cv']:>7.3f} {r['holes']:>6}")

def nz(a,lo=1,hi=99):
    p0,p1=np.percentile(a,[lo,hi]); return np.clip((a-p0)/(p1-p0+1e-9),0,1)
n=len(rows)+1
fig,ax=plt.subplots(2,n,figsize=(3.4*n,7))
ax[0,0].imshow(nz(zoom(real_c[np.argmax(real_c.reshape(real_c.shape[0],-1).mean(1))],simH/crop,order=1)),cmap="gray")
ax[0,0].set_title(f"REAL {SIM_FOV_UM:.0f}um\ndff{rdff:.2f} cv{rcv:.2f}")
ax[1,0].imshow(nz(zoom(real_c.mean(0),simH/crop,order=1)),cmap="gray"); ax[1,0].set_title("REAL mean")
for i,r in enumerate(rows,1):
    ax[0,i].imshow(nz(r["bright"]),cmap="gray")
    ax[0,i].set_title(f"scatter={r['sc']:.0f}\ndff{r['dff']:.2f} cv{r['cv']:.2f} h{r['holes']}")
    ax[1,i].imshow(nz(r["mean"]),cmap="gray"); ax[1,i].set_title(f"scatter={r['sc']:.0f} mean")
for a in ax.ravel(): a.axis("off")
fig.suptitle("GCaMP scatter progression (vignette, composite OFF) vs real — pick sweet spot between 0 (sharp) and 12 (blurry)")
fig.tight_layout()
outp=os.path.join(OUT,"_scatter_progression.png"); fig.savefig(outp,dpi=80); plt.close()
print("saved:",outp)
