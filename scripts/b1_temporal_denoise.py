"""B1 — self-supervised spatiotemporal denoise BASELINE (no motion compensation).

GO/NO-GO experiment (DENOISE_PLAN Part B). See docs/todo/B1_EXPERIMENT_LOG.md for the running log.

Premise (measured, 2026-07-25): on genuinely RAW drift-corrected 15 s data the consecutive-frame delta
is FLAT across frame gaps (gap1≈gap8) => it is almost entirely NOISE, not motion => Noise2Noise is
well-posed. (The earlier "SSIM~0.9 redundancy" was measured on already-cellpose-denoised data and
described leftover motion, not available redundancy.)

Method:
  * N2N (Lehtinen 2018): input frame x_t, target adjacent frame x_{t±1}; L1; the two frames are the same
    signal + independent noise, so net(x_t) -> the clean-signal estimator. Inference denoise(t)=net(x_t)
    (per-frame; uses the frame's own content, so moving cells are preserved not interpolated).
  * Network: compact UNet from coastal ConvBlock/Encoder/Decoder (Decision 4), 1->1, depth 3.
  * fg-WEIGHTED patch sampling: the FOV is ~90% dark background (sparse labelled cells), so patches are
    drawn centred on foreground pixels — else the net just learns to output background.

Honest eval WITHOUT clean GT — pseudo-GT + visual QC (the metric the plan calls for):
  * pseudo-GT[t] = temporal MEAN over [t-W..t+W]; because the data is noise-dominated, the mean is a
    near-clean estimate of the underlying signal in STATIC regions (noise ↓ ~sqrt(2W+1)).
  * Score each single-frame denoiser (raw / Cellpose-3 / learned-N2N) by PSNR & SSIM to pseudo-GT,
    restricted to STATIC foreground (Farneback flow < 0.5 px AND pseudo-GT > fg thr) — the only region
    where the pseudo-GT is trustworthy. This distinguishes noise-removal from structure-smear, which the
    raw-space Sobel metric cannot on noisy data.
  * Visual QC montage PNG: raw | Cellpose | N2N | pseudo-GT on fg-rich crops of a few val frames.

Run (cecelia pixi env): cd cecelia-pineapple && pixi run python <this> [--image PATH] [--channel 2]
Default image: Dnm0rS (full-FOV raw->drift copy, obWDNS/zolIMa). Channels [SHG,nuc-GFP,mem-TOM,CD169].
"""
import sys, types, argparse, numpy as np, cv2, torch, torch.nn as nn
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

CO = "/home/dominik/cc-workspace/coastal-temporal"
_pkg = types.ModuleType("coastal"); _pkg.__path__ = [CO + "/coastal"]; sys.modules["coastal"] = _pkg
from coastal.model import ConvBlock, Encoder, Decoder
from coastal.denoise import DenoiseModel
import cecelia.utils.zarr_utils as zu

SCRATCH = "/tmp/claude-1000/-home-dominik-cc-workspace-cecelia/c34a0b4b-f421-410b-a75a-e264f3c6f4e7/scratchpad"
ap = argparse.ArgumentParser()
ap.add_argument("--image", default="/home/dominik/cecelia-pineapple/projects/zolIMa/0/Dnm0rS/ccidDriftCorrected.ome.zarr")
ap.add_argument("--channel", type=int, default=2)               # 2 = mem-TOM (stress channel)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--zslab", type=int, default=1)                 # +-N z around mid-Z
ap.add_argument("--patch", type=int, default=128)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--win", type=int, default=4)                   # pseudo-GT temporal half-window (2W+1 frames)
ap.add_argument("--tag", default="dnm_memtom")
args = ap.parse_args()

OUT = open(f"{SCRATCH}/b1_{args.tag}.txt", "w")
def log(*a): print(*a, file=OUT); OUT.flush(); print(*a)
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CHN = {1: "nuc-GFP", 2: "mem-TOM", 3: "CD169-Kat"}; ch = args.channel; ps = args.patch
log(f"B1 temporal denoise — {args.image.split('/')[-3]} ch{ch}({CHN.get(ch,'?')}) dev={dev} tag={args.tag}")

# ── data (global norm; noise-dominated so one scale is valid and keeps neighbours comparable) ────────
arr, _ = zu.open_as_zarr(args.image, as_dask=True); arr = arr[0]
T, C, Z, Y, X = arr.shape; zmid = Z // 2; zlo, zhi = zmid - args.zslab, zmid + args.zslab + 1
raw = np.asarray(arr[:, ch, zlo:zhi]).astype(np.float32)        # [T, Zs, Y, X]
Zs = raw.shape[1]
_GLO, _GHI = np.percentile(raw, 1), np.percentile(raw, 99.5)
def norm(f): return np.clip((f - _GLO) / max(_GHI - _GLO, 1e-6), 0, 1).astype(np.float32)
N = np.stack([np.stack([norm(raw[t, z]) for z in range(Zs)]) for t in range(T)])   # [T,Zs,Y,X] float32
FGTHR = 0.2
log(f"T={T} Z[{zlo}:{zhi}]={Zs} Y={Y} X={X}  glo={_GLO:.0f} ghi={_GHI:.0f}  fg~{(N>FGTHR).mean():.1%}")

T_TRAIN_HI, T_VAL_LO = 140, 145
train_ts = list(range(1, T_TRAIN_HI - 1)); val_ts = list(range(T_VAL_LO, T - 1))
# precompute fg pixel pools per (train t, z) for fg-weighted sampling (patch centres land on cells)
half = ps // 2
def fg_centres(t, z):
    m = N[t, z] > FGTHR
    m[:half] = m[-half:] = False; m[:, :half] = m[:, -half:] = False   # keep patch in-bounds
    ys, xs = np.nonzero(m)
    return ys, xs

def sample_batch():
    xa = np.empty((args.batch, 1, ps, ps), np.float32); xb = np.empty_like(xa)
    for i in range(args.batch):
        t = train_ts[np.random.randint(len(train_ts))]; z = np.random.randint(Zs)
        nb = t + (1 if np.random.rand() < 0.5 else -1)
        ys, xs = fg_centres(t, z)
        if len(ys):
            k = np.random.randint(len(ys)); cy, cx = ys[k], xs[k]; yy, xx = cy - half, cx - half
        else:
            yy, xx = np.random.randint(0, Y - ps), np.random.randint(0, X - ps)
        xa[i, 0] = N[t,  z, yy:yy+ps, xx:xx+ps]; xb[i, 0] = N[nb, z, yy:yy+ps, xx:xx+ps]
    return torch.from_numpy(xa).to(dev), torch.from_numpy(xb).to(dev)

# ── model ────────────────────────────────────────────────────────────────────────────────────────
class DenoiseUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, init=32, depth=3):
        super().__init__()
        self.encoders = nn.ModuleList(); c = in_ch
        for i in range(depth):
            o = init * 2**i; self.encoders.append(Encoder(c, o)); c = o
        self.bottleneck = ConvBlock(c, init * 2**depth)
        self.decoders = nn.ModuleList(); bc = init * 2**depth
        for i in reversed(range(depth)):
            o = init * 2**i; self.decoders.append(Decoder(bc, o)); bc = o
        self.head = nn.Conv2d(init, out_ch, 1)
    def forward(self, x):
        skips = []
        for e in self.encoders:
            s, x = e(x); skips.append(s)
        x = self.bottleneck(x)
        for d, s in zip(self.decoders, reversed(skips)):
            x = d(x, s)
        return self.head(x)

net = DenoiseUNet().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
use_amp = (dev == "cuda"); scaler = torch.amp.GradScaler("cuda", enabled=use_amp); l1 = nn.L1Loss()
log(f"\ntraining {args.iters} iters (fg-weighted, batch {args.batch}, patch {ps})…")
net.train()
for it in range(1, args.iters + 1):
    xa, xb = sample_batch(); opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        loss = l1(net(xa), xb)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if it % 400 == 0 or it == 1: log(f"  iter {it:5d}  L1 {loss.item():.4f}")

# ── eval: pseudo-GT (flow-gated temporal mean) + visual QC ───────────────────────────────────────
net.eval(); dn = DenoiseModel(model_type="denoise_cyto3", device=None); W = args.win; zc = Zs // 2
def flowmag(a, b):
    fl = cv2.calcOpticalFlowFarneback((b*255).astype(np.uint8), (a*255).astype(np.uint8),
                                      None, 0.5, 3, 25, 3, 5, 1.2, 0)
    return np.sqrt(fl[..., 0]**2 + fl[..., 1]**2)
def net_dn(a):
    with torch.no_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        y = net(torch.from_numpy(a[None, None]).to(dev))
    return np.clip(y[0, 0].float().cpu().numpy(), 0, 1)

scores = {k: {"psnr": [], "ssim": []} for k in ("raw", "cellpose", "net")}
panels = []
for t in val_ts:
    if t - W < 0 or t + W >= T: continue
    a = N[t, zc]
    pgt = N[t-W:t+W+1, zc].mean(0)                              # temporal-mean pseudo-GT (near-clean in static)
    # static-fg validity: low flow to both neighbours AND foreground in the pseudo-GT
    stat = (flowmag(a, N[t-1, zc]) < 0.5) & (flowmag(a, N[t+1, zc]) < 0.5) & (pgt > FGTHR)
    if stat.sum() < 200: continue
    dc = norm(dn.eval(raw[t, zc], diameter=8., autocast=use_amp)); dnet = net_dn(a)
    for k, p in (("raw", a), ("cellpose", dc), ("net", dnet)):
        scores[k]["psnr"].append(float(psnr(pgt[stat], p[stat], data_range=1.0)))
        # window SSIM needs 2D; score a bbox around the static region, masked
        scores[k]["ssim"].append(float(ssim(pgt*stat, p*stat, data_range=1.0)))
    if len(panels) < 4:                                        # collect fg-rich crops for the montage
        ys, xs = np.nonzero(pgt > FGTHR)
        if len(ys):
            cy, cx = int(ys.mean()), int(xs.mean()); s = 160
            y0, x0 = max(0, cy-s), max(0, cx-s)
            sl = (slice(y0, y0+2*s), slice(x0, x0+2*s))
            panels.append(np.concatenate([a[sl], dc[sl], dnet[sl], pgt[sl]], axis=1))

def m(x): return float(np.mean(x)) if x else float("nan")
log(f"\n=== quality vs temporal-mean pseudo-GT (STATIC fg only; W={W} → {2*W+1}-frame mean) ===")
log(f"{'':10s}{'PSNR(dB)':>10s}{'SSIM':>8s}")
for k in ("raw", "cellpose", "net"):
    log(f"{k:10s}{m(scores[k]['psnr']):>10.2f}{m(scores[k]['ssim']):>8.3f}")
pr, pc, pn = m(scores['raw']['psnr']), m(scores['cellpose']['psnr']), m(scores['net']['psnr'])
log(f"\nPSNR gain vs raw:  cellpose {pc-pr:+.2f} dB   learned-N2N {pn-pr:+.2f} dB")
log(f"learned-N2N vs cellpose: {pn-pc:+.2f} dB  (>0 => temporal self-sup beats off-the-shelf single-frame denoise)")

if panels:
    mont = (np.clip(np.concatenate(panels, axis=0), 0, 1) * 255).astype(np.uint8)
    outpng = f"{SCRATCH}/b1_{args.tag}_qc.png"; cv2.imwrite(outpng, mont)
    log(f"\nQC montage (cols: raw | cellpose | N2N | pseudo-GT; rows: val frames) -> {outpng}")
log("\nGO/NO-GO: learned-N2N should beat BOTH raw and cellpose in static-fg PSNR/SSIM to pseudo-GT,")
log("AND the QC montage must show noise removed WITHOUT washing out cell structure.")
log("DONE")
OUT.close()
