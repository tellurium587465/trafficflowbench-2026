"""G31: where the local in-distribution T2 score is lost (10/12 night; the top teams' LB T2 ~0.91-0.94 exceeds our local
level, so the loss map of our own model shows what a stronger T2 must fix). OOF of the no-memory NN blend on the
selector-like windows (t2_eval_surprise.SEL), production post-processing not applied (in-distribution).
Per window IoU (expected-IoU set); by condition, by how often the true site queues in other months (f), weekday/weekend."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_eval_surprise as S
from t2_eval_oof import T, load_oof
SPEC = "q25_nomem_s3+q25_nomem_s4+q31_nomem_wp05_s0:2+q31_nomem_attn1_s0:2+q31_nomem_r13_s0+q31_nomem_drop1_s0"
oof = load_oof(SPEC)


def iou(p, lab, el):
    pe = p.ravel(); o = np.argsort(-pe); cs = np.cumsum(pe[o]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[o[:kb]] = True; s = s.reshape(p.shape)
    u = ((s | lab) & el).sum(); return 1.0 if u == 0 else (s & lab & el).sum() / u, int(s.sum()), int((lab & el).sum())


rows = []
for j, k in enumerate(S.SEL):
    if k not in oof:
        continue
    lab, el = T.LAB[k]; v, ns, nt = iou(oof[k], lab.astype(bool), el.astype(bool))
    rows.append(dict(cond=str(T.META["cond"][k]), f=S.F[j], iou=v, npred=ns, ntrue=nt,
                     we=pd.Timestamp(str(T.META["date"][k])[:10]).dayofweek >= 5))
R = pd.DataFrame(rows)
for c, g in R.groupby("cond"):
    print(f"== {c}: n {len(g)}, mean IoU {g.iou.mean():.4f}; IoU=0 share {np.mean(g.iou == 0):.3f} (loss share {np.sum(g.iou == 0) / max(np.sum(1 - g.iou), 1e-9):.2f}); IoU<0.5 share {np.mean(g.iou < 0.5):.3f}")
    g = g.assign(fb=pd.cut(g.f, [-0.01, 0.1, 0.3, 0.6, 0.9, 1.01]))
    t = g.groupby("fb", observed=True).agg(n=("iou", "size"), iou=("iou", "mean"), zero=("iou", lambda x: np.mean(x == 0)))
    t["loss_share"] = g.groupby("fb", observed=True).apply(lambda x: (1 - x.iou).sum(), include_groups=False) / (1 - g.iou).sum()
    print(t.round(3).to_string())
    print(f"   weekday {g[~g.we].iou.mean():.4f} / weekend {g[g.we].iou.mean():.4f}; true cells median {g.ntrue.median():.0f}, predicted median {g.npred.median():.0f}")
