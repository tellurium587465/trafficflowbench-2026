import time, sys, numpy as np, torch
sys.argv=["x","prof","1"]
t0=time.time()
exec(open("/mnt/c/Users/user/tfb/src/t2_nn.py").read().split("oof = {}")[0])
print("load+setup %.1fs"%(time.time()-t0),flush=True)
tr=np.flatnonzero(~np.isin(mon,FOLDS[0]) & ~is_sel); mu,sd=stats(tr); mu=np.nan_to_num(mu); sd=np.where(np.isfinite(sd),sd,1.0).astype(np.float32)
net=Net(X.shape[1]).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
ks=[tr[i:i+64] for i in range(0,64*30,64)]
a=time.time(); bs=[batch(k,mu,sd) for k in ks]; torch.cuda.synchronize(); b=time.time()
print("batch build 30: %.2fs (%.3f s/batch)"%(b-a,(b-a)/30),flush=True)
for it in range(3):
    a=time.time()
    for xb,yb,eb,mb,wb in bs:
        logit=net(xb,mb); l=torch.nn.functional.binary_cross_entropy_with_logits(logit,yb,reduction="none")*eb
        loss=(l.sum(dim=(1,2))*wb).sum()/(eb.sum()+1); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); print("gpu fwd/bwd 30: %.2fs (%.3f s/batch)"%(time.time()-a,(time.time()-a)/30),flush=True)
