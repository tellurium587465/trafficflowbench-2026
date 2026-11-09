import numpy as np
d='E:/tfb_nn/nn_data_afdem/'
n=np.load(d+'names.npy',allow_pickle=True); off=np.load(d+'off.npy'); X=np.load(d+'X.npy',mmap_mode='r')
print(X.shape, X.dtype, len(n), len(off)-1)
print(list(n))
c=np.load(d+'cond.npy',allow_pickle=True); print(np.unique(c,return_counts=True))
ie=np.load(d+'is_eval.npy'); isel=np.load(d+'is_sel.npy'); print(ie.sum(), isel.sum())
p=np.load(d+'panel.npy',allow_pickle=True); print(np.unique(p,return_counts=True))
print(np.unique(np.diff(off)))
