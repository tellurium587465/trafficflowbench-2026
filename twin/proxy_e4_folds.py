"""E4 proxy (strategy session 10/22; STATE.md pre-registration): does one added shifted month help other shifted months,
and with which weight? After build_merge wrote NN/nn_data_e4p1 = nn_data_afdem + twin family C worlds
C6 (W1 = the added month, calendar 2220-10, template 2030-12) and C7 / C8 / C16 / C17 (the scored months, templates
2031-01 / 2031-02), set per window:
  W1           -> fold_month "proxyW1" (trained in month-CV fold 2), is_eval False
  scored worlds-> fold_month = template (held out in fold 2), is_eval True (their OOF is written)
and make NN/nn_data_e4p0 = the same arrays with W1 held out and never scored (fold_month 2030-12, is_eval False):
hard links for the arrays, own fold_month / is_eval.
usage: python proxy_e4_folds.py"""
import os, sys, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN

W1, EVAL = "2220-10", ("2223-10", "2224-05", "2229-07", "2229-08")
D1, D0 = f"{NN}/nn_data_e4p1", f"{NN}/nn_data_e4p0"
date = np.array([str(d)[:7] for d in np.load(f"{D1}/date.npy", allow_pickle=True)])
fm = np.load(f"{D1}/fold_month.npy", allow_pickle=True).astype(object); ev = np.load(f"{D1}/is_eval.npy").astype(bool)
w1, sc = date == W1, np.isin(date, EVAL)
assert w1.sum() > 0 and sc.sum() > 0 and set(fm[sc]) <= {"2031-01", "2031-02"} and set(fm[w1]) == {"2030-12"}, "unexpected worlds / templates"
fm1, ev1 = fm.copy(), ev.copy(); fm1[w1] = "proxyW1"; ev1[sc] = True
np.save(f"{D1}/fold_month.npy", fm1.astype(str)); np.save(f"{D1}/is_eval.npy", ev1)
os.makedirs(D0, exist_ok=True)
for f in os.listdir(D1):
    if f not in ("fold_month.npy", "is_eval.npy") and not os.path.exists(f"{D0}/{f}"):
        os.link(f"{D1}/{f}", f"{D0}/{f}")
np.save(f"{D0}/fold_month.npy", fm.astype(str)); np.save(f"{D0}/is_eval.npy", ev1)
print("W1 windows", int(w1.sum()), "| scored synthetic windows", int(sc.sum()), "| total", len(date))
