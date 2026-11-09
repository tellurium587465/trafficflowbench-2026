p = "C:/Users/user/tfb/audit/c2c_floor_panel.py"; s = open(p, encoding="utf-8").read()
s = s.replace('"""C2b: reducible part', '"""C2c (= C2b with the white-noise level measured per panel from the truth: sd of the time 2nd difference / sqrt(1.5), free-flow cells): reducible part', 1)
s = s.replace("def loss(pc):", '''def noise_sd(Z):
    out = []
    for M in (Z["TV"], Z["TQ"]):
        ff = np.isfinite(Z["TV"]) & (Z["TV"] > 0.9 * Z["vf"][None, :])
        d2 = (M[1:-1] - 0.5 * (M[:-2] + M[2:])) / M[1:-1]
        ok = ff[1:-1] & ff[:-2] & ff[2:] & np.isfinite(d2)
        x = d2[ok]; x = x[np.abs(x) < 10 * np.nanstd(x)]
        out.append(float(np.std(x) / np.sqrt(1.5)))
    return out


def loss(pc):''', 1)
s = s.replace('rows.append(dict(panel=p, regime=r, kind="ALL",', 'rows.append(dict(panel=p, regime=r, kind="ALL", sv=sv, sq=sq,', 1)
s = s.replace('print("S_LWR proxy now vs noise floor', 'print("measured noise sd per panel (v, q):", a.groupby("panel")[["sv", "sq"]].first().round(4).to_dict())\nprint("S_LWR proxy now vs noise floor', 1)
open(p, "w", encoding="utf-8").write(s)
print("patched", s.count("noise_sd"), s.count("sv * rng"))
