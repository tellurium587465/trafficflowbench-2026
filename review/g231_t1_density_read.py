"""G231 (11/03, ledger #194): read t1d.log. Per seed, month and read-rho, the even-day S of each train-rho arm after the 32-epoch stage A
against the current fine-tune (train-rho 1): pooled difference and the panels whose own S rises (S per panel from the summed squared
errors, as the instrument). Line (#194): primary read 0.15, rho arm - current >= +0.0002 in March and in April, panels >= 7 of 10.
usage: py g231_t1_density_read.py [log=C:/tfb_data/tfb_nn/t1d.log]"""
import sys, json, re, collections
LOG = sys.argv[1] if len(sys.argv) > 1 else "C:/tfb_data/tfb_nn/t1d.log"
S = lambda sv, sq, n: 0.54 * max(0, 1 - (sv / max(n, 1)) ** 0.5 / 25) + 0.46 * max(0, 1 - (sq / max(n, 1)) ** 0.5 / 600)
R = {}
for l in open(LOG, encoding="utf-8", errors="ignore"):
    m = re.match(r"RES seed (\d+) (\d+) rho_train ([\d.]+) (\w+) rho_read ([\d.]+) S ([\d.]+) (\{.*\})", l)
    if m:
        R[(int(m[1]), m[2], float(m[3]), m[4], float(m[5]))] = json.loads(m[7])
seeds = sorted({k[0] for k in R}); arms = sorted({k[2] for k in R if k[3].startswith("ft")})
for sd in seeds:
    for rr in (0.15, 0.3, 1.0):
        for ar in [a for a in arms if a != 1.0]:
            out, ok = [], True
            for mo in ("03", "04"):
                a, b = R.get((sd, mo, ar, "ft32", rr)), R.get((sd, mo, 1.0, "ft32", rr))
                if a is None or b is None:
                    out.append(f"{mo} n/a"); ok = False; continue
                tot = lambda d: S(*[sum(v[i] for v in d.values()) for i in range(3)])
                d = tot(a) - tot(b); up = sum(S(*a[p]) > S(*b[p]) for p in a); out.append(f"{mo} {d:+.5f} (panels up {up}/{len(a)})")
                ok &= d >= 0.0002 and up >= 7
            print(f"seed {sd} read {rr}: rho {ar} - current: " + " | ".join(out) + (f" | line {'met' if ok else 'not met'}" if rr == 0.15 else ""))
    for rr in (0.15, 1.0):
        base = {mo: R.get((sd, mo, 1.0, "base", rr)) for mo in ("03", "04")}
        print(f"seed {sd} read {rr}: no fine-tune " + " | ".join(f"{mo} {S(*[sum(v[i] for v in base[mo].values()) for i in range(3)]):.5f}" for mo in base if base[mo]))
