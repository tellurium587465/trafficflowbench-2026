"""G174 (11/01, ledger #124): the same-day X1 of G141 (arm sd) on the hold-out ongoing windows, probabilities only. Nothing here reads a score: the labels enter only through the fine-tune, as they do for every X1 (windows with T' + 6 <= c), and no IoU is
computed. The read is one, on 11/4, by g175_bundle_read.py after #63. Writes work/t2/x1_arms/parts_holdout/sd__<tag>__<D0>.pkl per hold-out day, restart-safe.
usage: TFB_M3VER=v3 py -B -X utf8 g174_sd_holdout.py <tag>"""
import sys
R_ = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True; tag = sys.argv[1]
exec(compile(open(R_ + "g141_x1_arms.py", encoding="utf-8").read().split("MODE = sys.argv[1]")[0], "g141", "exec"))
O2 = OUT + "parts_holdout/"; os.makedirs(O2, exist_ok=True); HOLD = [m for m in META if m["cond"] == "queue_ongoing" and hold(m["day"])]; G_ = {}
for m in HOLD:
    G_.setdefault((m["day"], m["t"] // 12), []).append(m["window_id"])
print("hold-out ongoing windows", len(HOLD), "groups", len(G_), flush=True); nt = Net(tag); log = open(OUT + f"sd_hold__{tag}.log", "a"); t0 = time.time(); nft = 0
for D0 in sorted({d for d, _ in G_}):
    f = O2 + f"sd__{tag}__{D0}.pkl"
    if os.path.exists(f):
        continue
    out = {}
    for (d, c), wids in sorted(G_.items(), key=str):
        if d != D0:
            continue
        for k in range(0, min(6, D0) + 1):
            nt.finetune(pool(D0, k, c * 12)); nft += 1; pr = nt.predict(wids); out.update({(w, k): pr[w] for w in wids})
    pickle.dump(out, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f); log.write(f"day {D0 + 1}: {nft} fine-tunes, {time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}\n"); log.flush()
log.write(f"ALLDONE {time.time() - t0:.0f}s\n"); log.close()
