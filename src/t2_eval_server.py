"""Persistent evaluator: labels/metadata/virtual-LB windows are loaded once; every "done <tag>" in the GPU daemon
log is scored (single NN and blend with the reference) and appended to a CSV. Runs on the CPU beside the GPU job.

usage: python t2_eval_server.py <daemon_log> <ref_spec> <out_csv>
"""
import os, re, sys, time
import t2_eval_oof as E
import t2_eval_surprise as S                                    # onset yardstick for atypical sites

LOG, REF, OUT = sys.argv[1:4]; TW = sys.argv[4] if len(sys.argv) > 4 else "1"   # blend: REF + tag:TW (e.g. REF l127_fast_af:3, TW 7)
E.KEEP.update(t.split(":")[0] for t in REF.split("+"))
done = set()
if os.path.exists(OUT):
    done = {l.split(",")[1] for l in open(OUT) if l.count(",") > 2}
else:
    open(OUT, "w").write("time,tag,spec,eval,eval_on,eval_og,sel,sel_on,sel_og,vlb,vlb_on,vlb_og,sur_plain,sur_w,sur_sub\n")
print("eval server ready", flush=True)
while True:
    txt = open(LOG).read() if os.path.exists(LOG) else ""
    for t in re.findall(r"^done (\S+)", txt, re.M):
        if t in done:
            continue
        for spec in (t, f"{REF}+{t}:{TW}"):
            try:
                e, s, v = E.evaluate(spec)
            except Exception as ex:
                print("EVAL_FAILED", spec, type(ex).__name__, ex, flush=True); continue
            sw = [round(float(x), 4) for x in S.stats(S.per_window(spec))]
            row = [time.strftime("%H:%M:%S"), t, spec, *e, *s, *[round(float(x), 4) for x in v], *sw]
            open(OUT, "a").write(",".join(map(str, row)) + "\n")
            print(E.fmt(spec, (e, s, v)), "| surprise plain/w/sub", sw, flush=True)
        done.add(t)
    if os.path.exists(OUT + ".stop"):                           # the GPU daemon may restart; stop only on request
        print("EVAL_SERVER_END", flush=True); break
    time.sleep(10)
