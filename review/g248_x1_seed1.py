"""G248 (11/04, ledger #212, the strategy's (B) from panel 6): G141's same-day X1 (arm sd) on the M3 v3 dev windows again with the
fine-tune's seed 1 (G141 / production: seed 0, t2_nn_finetune and G141's Net.finetune default), the same networks c24f_attnglob s0, s1,
the same pools and k. Writes work/t2/x1_arms/parts/sd1__<tag>__<D0>.pkl (restart-safe), log x1_arms/sd1__<tag>.log. Hold-out days
are not computed (G141's groups are dev only). The read is G249 (seed 0 alone against the mean of seeds 0 and 1).
usage: TFB_M3VER=v3 py -B -X utf8 g248_x1_seed1.py <tag>"""
import sys
R_ = r"C:/Users/user/tfb/review/"; s = open(R_ + "g141_x1_arms.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


sub('ks = pool(D0, k, c * 12); nt.finetune(ks); nft += 1', 'ks = pool(D0, k, c * 12); nt.finetune(ks, seed=1); nft += 1')
sub('f = OUT + f"parts/{arm}__{tag}__{D0}.pkl"', 'f = OUT + f"parts/sd1__{tag}__{D0}.pkl"')
sub('log = open(OUT + f"{arm}__{tag}.log", "a")', 'log = open(OUT + f"sd1__{tag}.log", "a")')
sys.argv = [R_ + "g141_x1_arms.py", "fit", "sd", sys.argv[1]]
exec(compile(s, "g141s1", "exec"))
