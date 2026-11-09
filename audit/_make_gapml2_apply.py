"""Write src/t1_gapml2_apply.py = t1_gapml_apply.py with the v2 training set (core windows + 9k extra windows at a
0.35 sampling rate) and 255 leaves / lr 0.08; saves gapml2_*_final.txt and gapml2_fill.parquet."""
src = open("C:/Users/user/tfb/src/t1_gapml_apply.py", encoding="utf-8").read()
old_win = '''m = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c in zip(Z["panel"][m], Z["t"][m], Z["cond"][m]):
    wins.setdefault((str(p), int(t)), str(c))
del Z
rng = np.random.default_rng(11); TR = []; names = None'''
new_win = '''core = Z["is_eval"] | Z["is_sel"]
wins = {}
for p, t, c, k in zip(Z["panel"], Z["t"], Z["cond"], core):
    key = (str(p), int(t))
    if key not in wins or k:
        wins[key] = (str(c), bool(k))
del Z
rng = np.random.default_rng(21); TR = []; names = None
extra = [k for k, v in wins.items() if not v[1]]
pick = set(map(tuple, np.array(extra, dtype=object)[rng.choice(len(extra), min(9000, len(extra)), replace=False)].tolist()))'''
assert old_win in src; src = src.replace(old_win, new_win)
old_loop = '''    for t, c in sorted((t, c) for (pp, t), c in wins.items() if pp == p):
        if t + G + 31 > P.T:
            continue'''
new_loop = '''    for (pp, t), (c, is_core) in sorted(wins.items()):
        if pp != p or t + G + 31 > P.T or (not is_core and (pp, t) not in pick):
            continue'''
assert old_loop in src; src = src.replace(old_loop, new_loop)
old_pk = 'pk = np.where(near, 0.5, 0.08); keep = ok & (rng.random(len(ok)) < pk)'
assert old_pk in src; src = src.replace(old_pk, 'pk = np.where(near, 0.5, 0.08) * (1.0 if is_core else 0.35); keep = ok & (rng.random(len(ok)) < pk)')
old_prm = 'prm = dict(PARAMS, learning_rate=lr); models = {}'
assert old_prm in src; src = src.replace(old_prm, 'prm = dict(PARAMS, num_leaves=255, min_data_in_leaf=300, learning_rate=lr); models = {}')
src = src.replace('Xtr = np.vstack([a[0] for a in TR])', 'Xtr = np.vstack([a[0].astype(np.float32) for a in TR])')
src = src.replace('W + f"t1/gapml_{ch}_final.txt"', 'W + f"t1/gapml2_{ch}_final.txt"').replace('W + "t1/gapml_fill.parquet"', 'W + "t1/gapml2_fill.parquet"')
src = src.replace('"""Train the final blank-cell model', '"""v2 (audit b5: + 9k extra windows, 255 leaves; Feb blank RMSE v 5.53->5.23, q 62.6->60.5).\nTrain the final blank-cell model', 1)
open("C:/Users/user/tfb/src/t1_gapml2_apply.py", "w", encoding="utf-8").write(src)
print("written")
