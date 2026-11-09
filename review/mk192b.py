"""11/03 ledger #192, the fair test (report section 9, not shipped): review/t1_unet_base192.py = tfb_a8/src/t1_unet_pre.py with
U_PRE_DAYS=none adding no test-month cells (February stays in training), so the local baseline differs from the seed-4 pretrained
run only by the test months' odd-day cells (same b64, 45 epochs, seed, machine). By substitution only. usage: py mk192b.py"""
import hashlib
s = open(r"C:/Users/user/tfb_a8/src/t1_unet_pre.py", encoding="utf-8").read()
a = 'if PRE_DAYS == "all" or i % 2 == 0]'
assert s.count(a) == 1
s = s.replace(a, 'if PRE_DAYS != "none" and (PRE_DAYS == "all" or i % 2 == 0)]   # 11/03 #192 fair test: none = no test-month cells')
s = s.replace("# TFB (11/03, ledger #192): t1_unet_train.py", "# TFB (11/03, ledger #192 fair test): U_PRE_DAYS=none adds no test-month cells.\n# TFB (11/03, ledger #192): t1_unet_train.py", 1)
open(r"C:/Users/user/tfb/review/t1_unet_base192.py", "w", encoding="utf-8").write(s)
print("written", hashlib.sha256(s.encode()).hexdigest()[:8])
