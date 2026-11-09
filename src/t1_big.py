import sys; from t1_exp import run; from common import PANELS
run(3,int(sys.argv[1]),PANELS,sys.argv[2],params=dict(learning_rate=float(sys.argv[3])),rounds=6000,save=True,cong_boost=10.0)
