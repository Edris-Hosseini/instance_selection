import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from src.method import idea
## method Param
K=5
frac=0.2
eval_model='DT'
## Read dataset
datasets=r'./datasets/Appendicitis.csv'
dt=pd.read_csv(datasets,header=None).to_numpy()

X_train, X_test = train_test_split(dt, test_size=0.2, shuffle=True)
X_train = np.array(np.insert(X_train, 0, range(len(X_train)), axis=1))

print("RUNNING METHOD")
st = time()
ob, rm_idx = idea().i_s(X_train, K_neighbors=K, frac=frac).i_s_methods(1)  ##params
# dp, rm_idx = idea().i_s(X_train, K_neighbors=K, frac=frac).i_s_methods(2)  ##params
end = time() - st
print("DONE")

print("TIME",end)

rm_idx = np.array(rm_idx)
df = pd.DataFrame(X_train).set_index(0)
df.drop(rm_idx, axis=0, inplace=True)
final_data = df.to_numpy()
rr = ((len(rm_idx) * 100) / len(X_train))

ob.eval_(eval_model, X_test, final_data)
print("ACC : ",ob.get_score())
print("F1 : ",ob.get_f1())
