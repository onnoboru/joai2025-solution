import pandas as pd

exp = "114_oneout_default_pooling"
exp_1 = "a05_default_pooling_anotherseed"
exp_2 = "024_pooling_none"

df1 = pd.read_csv(f"../outputs/{exp_1}_oof.csv")
df2 = pd.read_csv(f"../outputs/{exp_2}_oof.csv")
target_cols=["Mixture","NoGas","Perfume","Smoke"]
df = df1.copy()
df[target_cols] += df2[target_cols]
df[target_cols] /= 2
df.to_csv(f"../outputs/{exp}_oof.csv", index=False)

df1 = pd.read_csv(f"../outputs/{exp_1}_test_probs.csv")
df2 = pd.read_csv(f"../outputs/{exp_2}_test_probs.csv")
target_cols=["Mixture","NoGas","Perfume","Smoke"]
df = df1.copy()
df[target_cols] += df2[target_cols]
df[target_cols] /= 2
df.to_csv(f"../outputs/{exp}_test_probs.csv", index=False)
