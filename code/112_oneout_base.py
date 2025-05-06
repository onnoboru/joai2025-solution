import pandas as pd

exp = "112_oneout_base"
exp_1 = "a01_gem_deberta_anotherseed"
exp_2 = "022_one_out_100epoch_fixed"

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
