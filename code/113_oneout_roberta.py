import pandas as pd

exp = "113_oneout_roberta"
exp_1 = "a02_gem_roberta_anotherseed"
exp_2 = "023_film_four_channel_one_out_roberta_fixed"

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
