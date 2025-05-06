import pandas as pd

df = pd.read_csv("../outputs/024_pooling_none_oof.csv")
target_cols = ['Mixture', 'NoGas', 'Perfume', 'Smoke']
for col in target_cols:
    df[col] /= 5
df.to_csv("../outputs/024_pooling_none_fixed_oof.csv", index=False)
df = pd.read_csv("../outputs/024_pooling_none_test_probs.csv")
target_cols = ['Mixture', 'NoGas', 'Perfume', 'Smoke']
for col in target_cols:
    df[col] /= 5
df.to_csv("../outputs/024_pooling_none_fixed_test_probs.csv", index=False)
