import pandas as pd

df = pd.read_csv("../outputs/023_film_four_channel_one_out_roberta_oof.csv")

target_cols = ['Mixture', 'NoGas', 'Perfume', 'Smoke']

df.to_csv("../outputs/023_film_four_channel_one_out_roberta_fixed_oof.csv", index=False)

df = pd.read_csv("../outputs/023_film_four_channel_one_out_roberta_test_probs.csv")

df[target_cols] /= 5

df.to_csv("../outputs/023_film_four_channel_one_out_roberta_fixed_test_probs.csv", index=False)