# %%
import os 
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# %%
dataset_dir = "../dataset/"
train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
train_df['fold'] = -1

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_number, (train_index, val_index) in enumerate(fold.split(X=train_df, y=train_df['Gas'])):
    train_df.loc[val_index, 'fold'] = fold_number
    
train_df.to_csv(os.path.join(dataset_dir, "train_folds.csv"), index=False)



