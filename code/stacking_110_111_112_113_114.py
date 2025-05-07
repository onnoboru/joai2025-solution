# %%
from sklearn.metrics import f1_score
import numpy as np

def lgb_f1_score_weighted(preds, data):
    y_true = data.get_label()
    # LightGBM では preds は [num_class * num_samples] の形で渡ってくる
    y_pred = preds.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'f1_weighted', score, True


# %%
import pandas as pd
import os
import lightgbm

# 実験リスト
exp_list = [
    # "010_film", 
    # "016_film_roberta",
    # "017_film_mean_pool",
    # "018_film_gem",
    # "019_film_gem_gray",
    # "020_film_four_channel",
    # "021_film_four_channel_1out",
    # "022_one_out_100epoch_fixed",
    # "023_film_four_channel_one_out_roberta_fixed",
    # "a01_gem_deberta_anotherseed",
    # "a02_gem_roberta_anotherseed",
    # "a03_film_anotherseed",
    # "a04_multiclass_gem_anotherseed",
    # "024_pooling_none_fixed", 
    "110_film",
    "111_gem",
    "112_oneout_base",
    "113_oneout_roberta",
    "114_oneout_default_pooling",
]

# 基本カラムとターゲットカラム
base_cols = ["MQ8", "MQ5", "min_temp", "max_temp", "dff_tmp", "fold"]
target_cols = ['Mixture', 'NoGas', 'Perfume', 'Smoke']

# 結合するカラムを定義
columns_list = base_cols.copy()
for exp in exp_list:
    for col in target_cols:
        columns_list.append(f"{exp}_{col}")

# クロスバリデーション結果と予測結果を取得
oof_df_path_list = [os.path.join("../outputs", f'{exp}_oof.csv') for exp in exp_list]
test_df_path_list = [os.path.join("../outputs", f'{exp}_test_probs.csv') for exp in exp_list]

# CSVをデータフレームに読み込み
oof_df_list = [pd.read_csv(path) for path in oof_df_path_list]
test_df_list = [pd.read_csv(path) for path in test_df_path_list]

def get_oof_df(df_list, is_train=True):
    """異なるモデルの出力を結合したデータフレームを作成"""
    # 行数を取得
    nrows = len(df_list[0])
    
    # 結合用の空のデータフレームを作成
    result_df = pd.DataFrame(index=range(nrows))
    
    # 基本列のコピー (訓練データから)
    for col in base_cols:
        if col in df_list[0].columns:
            result_df[col] = df_list[0][col].values
    
    # 訓練データにはGas列が含まれる場合がある
    if is_train and "Gas" in df_list[0].columns:
        result_df["Gas"] = df_list[0]["Gas"].values
    
    # 各モデルの予測確率を追加
    for i, exp in enumerate(exp_list):
        for col in target_cols:
            if col in df_list[i].columns:
                result_df[f"{exp}_{col}"] = df_list[i][col].values
    
    return result_df

# 訓練データと予測データを構築
oof_df_train = get_oof_df(oof_df_list, is_train=True)
oof_df_test = get_oof_df(test_df_list, is_train=False)

oof_df_train["another_fold"] = pd.read_csv("../dataset/train_folds_another_seed.csv")["fold"]


test_preds = []
oof_df_train["Mixture"] = 0
oof_df_train["NoGas"] = 0
oof_df_train["Perfume"] = 0
oof_df_train["Smoke"] = 0

fold_list = ["fold", "another_fold"]
# 訓練データと検証データの準備
for fold in fold_list:
    df_X = oof_df_train.drop(columns=["Gas"])
    df_y = oof_df_train["Gas"]
    
    # 検証用の分割
    fold_results = []
    for i in range(5):
        train_X = df_X[df_X[fold] != i].drop(columns=["fold","another_fold", "Mixture", "NoGas", "Perfume", "Smoke"])
        train_y = df_y[df_X[fold] != i]
        valid_X = df_X[df_X[fold] == i].drop(columns=["fold","another_fold", "Mixture", "NoGas", "Perfume", "Smoke"])
        valid_y = df_y[df_X[fold] == i]
        valid_index = df_X[df_X[fold] == i].index
        
        def transform(df):
            for target_col in target_cols:
                df[f"diff_max_{target_col}_MQ5"] = train_X[train_y == target_col]["MQ5"].max() - df["MQ5"]
                df[f"diff_max_{target_col}_MQ8"] = train_X[train_y == target_col]["MQ8"].max() - df["MQ8"]
                df[f"diff_min_{target_col}_MQ5"] = train_X[train_y == target_col]["MQ5"].max() - df["MQ5"]
                df[f"diff_min_{target_col}_MQ8"] = train_X[train_y == target_col]["MQ8"].max() - df["MQ8"]
                
            return df
        
        train_X = transform(train_X)
        valid_X = transform(valid_X)
        
        # ここで学習・予測処理を行う
        # モデル学習のコード...
        
        # テストデータの準備
        test_X = oof_df_test.copy()
        if "fold" in test_X.columns:
            test_X = test_X.drop(columns=["fold"])

        if "another_fold" in test_X.columns:
            test_X = test_X.drop(columns=["another_fold"])
        
        name_to_label = {
                    "Mixture": 0,
                    "NoGas": 1,
                    "Perfume": 2,
                    "Smoke": 3
        }
        train_y = train_y.map(name_to_label)
        valid_y = valid_y.map(name_to_label)

        lgbm_params = {
            "objective": "multiclass",
            "num_class": 4,
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "max_depth": 5,
            "num_leaves": 16,
            "bagging_fraction": 0.2,
            "feature_fraction": 0.2,
            "bagging_seed": 42,
            "seed": 42 if fold == "fold" else 496,
            "verbosity": -1,
            "num_threads": 4,
        }
        # モデル学習のコード...
        print("A")
        model = lightgbm.train(
            params=lgbm_params,
            train_set=lightgbm.Dataset(train_X, label=train_y),
            valid_sets=[lightgbm.Dataset(valid_X, label=valid_y)],
            num_boost_round=20000,
            callbacks=[lightgbm.log_evaluation(100), lightgbm.early_stopping(200)],
        )
        test_X = transform(test_X)
        preds = model.predict(test_X)
        preds_df = pd.DataFrame(preds, columns=["Mixture", "NoGas", "Perfume", "Smoke"])
        test_preds.append(preds)
        preds = model.predict(valid_X)
        # Get feature importances
        # .py ファイルにつき略
        # oof_df_train の該当行に予測結果を保存
        for j, col in enumerate(target_cols):
            oof_df_train.loc[valid_index, col] += preds[:, j] / len(fold_list)
            
    
test_preds = np.mean(test_preds, axis=0)
preds_df = pd.DataFrame(test_preds, columns=["Mixture", "NoGas", "Perfume", "Smoke"])

y_true_numeric = oof_df_train["Gas"].map(name_to_label)
y_pred_labels = oof_df_train[target_cols].idxmax(axis=1)
y_pred_numeric = y_pred_labels.map(name_to_label)
print(f"OOF F1スコア: {f1_score(y_true_numeric, y_pred_numeric, average='weighted')}")
# 出力ファイル名の設定
out_str = "_".join([exp[:3] for exp in exp_list])
sub = pd.read_csv("../dataset/sample_submission.csv")
sub["Gas"] = preds_df.idxmax(axis=1)


# %%
oof_df_train.to_csv(f"../outputs/stacking_{out_str}_oof.csv", index=False)
sub.to_csv(f"../outputs/stacking_{out_str}_submission.csv", index=False)


