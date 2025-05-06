# JOAI2025 学生 3 位 solution
- JOAI2025 での最終 sub を構築するのに必要なコードです
- 解法は提出したスライドを参考にしてください。

# Directory Layout
```
.
├── code      : コードを置きます
├── dataset   : kaggle からダウンロードしたデータや、作成したデータを置きます。
└── outputs   : log や重み、 sub 用の csv ファイルなどを出力します

```

# Requirements
詳しくは requirements.txt を参考にしてください。  
コードの実行は以下の 3 つの環境で行われました。それぞれコード先頭の 3 文字が対応しています。
## kaggle 環境
- GPU : Tesla P100-PCIE-16GB x 1
- 022, 024, a03, a04 で使用
- 実行時から data_dir と output_dir を変更しています。 wandb の config.yaml を参照すると変更箇所が分かります。

## GCP 環境
- GPU : NVIDIA L4 x 1
- a05 で使用

## local 環境
- GPU : RTX4070 Laptop x1
- 010, 018, 023, a01, a02, 及びその他ファイル作成等で使用
- 実行時から joai_toolkit を用いた GridMask の呼び出しを行わないように変更しています。

# how to run
## 全てのライブラリをインストールする
それぞれの環境で requirements.txt を用意しています。  
一例として、ローカルの主なライブラリの状況は以下です。
- torch==2.6.0
- lightning==2.5.0.post0
- scikit-learn==1.6.1
- timm==1.0.13
- transformers==4.51.3
- albumentations==2.0.0
- numpy==1.26.4
- wandb==0.19.5

また、この際 wandb に login することを忘れないでください。
## データをダウンロードする
dataset ディレクトリ直下にデータセットをダウンロードしてください。下図のようなイメージです。  
```
dataset
├── train.csv
├── test.csv
├── sample_submission.csv
└── images

```

## csv を fold に分ける
**以下の (a05_ を除く) 全てのコードは、実際には jupyter notebook 上で実行しましたが、 jupyter notebook を同名 python ファイルに書き出したため、 .py ファイルとして実行できるようになっています。**

今回は、 leak 防止のために事前に fold 列を付けた csv を書き出す、という方針を採用しました。  
```sh
python3 000_add_fold.py
python3 a00_add_fold.py
```

## 逆変換画像を用意する
一部のコードでは、画像に対して近似的な jet の逆変換をしてグレースケールの画像を作っています。  
```sh
python3 b01_temp_reverse_grey.py
```
## DeBERTa, RoBERTa の zero-shot 埋め込み表現を作る
```sh
python3 b02_save_deberta_feature.py
python3 b03_save_roberta_feature.py
```

## 10 個のモデルを学習させる
エラーが出た場合は CFG の data_dir や output_dir を見直してください。
## 023 と 024 はコードにバグを埋め込んでいたためそれの修正をかける

## random seed average で 5 個のファイルを作る
## スタッキングを実行する

# WandB について

# checkpoint について 