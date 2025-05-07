# JOAI2025 選抜枠 3 位 solution
- JOAI2025 での最終 sub を構築するのに必要なコードです
- 解法は提出したスライドを参考にしてください。 JOAI2025_selection_3rd_place_solution.pdf という名前で zip ファイル内にのみ存在しています。

# Directory Layout
```
.
├── code      : コードを置きます
├── dataset   : kaggle からダウンロードしたデータや、作成したデータを置きます。
└── outputs   : log や重み、 sub 用の csv ファイルなどを出力します

```

# Requirements
詳しくはそれぞれの環境の requirements.txt を参考にしてください。  
学習コードの実行は以下の 3 つの環境で行われました。それぞれの学習コード先頭の 3 文字が対応しています。データセットの作成や stacking などは全て local 環境で行われました。
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
- 実行時と比較して joai_toolkit を用いた GridMask の呼び出しを行わないように変更しています。

# how to run
全てのコードにおいて random seed は固定していますが、実行環境等の違いによって 完全に同一の結果が再現されない可能性があります。ご容赦ください。 
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

## 注意点
- **以下の (a05_024_anotherseed と 023_fix と 024_fix を除く) 全てのコードは、実際には jupyter notebook 上で実行しましたが、 jupyter notebook と同名の python ファイルに書き出したため、 .py ファイルとして実行できるようになっています。厳密に再現したい場合は、 code/jupyter_notebook_files にあるファイルを、 code 直下に移動させて実行してください。**
- コードは code/ に移動して実行してください。  
## csv を fold に分ける

今回は、 leak 防止のために事前に fold 列を付けた csv を書き出す、という方針を採用しました。  
```sh
$ python3 000_add_fold.py
$ python3 a00_add_fold.py
```

## 逆変換画像を用意する
一部のコードでは、画像に対して近似的な jet の逆変換をしてグレースケールの画像を作っています。  
```sh
$ python3 b01_temp_reverse_grey.py
```
## DeBERTa, RoBERTa の zero-shot 埋め込み表現を作る
```sh
$ python3 b02_save_deberta_feature.py
$ python3 b03_save_roberta_feature.py
```

## 10 個のモデルを学習させる
エラーが出た場合は CFG の data_dir や output_dir を見直してください。本番は a05 を除いて .ipynb で動かしたため、 .py ファイルは一部動かない可能性があります。
```sh
$ python3 010_film.py
$ python3 018_film_gem.py
$ python3 022_film_four_channel_1out_longepoch.py
$ python3 023_film_four_channel_1out_roberta.py
$ python3 024_pooling_none.py
$ python3 a01_022_anotherseed.py
$ python3 a02_023_anotherseed.py
$ python3 a03_film_anotherseed.py
$ python3 a04_multiclass_gem_anotherseed.py
$ python3 a05_024_anotherseed.py
```
## 023 と 024 はコードにバグを埋め込んでいたためそれの修正をかける
本番でも test データの確率が実際の 5 倍になっているというバグを取り除くためにこのようなファイルを実行しました。この 2 つのファイルは .py ファイルで動かしました。
```sh
$ python3 023_fix.py
$ python3 024_fix.py
```

## random seed average で 5 個のファイルを作る
この 5 つのファイルも .py ファイルで動かしました。
```sh
$ python3 110_film_avg.py
$ python3 111_gem_avg.py
$ python3 112_oneout_base.py
$ python3 113_oneout_roberta.py
$ python3 114_oneout_default_pooling.py
```

## スタッキングを実行する
このコードは本番は .ipynb ファイルで実行しました。
```sh
$ python3 stacking_110_111_112_113_114.py
```

# WandB について
以下のリンクから実行ログが確認できます。提出ファイルの再現に関係ない実験記録も一部含まれていますがご容赦ください。  
リンク : https://wandb.ai/onnoboru/JOAI
# checkpoint について 
全ての checkpoint は以下のリンクから参照できます。いくつかバージョンがある場合は最新のバージョンを参照してください。  
リンク (google drive) : https://drive.google.com/drive/u/3/folders/1j0ahZNpJvJOBl_Ne52d3jEGUzQWwNuyu  
リンク (kaggle dataset) : https://www.kaggle.com/datasets/onnoboru/joai2025-checkpoint

