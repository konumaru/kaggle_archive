# DSB 2019


## Winner Solution
### 1st
url: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127469

- 5-seeds averaged 5-fold, LightGBM
- 重み付き損失を使って学習
    - （多分）userIdの出現回数の逆数
- 特徴量エンジニアリング
  - 要約統計量いろいろ
  - 集計期間を変えた特徴量
    - 全て
    - 5, 12, 48 時間
    - 直前の課題までのデータ
  - イベント間の間隔
    - 平均、最後


### 2nd
url: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127388

- モデリング
  - validation-setでは、1user-1sampleになるようにした
  - StratifiedGroupKFold, 5-fold
  - RandomSeedAveraged(5seeds), LightGBM+CatBoost+NN
- 特徴量エンジニアリング
  - タイトルの列をベクトル化(word2vec)


### 3rd
url: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127891

- 前処理
  - game_session ごとのシーケンスデータを作る
- モデル
  - Transformer
