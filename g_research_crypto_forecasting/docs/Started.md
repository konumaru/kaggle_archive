# コンペティション概要

## 評価指標

- 評価指標は、相関係数

### 評価指標の解釈

時点 t を
t+1 と t+16 の 15 分後の上昇幅（下降幅）を log(Price_t16 / Price_t1) で表し、
これを各通貨で計算し、Portflio の割合(Weigth) を考慮し、評価する

つまり、

Tartget は、時刻 t における将来 15 分後の資産全体の対数変後の変化量を表している、といえる

で今回はこれの相関係数が評価関数になるとのこと j

## 評価データ

- Time Series API で提供
- 約 3 ヶ月分のデータが用意される予定
- `env.iter_test()` を実行してからモデルのロードをしたほうがよい
  - 上記がメモリを利用するため

## データ

- テストデータでロードされるものはデータの型が決まっているので学習時も固定したほうがよさそう

## 検証環境について

前提として、TimeSeriesAPI 内で再学習を行うかという議論がある
一見最新のデータを使ったほうが予測できるのではないか、と思ってしまったが必ずしもそうではなく、上昇下降のパターンが重要な可能性も大いにある

このことから、一旦検証データは提供されている TimeSeriesAPI と同様に 3 ヶ月とする

学習データの期間は一旦、検証データ機関の倍である 6 ヶ月とする

DMM bitcoin によると、１時間ローソク足のとき 60 時間分表示されている。
一旦これを踏襲し、特徴量も 60min のローソク足や専門的な分析手法を利用したい
