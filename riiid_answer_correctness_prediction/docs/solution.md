# Solution

## Feature Engineering
前処理は基本的にBigqueryで行う

- raw_data の情報
- questioin, lecture のコンテンツ情報
- user_id ごとの平均正解率、平均セッション時間、平均問題回答回数、現在と1~3こ前のtimestampの差分
- content_id ごとの平均正解率、平均セッション時間、平均問題回答回数、ユニークユーザー数
- user_id-contetn_id ごとの count


## Sampling
- user_idが分断されないようにサンプリングする

## Model
- XGBoost, sample_weight=1/uu_record_count
- Group 5-Fold, group=user_id
