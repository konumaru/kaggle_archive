  /*
-- train_agged_features の 10% をサンプリングする。
-- ただし、user_idは分断されないようにする

Dump to
gs://temp_konuma_01/train_agged_features_sample/train_agged_features_sample_*.csv

Download Command:
gsutil -m cp gs://temp_konuma_01/train_agged_features_sample/train_agged_features_sample_*.csv data/features/
*/
WITH
  t_apply_rand_val AS (
  SELECT
    user_id,
    rand() AS random_value
  FROM
    `repro-lab.temp_konuma_01.train_aggedby_user_id`
  GROUP BY
    user_id )
SELECT
  t_raw.*,
  t_content.* EXCEPT(content_id),
  t_rand.random_value
FROM
  `repro-lab.temp_konuma_01.train_aggedby_user_id` t_raw
LEFT JOIN
  t_apply_rand_val AS t_rand
ON
  t_raw.user_id = t_rand.user_id
LEFT JOIN
  `repro-lab.temp_konuma_01.train_groupby_content_id` AS t_content
ON
  t_content.content_id = t_raw.content_id
ORDER BY
  t_raw.user_id,
  t_raw.row_id
