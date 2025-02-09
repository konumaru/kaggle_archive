SELECT
  * except(random_value)
FROM `repro-lab.temp_konuma_01.train_add_random` as train
WHERE
  random_value < 0.1
ORDER BY
  train.user_id,
  train.row_id
