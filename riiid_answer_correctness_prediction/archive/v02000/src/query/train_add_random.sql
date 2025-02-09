WITH add_random_value as (
  SELECT
    user_id
    , rand() as random_value
  FROM `repro-lab.temp_konuma_01.train`
  GROUP BY user_id
)

SELECT
  train.*
  , rand.random_value
FROM `repro-lab.temp_konuma_01.train` as train
  LEFT JOIN add_random_value as rand
    ON rand.user_id = train.user_id
ORDER BY
  train.user_id,
  train.row_id
