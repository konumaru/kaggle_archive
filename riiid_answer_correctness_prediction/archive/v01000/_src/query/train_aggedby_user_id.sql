  /*
Dump to
gs://temp_konuma_01/features/train_agged_features_*.csv

Download command:
gsutil -m cp gs://temp_konuma_01/features/train_agged_features_*.csv data/features/
*/
WITH
  train_add_lag_feature AS (
  SELECT
    row_id,
    timestamp,
    user_id,
    answered_correctly,
    content_id,
    content_type_id,
    task_container_id,
    user_answer,
    prior_question_elapsed_time,
    prior_question_had_explanation
    -- lag features
    ,
    lag (answered_correctly,
      1) OVER (PARTITION BY user_id ORDER BY row_id) AS lag_answered_correctly,
    lag (content_id,
      1) OVER (PARTITION BY user_id ORDER BY row_id) AS lag_content_id,
    lag (content_type_id,
      1) OVER (PARTITION BY user_id ORDER BY row_id) AS lag_content_type_id,
    lag (task_container_id,
      1) OVER (PARTITION BY user_id ORDER BY row_id) AS lag_task_container_id
  FROM
    `repro-lab.temp_konuma_01.train`
    -- where
    --   user_id = 2500046
    ),
  train_aggedby_user_id AS (
  SELECT
    * EXCEPT(prior_question_had_explanation),
    ifnull(CAST(prior_question_had_explanation AS int64),
      -1 ) AS prior_question_had_explanation
    -- As sample weight
    ,
    COUNT(row_id) OVER (PARTITION BY user_id ORDER BY row_id) AS past_total_record_count
    -- answered_correctly
    -- SUM and AVG of lag_answered_correctly
    ,
    SUM(CASE
        WHEN lag_answered_correctly=-1 THEN NULL
      ELSE
      lag_answered_correctly
    END
      ) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cumsum_answered_correctly,
    AVG(CASE
        WHEN lag_answered_correctly=-1 THEN NULL
      ELSE
      lag_answered_correctly
    END
      ) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cummean_answered_correctly,
    -- prior_question_had_explanation
    -- SUM, AVG and STD of prior_question_had_explanation
    SUM(CAST(prior_question_had_explanation AS INT64)) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cumsum_prior_question_had_explanation,
    AVG(CAST(prior_question_had_explanation AS INT64)) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cummean_prior_question_had_explanation,
    STDDEV(CAST(prior_question_had_explanation AS INT64)) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cummstd_prior_question_had_explanation
    -- lag_content_type_id
    -- Average and Sum of Quetion and Lecture
    ,
    SUM(1 - lag_content_type_id) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cumsum_question_count,
    AVG(1 - lag_content_type_id) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cummean_question_count,
    SUM(lag_content_type_id) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cumsum_lecture_count,
    AVG(lag_content_type_id) OVER (PARTITION BY user_id ORDER BY row_id) AS past_cummean_lecture_count
  FROM
    train_add_lag_feature )
SELECT
  *,
  1 / past_total_record_count AS reci_past_total_answered_count
FROM
  train_aggedby_user_id
WHERE
  content_type_id = 0
  AND answered_correctly != -1
ORDER BY
  user_id,
  row_id
