WITH
  train_groupby_content_id AS (
  SELECT
    content_id,
    AVG(CASE
        WHEN answered_correctly != -1 THEN answered_correctly
      ELSE
      NULL
    END
      ) AS mean_answered_correctly_by_content_id,
    COUNT(DISTINCT task_container_id) AS nunique_task_container_id
    -- user_answer の 割合
    ,
    AVG(CASE
        WHEN user_answer=0 THEN 1
      ELSE
      0
    END
      ) AS mean_user_answered_0,
    AVG(CASE
        WHEN user_answer=1 THEN 1
      ELSE
      0
    END
      ) AS mean_user_answered_1,
    AVG(CASE
        WHEN user_answer=2 THEN 1
      ELSE
      0
    END
      ) AS mean_user_answered_2,
    AVG(CASE
        WHEN user_answer=3 THEN 1
      ELSE
      0
    END
      ) AS mean_user_answered_3
  FROM
    `repro-lab.temp_konuma_01.train`
  GROUP BY
    content_id )
SELECT
  t1.*
  , contet_data.* except(content_id)
FROM
  train_groupby_content_id as t1
  LEFT JOIN `repro-lab.temp_konuma_01.content_metadata` as contet_data
    ON t1.content_id = contet_data.content_id
