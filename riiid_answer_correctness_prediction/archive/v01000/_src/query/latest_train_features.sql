WITH
  t1 AS (
  SELECT
    *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) AS row_num
  FROM
    `repro-lab.temp_konuma_01.train_aggedby_user_id` )
SELECT
  * EXCEPT(row_num)
FROM
  t1
WHERE
  row_num=1
