SELECT
  * except(
  row_id
  , timestamp
  , user_answer
  , random_value
  )
FROM
  `repro-lab.temp_konuma_01.train_join_userId_contentId`
where
  random_value < 0.1
