SELECT
 task_container_id
 , count(task_container_id) / cast(101230332 as INT64) as tatal_task_container_id_count_ratio
 , avg( answered_correctly ) as total_avg_answered_correctly_by_task_container_id
 , stddev( answered_correctly ) as total_std_answered_correctly_by_task_container_id
 , avg( prior_question_elapsed_time ) as total_avg_prior_question_elapsed_time_by_task_container_id
 , stddev( prior_question_elapsed_time ) as total_std_prior_question_elapsed_time_by_task_container_id
 , avg( cast(prior_question_had_explanation as int64) ) as total_avg_prior_question_had_explanation_by_task_container_id
 , stddev( cast(prior_question_had_explanation as int64) ) as total_std_prior_question_had_explanation_by_task_container_id
 , avg(case when user_answer=0 then 1 else 0 end) as total_avg_user_answered_0_by_task_container_id
 , stddev(case when user_answer=0 then 1 else 0 end) as total_std_user_answered_0_by_task_container_id
 , avg(case when user_answer=1 then 1 else 0 end) as total_avg_user_answered_1_by_task_container_id
 , stddev(case when user_answer=1 then 1 else 0 end) as total_std_user_answered_1_by_task_container_id
 , avg(case when user_answer=2 then 1 else 0 end) as total_avg_user_answered_2_by_task_container_id
 , stddev(case when user_answer=2 then 1 else 0 end) as total_std_user_answered_2_by_task_container_id
 , avg(case when user_answer=3 then 1 else 0 end) as total_avg_user_answered_3_by_task_container_id
 , stddev(case when user_answer=3 then 1 else 0 end) as total_std_user_answered_3_by_task_container_id
FROM
  `repro-lab.temp_konuma_01.train`
GROUP BY
  task_container_id
