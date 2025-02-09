WITH agg_train_by_content_id as (
SELECT
 content_id
 , count(content_id) / cast(101230332 as INT64) as tatal_content_id_count_ratio
 , avg( answered_correctly ) as total_avg_answered_correctly_by_content_id
 , stddev( answered_correctly ) as total_std_answered_correctly_by_content_id
 , avg( prior_question_elapsed_time ) as total_avg_prior_question_elapsed_time_by_content_id
 , stddev( prior_question_elapsed_time ) as total_std_prior_question_elapsed_time_by_content_id
 , avg( cast(prior_question_had_explanation as int64) ) as total_avg_prior_question_had_explanation_by_content_id
 , stddev( cast(prior_question_had_explanation as int64) ) as total_std_prior_question_had_explanation_by_content_id
 , avg(case when user_answer=0 then 1 else 0 end) as total_avg_user_answered_0_by_content_id
 , stddev(case when user_answer=0 then 1 else 0 end) as total_std_user_answered_0_by_content_id
 , avg(case when user_answer=1 then 1 else 0 end) as total_avg_user_answered_1_by_content_id
 , stddev(case when user_answer=1 then 1 else 0 end) as total_std_user_answered_1_by_content_id
 , avg(case when user_answer=2 then 1 else 0 end) as total_avg_user_answered_2_by_content_id
 , stddev(case when user_answer=2 then 1 else 0 end) as total_std_user_answered_2_by_content_id
 , avg(case when user_answer=3 then 1 else 0 end) as total_avg_user_answered_3_by_content_id
 , stddev(case when user_answer=3 then 1 else 0 end) as total_std_user_answered_3_by_content_id
 , avg( timestamp ) as total_avg_timestamp_by_content_id
 , stddev( timestamp ) as total_std_timestamp_by_content_id
 --, count( distinct task_container_id ) as nunique_task_container_id_by_content_id
 , min( task_container_id ) as total_min_task_container_id_by_content_id
 , max( task_container_id ) as total_max_task_container_id_by_content_id
 , avg( task_container_id ) as total_avg_task_container_id_by_content_id
 , fhoffa.x.median(ARRAY_AGG(task_container_id)) AS total_mid_task_container_id_by_content_id
 , stddev( task_container_id ) as total_std_task_container_id_by_content_id
FROM
  `repro-lab.temp_konuma_01.train`
where
  content_type_id = 0
GROUP BY
  content_id
),
agg_train_by_part as (
SELECT
 questions.part
 , count(questions.part) / cast(101230332 as INT64) as tatal_part_count_ratio
 , avg( answered_correctly ) as total_avg_answered_correctly_by_part
 , stddev( answered_correctly ) as total_std_answered_correctly_by_part
 , avg( prior_question_elapsed_time ) as total_avg_prior_question_elapsed_time_by_part
 , stddev( prior_question_elapsed_time ) as total_std_prior_question_elapsed_time_by_part
 , avg( cast(prior_question_had_explanation as int64) ) as total_avg_prior_question_had_explanation_by_part
 , stddev( cast(prior_question_had_explanation as int64) ) as total_std_prior_question_had_explanation_by_part
 --, count( distinct task_container_id ) as nunique_task_container_id_by_part
 , avg(case when user_answer=0 then 1 else 0 end) as total_avg_user_answered_0_by_part
 , stddev(case when user_answer=0 then 1 else 0 end) as total_std_user_answered_0_by_part
 , avg(case when user_answer=1 then 1 else 0 end) as total_avg_user_answered_1_by_part
 , stddev(case when user_answer=1 then 1 else 0 end) as total_std_user_answered_1_by_part
 , avg(case when user_answer=2 then 1 else 0 end) as total_avg_user_answered_2_by_part
 , stddev(case when user_answer=2 then 1 else 0 end) as total_std_user_answered_2_by_part
 , avg(case when user_answer=3 then 1 else 0 end) as total_avg_user_answered_3_by_part
 , stddev(case when user_answer=3 then 1 else 0 end) as total_std_user_answered_3_by_part
FROM
  `repro-lab.temp_konuma_01.train` as t1
     left join `repro-lab.temp_konuma_01.questions` as questions
       on t1.content_id = questions.question_id
where
  content_type_id = 0
GROUP BY
  part
)
, concat_agg_train_and_questions as (
select
  t1.*
  , questions.* except(question_id)
from agg_train_by_content_id as t1
  left join `repro-lab.temp_konuma_01.questions` as questions
    on t1.content_id = questions.question_id
)


select
  t1.*
  , t_part.* except(part)
from concat_agg_train_and_questions as t1
  left join agg_train_by_part as t_part
    on t1.part = t_part.part




