create or replace table chizhang.pred_wait_sq_r2c_1013_v2 as(
select
    t1.*
 ,  ai.r2c_duration
from chizhang.pred_wait_sq_1012_v2 t1
join SEGMENT_EVENTS.SERVER_EVENTS_PRODUCTION.DEEP_RED_ASSIGNMENT_INFO ai
on  t1.delivery_id = ai.delivery_id
and t1.ASSIGNMENT_RUN_ID = ai.assignment_run_id
)