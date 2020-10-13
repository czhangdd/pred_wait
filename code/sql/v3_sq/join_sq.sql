-- change v1 to v2 to get v2 SQ
create or replace table chizhang.pred_wait_sq_1012_v1 as(
select 
     pw.*
,    sq.score
,    sq.smoothing_score
,    sq.version
,   row_number() over (partition by sq.delivery_id order by abs(timediff('s', cur_time, pw.original_timestamp))) as row_num
from CHIZHANG.pred_wait_final_remove_store_1012 pw
join PRODDB.RAGHAV.FACT_SUPPLY_QUALITY_METRIC_BACKFILL sq
on pw.delivery_id = sq.delivery_id
where sq.version = 1
qualify row_num = 1
)