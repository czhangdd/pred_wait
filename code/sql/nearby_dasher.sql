-- one month data --
set exp_start = '2020-05-01';
set exp_end = '2020-06-06'; 

------------------------------------------
-------------- geo table ----------------
------------------------------------------
drop table if exists four_weeks_geo_candidate_shifts;
create table four_weeks_geo_candidate_shifts as (
select 
    EVENT_CREATED_AT
    , DELIVERY_ID
    , SHIFT_LAT
    , SHIFT_LNG
    , case when IS_SHIFT_BUSY='false' then 0
         when IS_SHIFT_BUSY='true' then 1
         else -1 end as IS_SHIFT_BUSY
    , SP_ID
    , ASSIGNMENT_RUN_ID
    , SHIFT_ID
  from Ingest_deepred_server_events_prod.event_deep_red_initial_shift_candidates
  where EVENT_CREATED_AT between $exp_start and $exp_end
  and IS_PROD='True'
);

------------------------------------------
-------------- wait table ----------------
------------------------------------------
drop table if exists wait_table;
create table wait_table as(
-- get delivery_id, first_assignment_time (min assigned time)
with cds_first_assignment_ids as(
select
    cds.delivery_id, 
    min(ORIGINAL_TIMESTAMP) as first_assignment_time_utc
from segment_events.server_events_production.deep_red_munkres_considered_delivery_stats cds
    where to_timestamp_ntz(cds.TIMESTAMP) between $exp_start and $exp_end
group by delivery_id
),

-- self join to get useful columns from cds and oca
cds_real_time_feat as(
select
    cds.ORIGINAL_TIMESTAMP as ORIGINAL_TIMESTAMP,
    cds.delivery_id as delivery_id,
    cds_first_assignment_ids.first_assignment_time_utc as first_assignment_time_utc,
    cds.ACCEPTANCE_RATE_ON_CHECK_IN,
    to_timestamp_ntz(cds.order_ready_time) as order_ready_time_utc,
    cds.ASSIGNMENT_RUN_ID,
    cds.flf as flf
from
    segment_events.server_events_production.deep_red_munkres_considered_delivery_stats cds
join cds_first_assignment_ids
    on cds_first_assignment_ids.delivery_id = cds.delivery_id
    and cds_first_assignment_ids.first_assignment_time_utc = cds.ORIGINAL_TIMESTAMP
WHERE
    cds.is_prod = 'True'
),


-- join real-time feature with DD to get hist store level features
wait as(
select 
    -- real-time feat from cds
    cds_real_time_feat.*,
    -- feat from dd
    dd.created_at,
    -- timediff(s, dd.created_at, cds_real_time_feat.order_ready_time_utc) as pred_horizon,
    timediff(s, cds_real_time_feat.ORIGINAL_TIMESTAMP, cds_real_time_feat.order_ready_time_utc) as pred_horizon,
    dd.D2R_DURATION,
    dd.subtotal as subtotal,
    dd.tip as tip,
    dd.store_id,
    dd.NUM_ASSIGNS,
    dd.pickup_address_id,
    madd.EXT_POINT_LAT, 
    madd.EXT_POINT_LONG,
    timediff(s, dd.dasher_at_store_time, cds_real_time_feat.order_ready_time_utc) as wait_before_ready_time
from PUBLIC.DIMENSION_DELIVERIES dd
JOIN cds_real_time_feat
    ON dd.delivery_id = cds_real_time_feat.delivery_id
  -- join with maindb_address to get store lat and long
join maindb_address madd
    on dd.pickup_address_id = madd.id

WHERE dd.IS_ASAP = True
  AND dd.business_id not in (185179, 12284, 12970, 10171, 9144, 8159, 206181, 1855, 115, 11671, 1798, 25117, 2764, 53152, 491, 953, 13681, 1431, 15444, 4477, 139197, 5579, 58164,42492, 5235, 1673, 3612, 3673, 8312, 47852, 12860, 1133, 6731, 3720, 4815, 7376, 1125, 12264)
  AND dd.is_filtered = TRUE
)
,
-- calculate average assignments for each store
avg_store_feat as (
select 
  store_id, 
  avg(subtotal) as avg_subtotal,
  avg(tip) as avg_tip,
  avg(NUM_ASSIGNS) as avg_num_assigns,
  avg(D2R_DURATION) as avg_d2r_duration
from wait
group by store_id
),

wait_avg_store_assign as(
select 
  wait.*,
  avg_store_feat.avg_num_assigns,
  avg_store_feat.avg_subtotal,
  avg_store_feat.avg_tip,
  avg_store_feat.avg_d2r_duration
from wait 
left join avg_store_feat 
  on avg_store_feat.store_id = wait.store_id
)


select * from wait_avg_store_assign sample(20)
  );


---------------------------------------------
--------------- wait_geo table --------------
---------------------------------------------
drop table if exists CHIZHANG.wait_geo_table;
create table CHIZHANG.wait_geo_table as (
-- get distance from one shift to one store, for each assignment_run_id
with delivery_distance as (
select
       geo.delivery_id,
       geo.assignment_run_id,
       geo.shift_id,       
       geo.IS_SHIFT_BUSY,
       geo.shift_lat,
       geo.shift_lng,
       wait_table.EXT_POINT_LAT,
       wait_table.EXT_POINT_LONG,
       haversine(shift_lat, SHIFT_LNG, EXT_POINT_LAT, EXT_POINT_LONG) as distance,
       case when distance<2 then 1 else 0 end as is_nearby,
       case when (is_nearby=1 and IS_SHIFT_BUSY=0) then 'dasher_nearby_idle' 
            when IS_SHIFT_BUSY=1 then 'dasher_busy' end as dasher_status

from  wait_table
join four_weeks_geo_candidate_shifts geo
  on wait_table.delivery_id = geo.delivery_id and
     wait_table.assignment_run_id = geo.assignment_run_id
)
,

delivery_num_nearby as (
select
    delivery_id,
    assignment_run_id,
    sum(case when dasher_status = 'dasher_nearby_idle' then 1 else 0 end) as num_nearby_idle,
    sum(case when dasher_status = 'dasher_busy' then 1 else 0 end) as num_busy
    
from delivery_distance
group by 1, 2
)

select wait_table.*,
       dnn.num_nearby_idle,
       dnn.num_busy
from wait_table
join delivery_num_nearby dnn
  on dnn.delivery_id = wait_table.delivery_id and
     dnn.assignment_run_id = wait_table.assignment_run_id
-- limit 10
)