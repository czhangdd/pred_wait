-- sp level switchback --
set exp_start = '2020-07-10';
set exp_end = '2020-07-16'; 

-- -- store level switchback --
-- set exp_start = '2020-06-20';
-- set exp_end = '2020-07-15'; 

------------------------------------------------
-------------- geo table (temp) ----------------
-------------- used by nearby dasher table -----
------------------------------------------------
create or replace table chizhang.geo_candidate_shifts as (
select 
    EVENT_CREATED_AT
    , DELIVERY_ID
    , SHIFT_LAT
    , SHIFT_LNG
    , case when IS_SHIFT_BUSY='false' then 0
           when IS_SHIFT_BUSY='true' then 1
           end as IS_SHIFT_BUSY
    , SP_ID
    , ASSIGNMENT_RUN_ID
    , SHIFT_ID
  from Ingest_deepred_server_events_prod.event_deep_red_initial_shift_candidates
  where EVENT_CREATED_AT between $exp_start and $exp_end
  and IS_PROD='True'
//  limit 10
);


------------------------------------------------
-------------- est_d2r table (temp) ------------
-------------- used by nearby dasher table -----
------------------------------------------------
create or replace table chizhang.estimated_d2r as(
  select 
      EVENT_CREATED_AT
  ,   delivery_id
  ,   shift_id
  ,   ASSIGNMENT_RUN_ID
  ,   STORE_ARRIVAL_TIME
  ,   split(ASSIGNMENT_RUN_ID, '_')[2] / 1000  as assign_time
  ,   STORE_ARRIVAL_TIME - assign_time as est_d2r
  from PRODDB.INGEST_DEEPRED_SERVER_EVENTS_PROD.EVENT_DEEP_RED_OPTIMIZER_CANDIDATE_ASSIGNMENTS oca 
  where
      oca.IS_PROD='True'
  and EVENT_CREATED_AT between $exp_start and $exp_end
);


---------------------------------------------------
-------------- nearby dasher table -----------------
---------------------------------------------------

create or replace table chizhang.nearby_dasher as(
-- get distance from one shift/dasher to one store, for each assignment_run_id
with delivery_distance as (
select
       geo.delivery_id,
       geo.assignment_run_id,
       geo.shift_id,       
       geo.IS_SHIFT_BUSY,
       geo.shift_lat,
       geo.shift_lng,
       wt.EXT_POINT_LAT, --store lat
       wt.EXT_POINT_LONG,--store long
       haversine(shift_lat, SHIFT_LNG, EXT_POINT_LAT, EXT_POINT_LONG) as distance,
       case when distance<2 then 1 else 0 end as is_nearby,
       case when (is_nearby=1 and IS_SHIFT_BUSY=0) then 'dasher_nearby_idle' 
            when IS_SHIFT_BUSY=1 then 'dasher_busy' end as dasher_status,
       ed2r.est_d2r as est_d2r --dasher d2r
from  chizhang.wait_table wt
join chizhang.geo_candidate_shifts geo
  on wt.delivery_id = geo.delivery_id and
     wt.assignment_run_id = geo.assignment_run_id
join chizhang.estimated_d2r ed2r
  on  ed2r.delivery_id = geo.delivery_id
  and ed2r.shift_id = geo.shift_id
  and ed2r.assignment_run_id = geo.assignment_run_id
)
,

nth_close_dasher_d2r_all as(
select
   delivery_id
,  assignment_run_id
,  count(*) over (partition by delivery_id, assignment_run_id) as num_nearby_idle
,  nth_value(est_d2r, 1) over(partition by delivery_id, assignment_run_id order by est_d2r) as est_d2r_first
,  nth_value(est_d2r, 2) over(partition by delivery_id, assignment_run_id order by est_d2r) as est_d2r_second
,  nth_value(est_d2r, 3) over(partition by delivery_id, assignment_run_id order by est_d2r) as est_d2r_third
from delivery_distance
  where dasher_status='dasher_nearby_idle' --get d2r for idle dasher only, as num_busy is not an important feat
)
  select DISTINCT * from nth_close_dasher_d2r_all

);


--------------------------------------------------------
------------ new optimal dasher arrival (temp) ---------
------------ from mx pickup prep time exp data ---------
------------ used by wait table ------------------------
--------------------------------------------------------
create or replace table chizhang.optimal_dasher_arrival_table as (
with mx_pickup_exp_results as (
    -- select 'store' as exp_unit, store.* from chizhang.mx_pickup_exp_data_store store 
    -- union all
    select 'sp' as exp_unit, sp.* from chizhang.mx_pickup_exp_data_sp sp
)
select delivery_id, 
    created_at, 
    create_2_ready, 
    create_2_ready - 165 as create_to_arrive
from mx_pickup_exp_results where result = 'treatment'
and create_2_ready is not null
--  limit 10
);


------------------------------------------
-------------- wait table ----------------
------------------------------------------
create or replace table chizhang.wait_table as(

-- get delivery_id, first_assignment_time (min assigned time)
with cds_first_assignment_ids as(
select
    cds.delivery_id, 
    min(ORIGINAL_TIMESTAMP) as first_assignment_time_utc
from segment_events.server_events_production.deep_red_munkres_considered_delivery_stats cds
    where to_timestamp_ntz(cds.TIMESTAMP) between $exp_start and $exp_end
group by delivery_id
//limit 10
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
    dd.actual_pickup_time,
    dd.dasher_at_store_time,
//    dasher_arrives_25m as dasher_at_store_25m,
    timediff(s, cds_real_time_feat.ORIGINAL_TIMESTAMP, cds_real_time_feat.order_ready_time_utc) as pred_horizon,
    dd.D2R_DURATION,
    dd.subtotal as subtotal,
    dd.tip as tip,
    dd.store_id,
    dd.NUM_ASSIGNS,
    dd.pickup_address_id,
//    odat.create_to_arrive,
    madd.EXT_POINT_LAT, 
    madd.EXT_POINT_LONG,
    timediff(s, dd.dasher_at_store_time, cds_real_time_feat.order_ready_time_utc) as wait_before_ready_time
from PUBLIC.DIMENSION_DELIVERIES dd
JOIN cds_real_time_feat
    ON dd.delivery_id = cds_real_time_feat.delivery_id
  -- join with maindb_address to get store lat and long
join geo_intelligence.public.maindb_address madd
    on dd.pickup_address_id = madd.id
//join joeharkman.fact_geofence_arrival fga
//    on dd.delivery_id = fga.delivery_id
join chizhang.optimal_dasher_arrival_table odat
    on odat.delivery_id = dd.delivery_id
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

select * from wait_avg_store_assign
-- select * from wait_avg_store_assign sample(20)
--  limit 10
  );


-----------------------------------
----------- final table -----------
-----------------------------------
create or replace table CHIZHANG.pred_wait_final as (
select wt.*,
       dnn.num_nearby_idle,
       dnn.est_d2r_first,
       dnn.est_d2r_second,
       dnn.est_d2r_third
from chizhang.wait_table wt
join chizhang.nearby_dasher dnn
  on dnn.delivery_id = wt.delivery_id and
     dnn.assignment_run_id = wt.assignment_run_id
);

grant select on CHIZHANG.pred_wait_final to role read_only_users;

select count(*) from CHIZHANG.pred_wait_final;