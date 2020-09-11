set experiment_name = 'use_mx_experimental_prep_time';
set experiment_version = 10;
set start_time = '2020-07-09 22:00:00';
set end_time = '2020-07-17 16:00:00';
set outage_start = '2020-03-20 15:30';
set outage_end = '2020-03-20 20:00';

-- create table joeharkman.temp_exp_results_sp_v2 as
create or replace table chizhang.mx_pickup_exp_data_sp as

with result as 
(
select 
assignment_run_id
, region_id as sp_id
, unit_id
, max(result) as result
, max(se.received_at) as timing  
from segment_events.server_events_production.switchback_exposure se 
where experiment_name = $experiment_name
and received_at between $start_time and $end_time
and received_at not between $outage_start and $outage_end
and experiment_version=$experiment_version
group by 1, 2, 3
having count(distinct result) = 1 
), 

last_assign as 
(
select 
ai.DELIVERY_ID, ai.SHIFT_ID
, max(ai.received_at) as max_received_at
from SEGMENT_EVENTS.SERVER_EVENTS_PRODUCTION.DEEP_RED_ASSIGNMENT_INFO ai
where ai.ORIGINAL_TIMESTAMP between $start_time and $end_time
group by 1, 2
), 

assign as 
(
select 
distinct 
ai.delivery_id
, ai.shift_id
, se.unit_id
, se.sp_id
, se.result
, se.timing
, ai.ROUTE_IS_SAME_BLOCK_BATCH
, ai.FOOD_READY_ESTIMATION_SOURCE as order_READY_ESTIMATION_SOURCE
, to_timestamp_ntz(ai.FOOD_READY_TIME) as order_ready_time
, ai.ROUTE_IS_SAME_STORE_BATCH
, ai.FLF as flf_assign
, ai.IDEAL_FLF as ideal_flf_assign
from SEGMENT_EVENTS.SERVER_EVENTS_PRODUCTION.DEEP_RED_ASSIGNMENT_INFO ai
JOIN result se on ai.assignment_run_id=se.assignment_run_id and se.sp_id=ai.SP_ID
join last_assign fa on ai.received_at=fa.max_received_at and fa.delivery_id=ai.delivery_id and fa.shift_id=ai.SHIFT_ID
where ai.RECEIVED_AT between $start_time and $end_time
),

consider_latency as (
  select dca.DELIVERY_ID, min(dca.ORIGINAL_TIMESTAMP) as min_considered_time
  from segment_events.SERVER_EVENTS_PRODUCTION.DEEP_RED_DELIVERY_CONSIDERED_FOR_ASSIGNMENT dca 
  where dca.ORIGINAL_TIMESTAMP between $start_time and $end_time and (dca.is_time_to_assign is null or dca.is_time_to_assign='true')
  group by 1
),

solution_latency as (
  select cds.DELIVERY_ID, min(cds.ORIGINAL_TIMESTAMP) as min_solution_time
  from segment_events.SERVER_EVENTS_PRODUCTION.DEEP_RED_MUNKRES_CONSIDERED_DELIVERY_STATS cds
  where cds.ORIGINAL_TIMESTAMP between $start_time and $end_time
  group by 1
),

assignments as 
(
select
se.unit_id
, se.shift_id
, se.delivery_id
, se.result
, se.sp_id
, se.timing
, convert_timezone('UTC', 'America/Los_Angeles', se.timing) as la_timing
, convert_timezone('UTC', timezone, se.timing) as local_timing
, case when is_asap then datediff('seconds', dd.created_at, dd.actual_delivery_time) end as asap
, dd.distinct_active_duration as DAT
, dd.batch_id
, ROUTE_IS_SAME_BLOCK_BATCH
, ROUTE_IS_SAME_STORE_BATCH
, order_READY_ESTIMATION_SOURCE
, order_ready_time
, dd.dasher_assigned_time
, dd.is_asap
, dd.created_at
, dd.quoted_delivery_time
, dd.active_date
, dd.submarket_id
, dd.market_id
, extract(hour from convert_timezone(timezone, dd.created_at)) as hour
, dd.actual_pickup_time
, dd.actual_delivery_time
, dd.dasher_at_store_time 
, dd.estimated_delivery_time
, dd.dasher_confirmed_time
, dd.store_id
, dd.flf
, dis.ROAD_D2R_DISTANCE
, dis.road_r2c_distance as r2c
, case when is_asap then datediff('seconds', dd.created_at, csl.min_considered_time) end as considerlatency
, case when is_asap then datediff('seconds', dd.created_at, sl.min_solution_time) end as solulatency
, case when is_asap then datediff('seconds', dd.created_at, dd.first_assignment_made_time) end as alat
, datediff('seconds', dd.first_assignment_made_time, dd.dasher_assigned_time) as acceptlat
, datediff('seconds', dd.dasher_assigned_time, dd.dasher_confirmed_time) as conflat
, case when datediff('seconds', dd.quoted_delivery_time, dd.actual_delivery_time)>0
            then datediff('seconds', dd.quoted_delivery_time, dd.actual_delivery_time) else 0 end as lateness
, case when is_asap then datediff('seconds', dd.QUOTED_DELIVERY_TIME, dd.ACTUAL_DELIVERY_TIME) end as tardiness
, case when is_asap then datediff('seconds', dd.ESTIMATED_DELIVERY_TIME, dd.ACTUAL_DELIVERY_TIME) end as tardiness2
, case when is_asap then datediff('seconds', dd.CREATED_AT, dd.QUOTED_DELIVERY_TIME) end as quoted_asap
, case when is_asap then datediff('seconds', dd.CREATED_AT, dd.ESTIMATED_DELIVERY_TIME) end as est_asap
, case when is_asap then datediff('seconds', dd.CREATED_AT, order_ready_time) end as create_2_ready
, dd.d2c_duration
, dd.d2p_duration
, dd.d2r_duration
, dd.r2c_duration
, dd.t2p_duration
, dd.wap_duration
, dd.dasher_wait_duration
, dd.ORDER_PROTOCOL
, dd.BUSINESS_ID
, case when ddi.DELIVERY_WINDOW_END_TIME is not null or ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) then 1 else 0 end as is_windowed
, case when ddi.DELIVERY_WINDOW_END_TIME is not null and dd.ACTUAL_DELIVERY_TIME between ddi.DELIVERY_WINDOW_START_TIME and ddi.DELIVERY_WINDOW_END_TIME then 1
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) and abs(datediff('seconds', dd.ACTUAL_DELIVERY_TIME, dd.ESTIMATED_DELIVERY_TIME)/60.0)<15 then 1
       else 0 
  end as is_windowed_and_within_window
, case when ddi.DELIVERY_WINDOW_END_TIME is not null and dd.ACTUAL_DELIVERY_TIME < ddi.DELIVERY_WINDOW_START_TIME then 1
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) and (datediff('seconds', dd.ACTUAL_DELIVERY_TIME, dd.ESTIMATED_DELIVERY_TIME)/60.0)>15 then 1
       else 0 
  end as is_windowed_and_delivered_early
, case when ddi.DELIVERY_WINDOW_END_TIME is not null and dd.ACTUAL_DELIVERY_TIME > ddi.DELIVERY_WINDOW_END_TIME then 1
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) and (datediff('seconds', dd.ESTIMATED_DELIVERY_TIME, dd.ACTUAL_DELIVERY_TIME)/60.0)>15 then 1
       else 0 
  end as is_windowed_and_delivered_late
, flf_assign
, ideal_flf_assign
, xcredits_issued
, consumer_refund
, dd.mto 
, dd.delivery_rating
, dd.business_name
, case when ddi.DELIVERY_WINDOW_END_TIME is not null then 'with_end_time' 
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) then 'scheduled_not_drive'
       when dd.IS_ASAP then 'asap'
       else 'other'
  end as type_window
from assign se 
join dimension_deliveries dd on se.delivery_id=dd.delivery_id and se.shift_id=dd.SHIFT_ID
left join drive_prod.PUBLIC.MAINDB_DELIVERY_DRIVE_INFO ddi on dd.DELIVERY_ID=ddi.DELIVERY_ID and ddi.CREATED_AT between dateadd('day', -5, $start_time) and $end_time
left join public.fact_delivery_distances dis on dd.delivery_id=dis.delivery_id
left join consider_latency csl on csl.delivery_id=se.delivery_id
left join solution_latency sl on sl.delivery_id=se.DELIVERY_ID
where dd.DASHER_ASSIGNED_TIME between $start_time and $end_time and not dd.IS_GROUP_ORDER and not dd.IS_CONSUMER_PICKUP
), 


--account for edge cases where batches may straddle results by choosing the last delivery in a batch as determining the result
batch_assignments as
(
select 
delivery_id
, last_value(result) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as batch_result 
, last_value(ROUTE_IS_SAME_BLOCK_BATCH) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as is_same_block_batch
, last_value(ROUTE_IS_SAME_STORE_BATCH) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as is_same_store_batch
from assignments 
where actual_delivery_time is not NULL and batch_id is not NULL 
),

--combine main data and batch data
raw_data as
(
select a.*, batch_result, is_same_block_batch, is_same_store_batch
from assignments a 
left join batch_assignments ba on a.delivery_id=ba.delivery_id
),

--getting batch_length
batch_length as 
(
select batch_id, count(distinct delivery_id) as num, count(distinct store_id) as num_stores
from raw_data 
group by 1 
),

staging as(
  select 
  market_id, business_name
  , rd.delivery_id
  , order_READY_ESTIMATION_SOURCE
  , order_ready_time
  , result
  , asap 
  , quoted_asap
  , is_same_block_batch
  , is_same_store_batch
  , est_asap
  , ROAD_D2R_DISTANCE
  , tardiness
  , tardiness2
  , sp_id
  , create_2_ready
  , DAT
  , created_at
  , timing
  , local_timing
  , extract(hour from local_timing) as local_hour
  , la_timing
  , unit_id
  , rd.batch_id
  , e1.num as num_in_batch
  , e1.num_stores
  , is_asap
  , submarket_id
  , active_date
  , flf
  , dasher_assigned_time
  , quoted_delivery_time
  , actual_delivery_time
  , case when rd.batch_id is NULL then 'non-batch' else 'batch' end as batch_status
  , batch_result
  , alat
  , acceptlat
  , conflat
  , considerlatency
  , solulatency
  , lateness
  , r2c
  , d2c_duration
  , d2p_duration
  , d2r_duration
  , r2c_duration
  , t2p_duration
  , wap_duration
  , dasher_wait_duration
  , is_windowed
  , is_windowed_and_within_window
  , IS_WINDOWED_AND_DELIVERED_LATE
  , IS_WINDOWED_AND_DELIVERED_early
  , ORDER_PROTOCOL
  , type_window
  , flf_assign
  , ideal_flf_assign
  , coalesce(batch_result, result) as expt_group 
  , xcredits_issued + consumer_refund as cnr
  , mto 
  , delivery_rating
  from raw_data rd
  left join batch_length e1 on rd.batch_id=e1.batch_id 
)
select * from staging where asap < 2*60*60 and dat <2*60*60;