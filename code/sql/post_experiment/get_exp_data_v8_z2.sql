set experiment_name = 'pred_wait_based_future_dasher';
set experiment_version = '8';
set zone = 'zone2';
set start_date = '2020-08-26';
set end_date = '2030-06-14';


create or replace table chizhang.pred_wait_v8_z2_all_exp_data as(
  
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
and received_at between $start_date and $end_date
and experiment_version=$experiment_version
and switchback_zone = $zone
group by 1, 2, 3
having count(distinct result) = 1
),

last_assign as
(
select
ai.DELIVERY_ID, ai.SHIFT_ID
, max(ai.received_at) as max_received_at
from SEGMENT_EVENTS.SERVER_EVENTS_PRODUCTION.DEEP_RED_ASSIGNMENT_INFO ai
where ai.ORIGINAL_TIMESTAMP between $start_date and $end_date
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
, ai.ROUTE_IS_SAME_PICKUP_BLOCK_INTERLACED_BATCH
, ai.FOOD_READY_ESTIMATION_SOURCE as order_READY_ESTIMATION_SOURCE
, to_timestamp_ntz(ai.FOOD_READY_TIME) as order_ready_time
, ai.ROUTE_IS_SAME_STORE_INTERLACED_BATCH
, ai.FLF as flf_assign
, ai.IDEAL_FLF as ideal_flf_assign
, ai.DELIVERY_PACKAGE_SIZE
from SEGMENT_EVENTS.SERVER_EVENTS_PRODUCTION.DEEP_RED_ASSIGNMENT_INFO ai
JOIN result se on ai.assignment_run_id=se.assignment_run_id and se.sp_id=ai.SP_ID
join last_assign fa on ai.received_at=fa.max_received_at and fa.delivery_id=ai.delivery_id and fa.shift_id=ai.SHIFT_ID
where ai.RECEIVED_AT between $start_date and $end_date
),

consider_latency as (
  select dca.DELIVERY_ID, min(dca.ORIGINAL_TIMESTAMP) as min_considered_time, any_value(dca.IS_INSTANT_ASSIGN_STORE) as is_instant_assign
  from segment_events.SERVER_EVENTS_PRODUCTION.DEEP_RED_DELIVERY_CONSIDERED_FOR_ASSIGNMENT dca
  where dca.ORIGINAL_TIMESTAMP between $start_date and $end_date and (dca.is_time_to_assign is null or dca.is_time_to_assign='true')
  group by 1
),

solution_latency as (
  select cds.DELIVERY_ID, min(cds.ORIGINAL_TIMESTAMP) as min_solution_time
  from segment_events.SERVER_EVENTS_PRODUCTION.DEEP_RED_MUNKRES_CONSIDERED_DELIVERY_STATS cds
  where cds.ORIGINAL_TIMESTAMP between $start_date and $end_date
  group by 1
),

assignments as
(
select
distinct
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
, ROUTE_IS_SAME_PICKUP_BLOCK_INTERLACED_BATCH
, ROUTE_IS_SAME_STORE_INTERLACED_BATCH
, order_READY_ESTIMATION_SOURCE
, order_ready_time
, DELIVERY_PACKAGE_SIZE
, dd.dasher_assigned_time
, dd.is_asap
, dd.created_at
, dd.NUM_ASSIGNS
, dd.quoted_delivery_time
, dd.active_date
, dd.submarket_id
, dd.subtotal
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
, case when is_asap then datediff('seconds', dd.QUOTED_DELIVERY_TIME, dd.ACTUAL_DELIVERY_TIME) end as tardiness
, case when is_asap then datediff('seconds', dd.ESTIMATED_DELIVERY_TIME, dd.ACTUAL_DELIVERY_TIME) end as tardiness2
, case when is_asap then datediff('seconds', dd.CREATED_AT, dd.QUOTED_DELIVERY_TIME) end as quoted_asap
, case when is_asap then datediff('seconds', dd.CREATED_AT, dd.ESTIMATED_DELIVERY_TIME) end as est_asap
, case when is_asap and datediff('seconds', dd.CREATED_AT, order_ready_time) between 0 and 3600
      then datediff('seconds', dd.CREATED_AT, order_ready_time)
  end as create_2_ready
, case when is_asap then datediff('seconds', dd.CREATED_AT, dd.actual_pickup_time) end as create_2_pickup
, dd.d2c_duration
, dd.d2p_duration
, dd.d2r_duration
, dd.r2c_duration
, dd.t2p_duration
, dd.wap_duration
, dd.dasher_wait_duration
, dd.ORDER_PROTOCOL
, dd.BUSINESS_ID
, dd.STORE_STARTING_POINT_ID
, dd.IS_FIRST_ORDERCART
, xcredits_issued
, consumer_refund
, dd.IS_FROM_STORE_TO_US
, xcredits_issued + consumer_refund as cnr
, mto
, is_instant_assign
, delivery_rating
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
, case when ddi.DELIVERY_WINDOW_END_TIME is not null then 'with_end_date'
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) then 'scheduled_not_drive'
       when dd.IS_ASAP then 'asap'
       else 'other'
  end as type_window
, case when ddi.DELIVERY_WINDOW_END_TIME is not null then ddi.DELIVERY_WINDOW_END_TIME
       when ((not dd.IS_ASAP) and (not dd.IS_FROM_STORE_TO_US)) then dateadd('minutes', 15, dd.QUOTED_DELIVERY_TIME)
  end as window_end_time
, case when datediff('seconds', COALESCE(window_end_time, dd.quoted_delivery_time), dd.actual_delivery_time)>0
            then datediff('seconds', COALESCE(window_end_time, dd.quoted_delivery_time), dd.actual_delivery_time) else 0 end as lateness
from assign se
join dimension_deliveries dd on se.delivery_id=dd.delivery_id and se.shift_id=dd.SHIFT_ID
left join DRIVE_PROD.PUBLIC.MAINDB_DELIVERY_DRIVE_INFO ddi on dd.DELIVERY_ID=ddi.DELIVERY_ID and ddi.CREATED_AT between dateadd('day', -5, $start_date) and $end_date
left join public.fact_delivery_distances dis on dd.delivery_id=dis.delivery_id
left join consider_latency csl on csl.delivery_id=se.delivery_id
left join solution_latency sl on sl.delivery_id=se.DELIVERY_ID
where dd.DASHER_ASSIGNED_TIME between $start_date and $end_date and not dd.IS_GROUP_ORDER and not dd.IS_CONSUMER_PICKUP
and not (la_timing between '2020-04-24 9:30' and '2020-04-26 22:30' AND DD.MARKET_ID in(8, 68, 63, 44))

),


--account for edge cases where batches may straddle results by choosing the last delivery in a batch as determining the result
batch_assignments as
(
select
delivery_id
, last_value(result) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as batch_result
, last_value(ROUTE_IS_SAME_PICKUP_BLOCK_INTERLACED_BATCH) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as is_same_block_batch
, last_value(ROUTE_IS_SAME_STORE_INTERLACED_BATCH) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as is_same_store_batch
, last_value(unit_id) over (partition by batch_id order by dasher_assigned_time rows between unbounded preceding and unbounded following) as batch_unit_id
from assignments
where actual_delivery_time is not NULL and batch_id is not NULL
),

--combine main data and batch data
raw_data as
(
select a.*, batch_result, is_same_block_batch, is_same_store_batch, batch_unit_id
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
--  market_id
  rd.delivery_id
--  , order_READY_ESTIMATION_SOURCE
 , order_ready_time
 , created_at
 , store_id
 , actual_delivery_time
  , ACTUAL_PICKUP_TIME
  , dasher_at_store_time
  , result
  , asap
  , NUM_ASSIGNS
  , case when rd.subtotal>7500 or rd.IS_FIRST_ORDERCART then asap end asap_best_in_tier
  , case when rd.store_starting_point_id in (140,7746,1513,5067,59,1079,282,248,226,185,32,6,63,14,16,129,21,5777,230,408,409,410,426,441,460,493,494,495,506,508,673,677,698,699,700,2209,2210,4544,4545,4546)
      then asap end asap_strategic_geo
  , case when rd.subtotal>7500 or rd.IS_FIRST_ORDERCART then dat end dat_best_in_tier
  , case when rd.store_starting_point_id in (140,7746,1513,5067,59,1079,282,248,226,185,32,6,63,14,16,129,21,5777,230,408,409,410,426,441,460,493,494,495,506,508,673,677,698,699,700,2209,2210,4544,4545,4546)
      then dat end dat_strategic_geo
  , rd.STORE_STARTING_POINT_ID
  , case when rd.batch_id is not null then asap end as asap_batched
  , case when rd.batch_id is null then asap end as asap_nonbatched
  , case when rd.batch_id is not null then dat end as dat_batched
  , case when rd.batch_id is null then dat end as dat_nonbatched
  , case when rd.batch_id is not null then lateness end as lateness_batched
  , case when rd.batch_id is null then lateness end as lateness_nonbatched

  , case when rd.IS_FROM_STORE_TO_US and datediff('seconds', rd.created_at, rd.QUOTED_DELIVERY_TIME)<3600
        then datediff('seconds', rd.created_at, rd.actual_delivery_time) end as asap_drive
  , case when rd.IS_FROM_STORE_TO_US then dat end as dat_drive
  , case when rd.IS_FROM_STORE_TO_US then lateness end as lateness_drive
  , case when rd.IS_FROM_STORE_TO_US then
          case when lateness*1.0 > 20*60 then 100 else 0 end
    end as pct_20min_late_drive
  , quoted_asap
--  , is_same_block_batch
--  , is_same_store_batch
--  , est_asap
--  , ROAD_D2R_DISTANCE
--  , tardiness
  , tardiness2
--  , sp_id
  , DELIVERY_PACKAGE_SIZE
  , create_2_ready
  , create_2_pickup
  , DAT
--  , created_at
--  , timing
--  , local_timing
--  , extract(hour from local_timing) as local_hour
--  , la_timing
  , coalesce(batch_unit_id, unit_id) as unit_id
--  , rd.batch_id
--  , e1.num as num_in_batch
--  , e1.num_stores
--  , is_asap
  , submarket_id
--  , active_date
--  , flf
--  , dasher_assigned_time
--  , quoted_delivery_time
  --, dasher_confirmed_time
  --, actual_pickup_time
  --, pickup_route_point_time
--  , actual_delivery_time
  --, dropoff_route_point_time
  -- , estimated_delivery_time
  -- , datediff('seconds', dasher_assigned_time, dasher_confirmed_time) - 20 as conf_diff
  -- , datediff('seconds', pickup_route_point_time, actual_pickup_time) as pickup_diff
  -- , datediff('seconds', dropoff_route_point_time, actual_delivery_time) as dropoff_diff
  -- , datediff('seconds', actual_pickup_time, actual_delivery_time) - datediff('seconds', pickup_route_point_time, dropoff_route_point_time) as pickup_dropoff_diff
  -- , case when rd.batch_id is NULL then datediff('seconds', actual_delivery_time, estimated_delivery_time) end as eta_diff
  --, ad.asap_degradation
  --, asap_degradation*1.00/nullif((case when is_batched is TRUE then route_overlap_counterfactual_debiased::int-route_overlap_projection_debiased::int end),0) as asap_dat_tradeoff
--  , case when rd.batch_id is NULL then 'non-batch' else 'batch' end as batch_status
  --, assignment_run_id
--  , batch_result
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
--  , is_windowed
--  , is_windowed_and_within_window
--  , IS_WINDOWED_AND_DELIVERED_LATE
--  , IS_WINDOWED_AND_DELIVERED_early
--  , flf_assign
--  , ideal_flf_assign
  , case when lateness*1.0 > 20*60 then 100 else 0 end as pct_20min_late
  , case when rd.batch_id is not null then
         case when lateness*1.0 > 20*60 then 100 else 0 end
    end as pct_20min_late_batch
  , case when rd.batch_id is null then
         case when lateness*1.0 > 20*60 then 100 else 0 end
    end as pct_20min_late_nonbatch
  , cnr
  , mto
  , delivery_rating
  , case when rd.batch_id is not null then 100.0 else 0.0 end as pct_batched
  , case when is_same_block_batch='true' then 100.0 else 0.0 end as pct_same_block_batch
  , case when is_same_store_batch='true' then 100.0 else 0.0 end as pct_same_store_batch

  , coalesce(batch_result, result) as expt_group
  , case when DELIVERY_PACKAGE_SIZE>=2 then 100.0 else 0.0 end as pct_packaged
  , case when DELIVERY_PACKAGE_SIZE>=2 then asap end as asap_packaged
  , case when DELIVERY_PACKAGE_SIZE>=2 then dat end as dat_packaged
  , case when DELIVERY_PACKAGE_SIZE>=2 then lateness end as lateness_packaged

  , case when is_same_block_batch='true' and DELIVERY_PACKAGE_SIZE>=2 then 100.0 else 0.0 end as pct_same_block_package
  , case when is_same_store_batch='true' and DELIVERY_PACKAGE_SIZE>=2 then 100.0 else 0.0 end as pct_same_store_package

  , case when is_same_block_batch='true' then asap end as asap_same_block_batch
  , case when is_same_block_batch='true' then dat end as dat_same_block_batch
  , case when is_same_store_batch='true' then asap end as asap_same_store_batch
  , case when is_same_store_batch='true' then dat end as dat_same_store_batch
  , case when is_same_store_batch='true' then lateness end as lateness_same_store_batch
  , case when is_same_block_batch='true' then lateness end as lateness_same_block_batch

  , case when is_instant_assign='true' then asap end as asap_PDP
  , case when is_instant_assign='true' then dat end as dat_PDP
  , case when is_instant_assign='true' or ORDER_PROTOCOL='DASHER_PLACE' then asap end as asap_DP_PDP
  , case when is_instant_assign='true' or ORDER_PROTOCOL='DASHER_PLACE' then dat end as DAT_DP_PDP

  , case when is_instant_assign='false' and ORDER_PROTOCOL<>'DASHER_PLACE' then
      case when datediff('seconds', dasher_at_store_time, order_ready_time)>600 then 100 else 0 end
    end as pct_arrive_10min_early

  , case when is_instant_assign='false' and ORDER_PROTOCOL<>'DASHER_PLACE' then
      case when datediff('seconds', dasher_at_store_time, order_ready_time)>360 then 100 else 0 end
    end as pct_arrive_6min_early

  , case when is_instant_assign='false' and ORDER_PROTOCOL<>'DASHER_PLACE' then
      case when datediff('seconds', dasher_at_store_time, order_ready_time)<-300 then 100 else 0 end
    end as pct_arrive_5min_late

  , case when is_instant_assign='false' and ORDER_PROTOCOL<>'DASHER_PLACE' then
      case when datediff('seconds', dasher_at_store_time, order_ready_time)/60 < -5 then datediff('seconds', order_ready_time, dasher_at_store_time)/60 - 5 else 0 end
    end as minutes_5min_lateness
  , pow(minutes_5min_lateness, 2) as minutes_5min_lateness_square

  , case when is_instant_assign='false' and ORDER_PROTOCOL<>'DASHER_PLACE' then
      case when datediff('seconds', dasher_at_store_time, order_ready_time)/60 > 6 then datediff('seconds', dasher_at_store_time, order_ready_time)/60 - 6 else 0 end
    end as minutes_6min_earliness
  , pow(minutes_6min_earliness, 2) as minutes_6min_earliness_square

  , case when rd.ORDER_PROTOCOL='POINT_OF_SALE' then asap end as asap_pos
  , case when rd.ORDER_PROTOCOL='POINT_OF_SALE' then dat end as dat_pos
  , case when rd.ORDER_PROTOCOL='POINT_OF_SALE' then lateness end as lateness_pos

  , case when rd.ORDER_PROTOCOL='TABLET' then asap end as asap_tablet
  , case when rd.ORDER_PROTOCOL='TABLET' then dat end as dat_tablet
  , case when rd.ORDER_PROTOCOL='TABLET' then lateness end as lateness_tablet

  , case when rd.flf < 0.5 then asap end as asap_flf_00_05
  , case when rd.flf >=0.5 and rd.flf < 1 then asap end as asap_flf_05_10
  , case when rd.flf >= 1 and rd.flf < 1.5 then asap end as asap_flf_10_15
  , case when rd.flf >= 1.5 and rd.flf < 2.0 then asap end as asap_flf_15_20
  , case when rd.flf >= 2.0 then asap end as asap_flf_20_inf

  , case when rd.flf < 0.5 then dat end as dat_flf_00_05
  , case when rd.flf >=0.5 and rd.flf < 1 then dat end as dat_flf_05_10
  , case when rd.flf >= 1 and rd.flf < 1.5 then dat end as dat_flf_10_15
  , case when rd.flf >= 1.5 and rd.flf < 2.0 then dat end as dat_flf_15_20
  , case when rd.flf >= 2.0 then dat end as dat_flf_20_inf

  from raw_data rd
  left join batch_length e1 on rd.batch_id=e1.batch_id
)

-- select
-- s.*, coalesce(mer.ASAP_RESIDUAL, 0) ASAP_RESIDUAL, coalesce(mer.DAT_RESIDUAL, 0) DAT_RESIDUAL,
-- coalesce(mep.pred_asap, 2089) as pred_asap,
-- coalesce(mep.pred_dat, 1284) as pred_dat
-- from staging s
-- left join PRODDB.PUBLIC.fact_ae_metric_experiment_residuals mer on s.delivery_id=mer.delivery_id
-- left join PRODDB.PUBLIC.FACT_AE_METRIC_EXP_PREDICTION_HISTORICAL mep on s.delivery_id=mep.delivery_id

select
 s.*
from staging s
where
(asap is NULL or asap < 2*60*60) and dat <2*60*60
)