set parking_duration = 0*60;
set pickup_duration = 0*60;
//set start_time = '2020-06-15';
//set end_time = '2020-12-31';
set experiment_name = 'pred_wait_based_future_dasher';
set experiment_version = 3;
set start_time = '2020-06-11';
set end_time = '2030-02-01';

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
and experiment_version=$experiment_version
group by 1, 2, 3
having count(distinct result) = 1 
)
, 
considered_for_assignment as(
select to_timestamp(order_ready_time) as order_ready_time_deep_red
, delivery_id
, rank() over(partition by delivery_id order by original_timestamp desc) as rank_order
from segment_events.server_events_production.deep_red_delivery_considered_for_assignment 
where ORIGINAL_TIMESTAMP > $start_time
)
,
last_assign as (
select 
DELIVERY_ID
, RANK() OVER (PARTITION BY delivery_id  ORDER BY TIMESTAMP DESC ) as rank_order
, DASHER_WAIT_AT_STORE as pred_wait
, DASHER_WAIT_AT_STORE + PARKING_DURATION as pred_wait_plus_park
, parking_duration as pred_park
, r2c_duration
from segment_events.server_events_production.deep_red_assignment_info ai
JOIN result se on ai.assignment_run_id=se.assignment_run_id and se.sp_id=ai.SP_ID
where original_timestamp >  $start_time
)
,
deliveries as(
select 
dd.created_at 
, datediff('seconds', created_at, ACTUAL_PICKUP_TIME)/60 as create_to_pickup
, datediff('seconds', actual_pickup_time, ACTUAL_DELIVERY_TIME)/60 as pickup_to_deliver 
, datediff('seconds', created_at, order_ready_time_deep_red)/60 as estimated_prep_time
, dateadd('seconds', $PARKING_DURATION, DASHER_AT_STORE_TIME) as arrive_adj  -- added by Chi
, dateadd('seconds', -1*($PICKUP_DURATION), actual_pickup_time) as pickup_adj
--, datediff('seconds', least(arrive_adj, order_ready_time_deep_red), least(pickup_adj, order_ready_time_deep_red))/60 as wait_before_ready
, datediff('seconds', greatest(order_ready_time_deep_red, arrive_adj), greatest(order_ready_time_deep_red, pickup_adj))/60 as wait_after_ready
, dasher_wait_duration/60 as total_wait
--, wait_before_ready + wait_after_ready + {PICKUP_DURATION}/60 + {PARKING_DURATION}/60 as wait_check
, greatest(0,datediff('second', pickup_adj, order_ready_time_deep_red))/60 as pickup_before_ready
, greatest(0,datediff('seconds', order_ready_time_deep_red, arrive_adj))/60 as late_arrival_min
--, greatest(pred_wait_plus_park, 0)/60 as pred_wait_plus_park
--, la.r2c_duration as pred_r2c
, WAS_BATCHED
, dd.R2C_DURATION
from dimension_deliveries dd
left join considered_for_assignment cfa on dd.delivery_id = cfa.delivery_id and cfa.rank_order = 1
left join last_assign la on dd.delivery_id = la.delivery_id and la.rank_order = 1
where is_asap = true and is_consumer_pickup = false and is_filtered = true 
and created_at between $start_time and date_trunc('week', current_date)
and order_protocol != 'DASHER_PLACE'
and business_id not in (1855, 491, 10171, 47852, 7376, 5579)
    and dasher_at_store_time is not null
    and actual_pickup_time is not null
) 

select 
date_trunc('week', created_at) as week
--, avg(create_to_pickup) as create_to_pickup
, avg(case when estimated_prep_time between 0 and 120 then estimated_prep_time end) as estimated_prep_time
, avg(wait_after_ready) as wait_after_ready
, avg(case when late_arrival_min between 0 and 120 then late_arrival_min end) as late_arrival
, avg(pickup_to_deliver) as pickup_to_deliver_aka_r2c
, avg(pickup_before_ready) as early_pickup
from deliveries
group by 1
order by 1