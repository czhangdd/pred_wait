create or replace table CHIZHANG.pred_wait_final_0820_remove_store as(
SELECT *
FROM CHIZHANG.pred_wait_final_0820
WHERE store_id NOT IN
    (SELECT store_id 
    --  FROM proddb.yihantan.pseudo_dasher_store_ids_v1
     FROM proddb.yihantan.instant_assign_08072020)
);
grant select on CHIZHANG.pred_wait_final_0820_remove_store to role read_only_users;