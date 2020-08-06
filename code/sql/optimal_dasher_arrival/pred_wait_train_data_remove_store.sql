create or replace table CHIZHANG.pred_wait_final_0805_remove_store as(
SELECT *
FROM CHIZHANG.pred_wait_final_0805
WHERE store_id NOT IN
    (SELECT store_id 
     FROM proddb.yihantan.pseudo_dasher_store_ids_v1)
);
grant select on CHIZHANG.pred_wait_final_0805_remove_store to role read_only_users;