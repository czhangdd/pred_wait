drop table if exists CHIZHANG.WAIT_GEO_TABLE_REMOVE_STORE;
create table CHIZHANG.WAIT_GEO_TABLE_REMOVE_STORE as(
SELECT *
FROM CHIZHANG.WAIT_GEO_TABLE
WHERE store_id NOT IN
    (SELECT store_id 
     FROM proddb.yihantan.pseudo_dasher_store_ids_v1)
);
grant select on CHIZHANG.WAIT_GEO_TABLE_REMOVE_STORE to role read_only_users;