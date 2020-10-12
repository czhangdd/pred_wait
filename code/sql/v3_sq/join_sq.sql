select
    *
    sq.score

from CHIZHANG.pred_wait_final_remove_store_1012 pw
join PRODDB.RAGHAV.FACT_SUPPLY_QUALITY_METRIC_BACKFILL sq
on sq.delivery_id = pw.delivery_id