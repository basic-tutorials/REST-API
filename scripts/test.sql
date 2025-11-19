select fin,MKR_DATE from scoring.qnb_mkr_data_mba_backup;
select FINKOD as fin, BEGINDATE from mabayramova.yekun_data_v1;

select * from mabayramova.yekun_data_v1 a inner join scoring.qnb_mkr_data_mba_backup b ON a.FINKOD = b.fin;

WITH ranked_mkr AS (
    SELECT 
        b.FINKOD as fin,
        b.BEGINDATE,
        a.MKR_DATE,
        ROW_NUMBER() OVER (
            PARTITION BY b.FINKOD, b.BEGINDATE 
            ORDER BY a.MKR_DATE DESC
        ) as rn
    FROM mabayramova.yekun_data_v1 b
    INNER JOIN scoring.qnb_mkr_data_mba_backup a
        ON a.fin = b.FINKOD
        AND TO_DATE(a.MKR_DATE, 'DD/MM/YYYY') <= TO_DATE(b.BEGINDATE, 'DD/MM/YYYY')
        
)
SELECT *
FROM ranked_mkr
WHERE rn = 1;