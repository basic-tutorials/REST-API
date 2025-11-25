WITH ranked_mkr AS (
    SELECT 
        b.FINKOD as fin,
        b.BEGINDATE,
        a.MKR_DATE,
        ROW_NUMBER() OVER (
            PARTITION BY b.FINKOD, b.BEGINDATE 
            ORDER BY a.MKR_DATE DESC
        ) as rn,
        a.mkr_id,
        b.muqavile
    FROM mabayramova.yekun_data_v1 b
    INNER JOIN scoring.qnb_mkr_data_mba_backup a
        ON a.fin = b.FINKOD and mkr_id is not null
        AND  b.BEGINDATE between TO_DATE(a.MKR_DATE, 'DD.MM.YYYY')-11 and TO_DATE(a.MKR_DATE, 'DD.MM.YYYY')
        and extract( b.BEGINDATE from year ) ='2019'
)
SELECT M.FIN, M.BEGINDATE, M.MKR_DATE,M.MKR_ID,M.mUQAVILE, f.ids
FROM ranked_mkr m left join mabayramova.id_fin_date f 
on  m.fin=(select finrandom from iibahramova.tmp_mkr_fin_random where f.fincode = finreal)
and m.begindate = f.req_date
WHERE rn = 1;