/*
================================================================================
MKR IDS EXTRACTION (2018-2021)
================================================================================
Purpose: Link loan contracts to credit bureau (MKR) IDs for historical data
         Matches loans to the most recent MKR report within 11 days before loan start

Data Sources:
  - mabayramova.yekun_data_v1:           Loan contract data (FINKOD, BEGINDATE, MUQAVILE)
  - scoring.qnb_mkr_data_mba_backup:     Credit bureau backup data (MKR_ID, MKR_DATE)
  - mabayramova.id_fin_date:             FIN to ID mapping
  - iibahramova.tmp_mkr_fin_random:      FIN code anonymization mapping

Output:
  - MKR_IDS_2018_2021: Matched loan-to-MKR records for 2021 loans

Logic:
  1. Join loans to MKR records where MKR date is within 11 days before loan start
  2. Rank by most recent MKR date per loan (PARTITION BY FINKOD, BEGINDATE)
  3. Keep only the best match (rn = 1)
  4. Join to get anonymized IDs

Note: Update the year filter in WHERE clause as needed (currently 2021)
================================================================================
*/

CREATE TABLE MKR_IDS_2018_2021 AS
WITH ranked_mkr AS (
    SELECT
        loan.finkod                                                     AS fin,
        loan.begindate,
        mkr.mkr_date,
        ROW_NUMBER() OVER (
            PARTITION BY loan.finkod, loan.begindate
            ORDER BY mkr.mkr_date DESC
        )                                                               AS rn,
        mkr.mkr_id,
        loan.muqavile
    FROM mabayramova.yekun_data_v1 loan
    INNER JOIN scoring.qnb_mkr_data_mba_backup mkr
        ON mkr.fin = loan.finkod
        AND mkr.mkr_id IS NOT NULL
        AND loan.begindate BETWEEN TO_DATE(mkr.mkr_date, 'DD.MM.YYYY') - 11
                               AND TO_DATE(mkr.mkr_date, 'DD.MM.YYYY')
    WHERE EXTRACT(YEAR FROM loan.begindate) = 2021
)
SELECT
    m.fin,
    m.begindate,
    m.mkr_date,
    m.mkr_id,
    m.muqavile,
    f.ids
FROM ranked_mkr m
LEFT JOIN mabayramova.id_fin_date f
    ON m.fin = (
        SELECT finrandom
        FROM iibahramova.tmp_mkr_fin_random
        WHERE f.fincode = finreal
    )
    AND m.begindate = f.req_date
WHERE m.rn = 1;
