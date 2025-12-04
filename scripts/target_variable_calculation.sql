/*
================================================================================
TARGET VARIABLE CALCULATION
================================================================================
Purpose: Calculate binary target variable for credit scoring model
         TARGET = 1 if customer reaches 90+ days past due within 13 months
         TARGET = 0 otherwise

Data Sources:
  - BOB_MKR_DATA:      Customer demographic data (imported from Excel)
  - BOB_YEKUN_DATA_RAW: Loan payment history by month (imported from Excel)

Output:
  - BOB_YEKUN_DATA_2:  Final table with TARGET column for model training

Target Definition:
  - Performance window: 13 months after loan origination (BEGINDATE)
  - Default threshold: 90+ days past due (DPD)
  - Closed loans ('Bagli'): Treated as -1 (not delinquent)

Month Columns (Azerbaijani):
  SENTYABR=September, OKTYABR=October, NOYABR=November, DEKABR=December
  YANVAR=January, FEVRAL=February, MART=March, APREL=April
  MAY=May, IYUN=June, IYUL=July, AVQUST=August
================================================================================
*/


-- =============================================================================
-- STEP 1: DEDUPLICATE MKR CUSTOMER DATA
-- =============================================================================
-- Remove duplicate customer records from the Excel import

CALL DROP_TABLE('BOB_MKR_DATA_1');

CREATE TABLE BOB_MKR_DATA_1 PARALLEL 8 AS
SELECT DISTINCT
    mkr_date,
    fin,
    id,
    record_date_time,
    modification_date_time,
    status,
    person_type,
    place_of_birth,
    date_of_birth,
    location,
    registered_address,
    country,
    file_date,
    created_by,
    modified_by,
    fk_consent,
    participant_of_patriotic_war
FROM bob_mkr_data;  -- Source: Excel import

COMMIT;


-- =============================================================================
-- STEP 2: JOIN LOAN DATA WITH MKR CUSTOMER DATA
-- =============================================================================
-- Match loans to MKR records by FIN and date range (MKR date within 10 days before loan start)
-- Keep only the most recent MKR record per loan (ROW_NUMBER = 1)

CALL DROP_TABLE('BOB_YEKUN_DATA');

CREATE TABLE BOB_YEKUN_DATA PARALLEL 8 AS
SELECT
    finkod,
    muqavile,
    begindate,
    closedate,
    enddate,
    kredit_muddeti,
    yasam_muddeti,
    maas,
    bgn,
    meblag,
    max_gecikme,
    -- Monthly DPD columns (Sep 2018 - Nov 2022)
    sentyabr_2018, oktyabr_2018, noyabr_2018, dekabr_2018,
    yanvar_2019, fevral_2019, mart_2019, aprel_2019, may_2019, iyun_2019,
    iyul_2019, avqust_2019, sentyabr_2019, oktyabr_2019, noyabr_2019, dekabr_2019,
    yanvar_2020, fevral_2020, mart_2020, aprel_2020, may_2020, iyun_2020,
    iyul_2020, avqust_2020, sentyabr_2020, oktyabr_2020, noyabr_2020, dekabr_2020,
    yanvar_2021, fevral_2021, mart_2021, aprel_2021, may_2021, iyun_2021,
    iyul_2021, avqust_2021, sentyabr_2021, oktyabr_2021, noyabr_2021, dekabr_2021,
    yanvar_2022, fevral_2022, mart_2022, aprel_2022, may_2022, iyun_2022,
    iyul_2022, avqust_2022, sentyabr_2022, oktyabr_2022, noyabr_2022,
    -- MKR fields
    mkr_date,
    mkr_number,
    record_date_time,
    modification_date_time,
    person_type,
    place_of_birth,
    date_of_birth,
    location,
    registered_address,
    file_date,
    fk_consent,
    participant_of_patriotic_war
FROM (
    SELECT
        loan.*,
        mkr.mkr_date,
        mkr.fin,
        mkr.id                                                          AS mkr_number,
        mkr.record_date_time,
        mkr.modification_date_time,
        mkr.person_type,
        mkr.place_of_birth,
        mkr.date_of_birth,
        mkr.location,
        mkr.registered_address,
        mkr.file_date,
        mkr.fk_consent,
        mkr.participant_of_patriotic_war,
        ROW_NUMBER() OVER (PARTITION BY loan.muqavile ORDER BY mkr.mkr_date DESC) AS rn
    FROM bob_yekun_data_raw loan  -- Source: Excel import
    LEFT OUTER JOIN bob_mkr_data_1 mkr
        ON loan.finkod = mkr.fin
        AND mkr.mkr_date BETWEEN loan.begindate - 10 AND loan.begindate
)
WHERE rn = 1;

COMMIT;


-- =============================================================================
-- STEP 3: UNPIVOT MONTHLY DPD COLUMNS TO ROWS
-- =============================================================================
-- Convert wide format (51 month columns) to long format (rows)
-- This replaces 51 UNION ALL statements with a single UNPIVOT operation
-- 'Bagli' (closed) is converted to -1, numeric values are preserved

CALL DROP_TABLE('BOB_YEKUN_DATA_2_0');

CREATE TABLE BOB_YEKUN_DATA_2_0 COMPRESS FOR QUERY HIGH PARALLEL 8 AS
SELECT
    muqavile,
    part_id,
    DECODE(dpd_value, 'Bagli', '-1', dpd_value) * 1 AS gecikme
FROM bob_yekun_data
UNPIVOT (
    dpd_value FOR part_id IN (
        sentyabr_2018   AS 201809,
        oktyabr_2018    AS 201810,
        noyabr_2018     AS 201811,
        dekabr_2018     AS 201812,
        yanvar_2019     AS 201901,
        fevral_2019     AS 201902,
        mart_2019       AS 201903,
        aprel_2019      AS 201904,
        may_2019        AS 201905,
        iyun_2019       AS 201906,
        iyul_2019       AS 201907,
        avqust_2019     AS 201908,
        sentyabr_2019   AS 201909,
        oktyabr_2019    AS 201910,
        noyabr_2019     AS 201911,
        dekabr_2019     AS 201912,
        yanvar_2020     AS 202001,
        fevral_2020     AS 202002,
        mart_2020       AS 202003,
        aprel_2020      AS 202004,
        may_2020        AS 202005,
        iyun_2020       AS 202006,
        iyul_2020       AS 202007,
        avqust_2020     AS 202008,
        sentyabr_2020   AS 202009,
        oktyabr_2020    AS 202010,
        noyabr_2020     AS 202011,
        dekabr_2020     AS 202012,
        yanvar_2021     AS 202101,
        fevral_2021     AS 202102,
        mart_2021       AS 202103,
        aprel_2021      AS 202104,
        may_2021        AS 202105,
        iyun_2021       AS 202106,
        iyul_2021       AS 202107,
        avqust_2021     AS 202108,
        sentyabr_2021   AS 202109,
        oktyabr_2021    AS 202110,
        noyabr_2021     AS 202111,
        dekabr_2021     AS 202112,
        yanvar_2022     AS 202201,
        fevral_2022     AS 202202,
        mart_2022       AS 202203,
        aprel_2022      AS 202204,
        may_2022        AS 202205,
        iyun_2022       AS 202206,
        iyul_2022       AS 202207,
        avqust_2022     AS 202208,
        sentyabr_2022   AS 202209,
        oktyabr_2022    AS 202210,
        noyabr_2022     AS 202211
    )
);

COMMIT;


-- =============================================================================
-- STEP 4: CALCULATE MAX DPD IN 13-MONTH WINDOW
-- =============================================================================
-- For each loan, find the maximum DPD within 13 months after origination
-- Window: month 1 to month 13 after BEGINDATE

CALL DROP_TABLE('BOB_YEKUN_DATA_2');

CREATE TABLE BOB_YEKUN_DATA_2 COMPRESS FOR QUERY HIGH PARALLEL 8 AS
SELECT
    loan.*,
    dpd_calc.gecikme_13m
FROM bob_yekun_data loan
INNER JOIN (
    SELECT
        loan.muqavile,
        MAX(
            CASE
                WHEN dpd.part_id BETWEEN TO_NUMBER(TO_CHAR(ADD_MONTHS(loan.begindate, 1), 'YYYYMM'))
                                     AND TO_NUMBER(TO_CHAR(ADD_MONTHS(loan.begindate, 13), 'YYYYMM'))
                THEN dpd.gecikme
                ELSE -1
            END
        ) AS gecikme_13m
    FROM bob_yekun_data loan
    INNER JOIN bob_yekun_data_2_0 dpd
        ON loan.muqavile = dpd.muqavile
    GROUP BY loan.muqavile
) dpd_calc
    ON loan.muqavile = dpd_calc.muqavile;

COMMIT;


-- =============================================================================
-- STEP 5: GENERATE FINAL TARGET VARIABLE
-- =============================================================================
-- TARGET = 1 if max DPD in 13-month window exceeds 90 days
-- TARGET = 0 otherwise

SELECT
    muqavile,
    CASE WHEN gecikme_13m > 90 THEN 1 ELSE 0 END AS target
FROM bob_yekun_data_2;
