/*
================================================================================
SALARY DATA EXTRACTION FOR DATAMART
================================================================================
Purpose: Extract and calculate net salary for customers from ASAN Finance data
Source:  scoring.mkr_data, ibs.asan_finance_is_yeri_v2
Output:  fincode, id, salary (net)

Logic:
  1. Get distinct customers from MKR data
  2. Map FIN code to client_code via dpartcode_dbt (codekind=101)
  3. Map client_code to rskod via dpartcode_dbt (codekind=1)
  4. Get latest salary record from ASAN Finance
  5. Convert gross salary to net using calc_net_from_gross function
================================================================================
*/

WITH
-- =============================================================================
-- STEP 1: Get distinct customers from MKR (Credit Bureau) data
-- =============================================================================
mkr_customers AS (
    SELECT DISTINCT
        id,
        fin         AS fincode,
        mkr_date    AS request_date
    FROM scoring.mkr_data
),

-- =============================================================================
-- STEP 2: Map FIN code to internal client code (codekind=101)
-- =============================================================================
fin_to_client AS (
    SELECT
        mkr.id,
        mkr.fincode,
        mkr.request_date,
        party.t_partyid AS client_code
    FROM mkr_customers mkr
    INNER JOIN dwmain.dpartcode_dbt party
        ON mkr.fincode = party.t_code
        AND party.t_codekind = 101
),

-- =============================================================================
-- STEP 3: Map client code to RS code (codekind=1) - used for join only
-- =============================================================================
client_to_rskod AS (
    SELECT
        fc.id,
        fc.fincode,
        fc.request_date,
        fc.client_code,
        party.t_code AS rskod
    FROM fin_to_client fc
    INNER JOIN dwmain.dpartcode_dbt party
        ON fc.client_code = party.t_partyid
        AND party.t_codekind = 1
),

-- =============================================================================
-- STEP 4: Get latest salary from ASAN Finance work records
-- =============================================================================
latest_salary AS (
    SELECT
        mkr.fincode,
        mkr.id,
        NVL(asan.t_emp_salary, 0)   AS salary,
        2                            AS l_work_sector_vat  -- IBS.CONST_SCORING_CAMUNDA.SECTOR_OZEL
    FROM mkr_customers mkr
    INNER JOIN ibs.asan_finance_is_yeri_v2@ibs_ro asan
        ON mkr.fincode = asan.t_fin_code
        AND asan.t_contract_status_desc IS NOT NULL
        -- Get only the latest record per customer
        AND asan.t_id = (
            SELECT MAX(k.t_id)
            FROM ibs.asan_finance_is_yeri_v2@ibs_ro k
            WHERE k.t_fin_code = asan.t_fin_code
        )
),

-- =============================================================================
-- STEP 5: Aggregate salary per customer (in case of multiple records)
-- =============================================================================
aggregated_salary AS (
    SELECT
        fincode,
        id,
        l_work_sector_vat,
        SUM(salary) AS salary
    FROM latest_salary
    GROUP BY fincode, id, l_work_sector_vat
)

-- =============================================================================
-- FINAL: Calculate net salary from gross using IBS function
-- =============================================================================
SELECT
    agg.fincode,
    agg.id,
    NVL(
        ibs.api_scoring_camunda_main.calc_net_from_gross@ibs_ro(
            agg.salary,
            agg.l_work_sector_vat
        ),
        agg.salary
    ) AS salary
FROM aggregated_salary agg
LEFT JOIN client_to_rskod rsk
    ON agg.fincode = rsk.fincode
    AND agg.id = rsk.id;
