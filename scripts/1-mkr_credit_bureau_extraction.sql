/*
================================================================================
MKR (CREDIT BUREAU) DATA EXTRACTION SCRIPT
================================================================================
Purpose: Extract and transform credit bureau loan data with proper date handling
Source:  scoring.qnb_mkr_data_mba_backup
Output:  Cleaned credit bureau data with dates and lookup values
================================================================================
*/

WITH mkr_data AS (
    SELECT
        -- =====================================================================
        -- IDENTIFIERS
        -- =====================================================================
        qnb.mkr_id                                          AS id,
        fin_map.finreal                                     AS fin,
        qnb.p_id                                            AS id_2,
        qnb.mkr_id,

        -- =====================================================================
        -- BANK INFORMATION
        -- =====================================================================
        qnb.bank_id,
        qnb.bank_name,

        -- =====================================================================
        -- CREDIT TYPE & PURPOSE
        -- =====================================================================
        qnb.credit_type,
        credit_type_lookup.name_az                          AS credittype_name,
        credit_type_lookup.name_az                          AS cr_type_look,
        qnb.credit_purpose,
        purpose_lookup.name_az                              AS creditpurpose_name,
        qnb.org_type,

        -- =====================================================================
        -- LOAN DATES (with flexible date format parsing)
        -- =====================================================================
        -- Date of birth: handles DD/MM/YYYY and DD-MON-RR formats
        CASE
            WHEN REGEXP_LIKE(qnb.date_of_birth, '^\d{2}/\d{2}/\d{4}$')
                THEN TO_DATE(qnb.date_of_birth, 'DD/MM/YYYY')
            WHEN REGEXP_LIKE(qnb.date_of_birth, '^\d{2}-[A-Z]{3}-\d{2}$')
                THEN TO_DATE(qnb.date_of_birth, 'DD-MON-RR')
            ELSE NULL
        END                                                 AS date_of_birth,

        -- MKR report date
        TO_DATE(qnb.mkr_date, 'DD.MM.YYYY')                AS mkr_date,
        TO_DATE(qnb.mkr_date, 'DD.MM.YYYY')                AS req_date,

        -- Loan granted date
        TO_DATE(SUBSTR(qnb.granted_on, 1, 10), 'YYYY-MM-DD') AS granted_on,

        -- Contract due date
        TO_DATE(SUBSTR(qnb.contract_due_on, 1, 10), 'YYYY-MM-DD') AS contract_due_on,

        -- Last update date
        TO_DATE(SUBSTR(qnb.last_update_date, 1, 10), 'YYYY-MM-DD') AS last_update_date,

        -- Last payment date
        TO_DATE(SUBSTR(qnb.last_payment_date, 1, 10), 'YYYY-MM-DD') AS last_payment_date,

        -- Credit status close date
        TO_DATE(SUBSTR(qnb.credit_status_close_date, 1, 10), 'YYYY-MM-DD') AS credit_status_close_date,

        -- Overdue period
        TO_DATE(qnb.overdue_period, 'YYYY-MM-DD')          AS overdue_period,

        -- File date
        TO_DATE(SUBSTR(qnb.file_date, 1, 10), 'YYYY-MM-DD') AS file_date,

        -- =====================================================================
        -- LOAN AMOUNTS
        -- =====================================================================
        qnb.initial_amount,
        qnb.line_ammount,
        qnb.outstanding_debt_main,
        qnb.outstanding_debt_interest,
        qnb.monthly_payment_amount,
        qnb.interest_rate,

        -- =====================================================================
        -- DELINQUENCY INFO
        -- =====================================================================
        qnb.days_interest_overdue,
        qnb.days_main_sum_overdue,
        qnb.overdue_days,
        qnb.prolongations,

        -- =====================================================================
        -- CREDIT STATUS
        -- =====================================================================
        qnb.l_credit_status,
        qnb.lh_credit_status                               AS credit_status,

        -- =====================================================================
        -- COLLATERAL INFORMATION
        -- =====================================================================
        qnb.collateral_code,
        collateral_lookup.name                             AS collateralt_type_name,
        collateral_lookup.name                             AS colletaral,
        qnb.collateral_market_value,
        qnb.collateral_registry_agency,
        TO_DATE(SUBSTR(qnb.collateral_registry_date, 1, 10), 'YYYY-MM-DD') AS collateral_registry_date,
        qnb.collateral_registry_no,
        qnb.collateral_any_info,

        -- =====================================================================
        -- OTHER FIELDS
        -- =====================================================================
        qnb.currency,
        qnb.fk_person_liability                            AS liabilityid

    FROM scoring.qnb_mkr_data_mba_backup qnb

    -- =========================================================================
    -- LOOKUP JOINS
    -- =========================================================================
    LEFT JOIN iibahramova.tmp_mkr_fin_random fin_map
        ON qnb.fin = fin_map.finrandom

    LEFT JOIN dwmain.dg_credit_type credit_type_lookup
        ON qnb.credit_type = credit_type_lookup.code

    LEFT JOIN dwmain.dg_loan_purpose purpose_lookup
        ON qnb.credit_purpose = purpose_lookup.code

    LEFT JOIN dwmain.dg_colleteral_types collateral_lookup
        ON qnb.collateral_code = collateral_lookup.code

    -- =========================================================================
    -- DATE FILTER
    -- =========================================================================
    WHERE TRUNC(TO_DATE(qnb.mkr_date, 'DD.MM.YYYY'))
        BETWEEN TO_DATE('01.01.2018', 'DD.MM.YYYY')
            AND TO_DATE('01.01.2018', 'DD.MM.YYYY')
)
SELECT * FROM mkr_data;
