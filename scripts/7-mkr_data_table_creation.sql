/*
================================================================================
MKR DATA TABLE CREATION
================================================================================
Purpose: Extract credit bureau (MKR) data for scoring model
Source:  IBS_REP materialized views (ACB = Azerbaijan Credit Bureau)
Output:  mkr_data table with customer loan and liability information

Tables:
  - MV_ACB_REQUEST_DATA:      Credit bureau request metadata
  - MV_ACB_LIABILITY_INFO:    Loan/liability details
  - MV_ACB_LIABILITY_HISTORY: Payment history and overdue info
  - MV_ACB_BORROWER_INFO:     Customer demographic data

Date Range: Configurable (update dates in WHERE clause)
================================================================================
*/

DROP TABLE mkr_data;

CREATE TABLE mkr_data AS
SELECT
    -- =========================================================================
    -- REQUEST IDENTIFIERS
    -- =========================================================================
    a.id,
    a.request_type_id,
    a.client_code,
    a.querypurpose,
    a.active,
    a.is_sended,
    TRUNC(a.request_date)                               AS mkr_date,
    TRUNC(a.request_date)                               AS req_date,

    -- =========================================================================
    -- CUSTOMER INFO (from borrower table)
    -- =========================================================================
    d.fin                                               AS fin,
    d.dateofbith                                        AS date_of_birth,
    d.filedate                                          AS file_date,

    -- =========================================================================
    -- CREDIT TYPE & COLLATERAL LOOKUPS
    -- =========================================================================
    credit_type_lkp.name                                AS cr_type_look,
    collateral_lkp.name                                 AS colletaral,

    -- =========================================================================
    -- LIABILITY INFO (loan details)
    -- =========================================================================
    b.req_id                                            AS req_id_1,
    b.id                                                AS id_2,
    b.bankid                                            AS bank_id,
    b.bankname                                          AS bank_name,
    b.accountno,
    b.credittype                                        AS credit_type,
    b.credittypename,
    b.orgidtype                                         AS org_type,
    b.grantedon                                         AS granted_on,
    b.initialamount                                     AS initial_amount,
    b.lineamount                                        AS line_ammount,
    b.contractdueon                                     AS contract_due_on,
    b.interestrate                                      AS interest_rate,
    b.lastupdateddate                                   AS last_update_date,
    b.lastpaymentdate                                   AS last_payment_date,
    b.outstandingdebtmain                               AS outstanding_debt_main,
    b.outstandingdebtinterest                           AS outstanding_debt_interest,
    b.monthlypaymentamount                              AS monthly_payment_amount,
    b.prolongations,
    b.creditstatus                                      AS credit_status,
    b.creditstatusclosedate                             AS credit_status_close_date,
    b.creditpurpose                                     AS credit_purpose,
    b.creditpurposename,
    b.currency,
    b.mkrid                                             AS mkr_id,

    -- =========================================================================
    -- COLLATERAL INFO
    -- =========================================================================
    b.collateralcode                                    AS collateral_code,
    b.collateraltypename,
    b.collateralmarketvalue                             AS collateral_market_value,
    b.collateralregistryagency                          AS collateral_registry_agency,
    b.collateralregistrydate                            AS collateral_registry_date,
    b.collateralregistryno                              AS collateral_registry_no,
    b.collateralanyinfo                                 AS collateral_any_info,

    -- =========================================================================
    -- DELINQUENCY INFO (from liability history)
    -- =========================================================================
    b.daysinterestoverdue                               AS days_interest_overdue,
    b.daysmainsumoverdue                                AS days_main_sum_overdue,
    c.req_id                                            AS req_id_2,
    c.liabilityid,
    c.overduedays                                       AS overdue_days,
    TO_DATE(REPLACE(c.reportingperiod, 'x', '.'), 'MM.YYYY') AS overdue_period,
    c.creditstatus

FROM IBS_REP.MV_ACB_REQUEST_DATA a

-- =========================================================================
-- JOIN LIABILITY INFO
-- =========================================================================
INNER JOIN IBS_REP.MV_ACB_LIABILITY_INFO b
    ON a.id = b.req_id

-- =========================================================================
-- JOIN LIABILITY HISTORY
-- =========================================================================
INNER JOIN IBS_REP.MV_ACB_LIABILITY_HISTORY c
    ON a.id = c.req_id
    AND b.id = c.liabilityid

-- =========================================================================
-- JOIN BORROWER INFO
-- =========================================================================
INNER JOIN IBS_REP.MV_ACB_BORROWER_INFO d
    ON a.id = d.req_id

-- =========================================================================
-- LOOKUP JOINS
-- =========================================================================
LEFT JOIN IBS_REP.MV_ACB_CREDIT_TYPES credit_type_lkp
    ON b.credittype = credit_type_lkp.code

LEFT JOIN dwmain.DG_COLLETERAL_TYPES collateral_lkp
    ON b.collateralcode = collateral_lkp.code

-- =========================================================================
-- DATE FILTER (update these dates as needed)
-- =========================================================================
WHERE TRUNC(a.request_date) BETWEEN TO_DATE('29.01.2025', 'DD.MM.YYYY')
                                AND TO_DATE('06.05.2025', 'DD.MM.YYYY');
