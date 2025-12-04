/*
================================================================================
CREDIT BUREAU UTILITY FUNCTIONS
================================================================================
Purpose: Helper functions for processing credit bureau payment history strings
Target:  Create once in the scoring user schema
Note:    These functions are already deployed - no need to re-run

Functions:
  1. ACCPYMTSTDRV - Align payment history string to MKR report date
  2. WPS          - Calculate Worst Payment Status from payment string
================================================================================
*/


-- =============================================================================
-- FUNCTION 1: ACCPYMTSTDRV (Account Payment String Derivation)
-- =============================================================================
-- Purpose: Aligns a payment history string to the MKR (credit bureau) report date
--          by shifting/padding the string based on date difference
--
-- Parameters:
--   INSTRING         - Payment history string (e.g., '000012300')
--   MKR_DATE         - Credit bureau report date in YYYYMM format
--   MX_OVERDUE_PERIOD - Maximum overdue period date in YYYYMM format
--
-- Returns:
--   Adjusted payment string aligned to MKR date
--   - If MKR_DATE > MX_OVERDUE_PERIOD: Left-pads with 'N' (no data)
--   - If MKR_DATE <= MX_OVERDUE_PERIOD: Truncates from left
--
-- Example:
--   ACCPYMTSTDRV('000123', '202301', '202212') returns 'N000123' (1 month ahead)
--   ACCPYMTSTDRV('000123', '202212', '202301') returns '00123'   (1 month behind)
-- =============================================================================

CREATE OR REPLACE FUNCTION ACCPYMTSTDRV(
    INSTRING          VARCHAR2,
    MKR_DATE          VARCHAR2,
    MX_OVERDUE_PERIOD VARCHAR2
)
RETURN VARCHAR2
IS
    v_month_diff  NUMBER;
    v_result      VARCHAR2(250);
    v_length      NUMBER;
    v_pad_char    VARCHAR2(1) := 'N';  -- 'N' = No data available
BEGIN
    -- Calculate month difference between MKR date and overdue period
    v_month_diff := (SUBSTR(MKR_DATE, 1, 4) * 12 + SUBSTR(MKR_DATE, 5, 2))
                  - (SUBSTR(MX_OVERDUE_PERIOD, 1, 4) * 12 + SUBSTR(MX_OVERDUE_PERIOD, 5, 2));

    v_length := LENGTH(INSTRING);

    IF v_month_diff <= 0 THEN
        -- MKR date is before or equal to overdue period: trim from left
        v_result := SUBSTR(INSTRING, (-1 * v_month_diff) + 1, v_length + v_month_diff);
    ELSE
        -- MKR date is after overdue period: pad with 'N' on left
        v_result := LPAD(INSTRING, v_length + v_month_diff, v_pad_char);
    END IF;

    RETURN v_result;
END;
/


-- =============================================================================
-- FUNCTION 2: WPS (Worst Payment Status)
-- =============================================================================
-- Purpose: Calculates the worst (highest) payment status from a payment string
--          Used to determine the maximum delinquency level in credit history
--
-- Parameters:
--   INSTRING - Payment history string containing status codes
--
-- Status Code Mapping:
--   '0'-'9' = Days past due bucket (0=current, 1=30 days, ..., 9=270+ days)
--   'T'     = 8 (Transferred/Sold)
--   'L','K' = 9 (Legal/Collection)
--   'D'     = -1 (Dormant)
--   'U'     = -1 (Unknown)
--   'X'     = -1 (Account closed)
--   'N'     = -1 (No data)
--   Other   = -2 (Invalid character)
--   NULL    = -3 (Empty string)
--
-- Returns:
--   Maximum payment status value found in the string
--   Higher values indicate worse payment behavior
--
-- Example:
--   WPS('000012300') returns 3 (worst status is '3')
--   WPS('00001TL00') returns 9 (worst status is 'L'=9)
--   WPS('NNNN')      returns -1 (all no-data)
--   WPS(NULL)        returns -3 (empty input)
-- =============================================================================

CREATE OR REPLACE FUNCTION WPS(INSTRING VARCHAR2)
RETURN NUMBER
IS
    v_current_status  NUMBER;
    v_worst_status    NUMBER := -1;
    v_char            VARCHAR2(1);
    v_input           VARCHAR2(2000);
BEGIN
    -- Handle NULL or empty input
    v_input := UPPER(INSTRING);

    IF TRIM(v_input) IS NULL THEN
        RETURN -3;
    END IF;

    -- Iterate through each character to find worst status
    FOR i IN 1 .. LENGTH(v_input) LOOP
        v_char := SUBSTR(v_input, i, 1);

        -- Map character to numeric status
        v_current_status := CASE
            WHEN v_char IN ('D', 'U', 'X', 'N') THEN -1  -- Dormant/Unknown/Closed/NoData
            WHEN v_char = 'T'                   THEN 8   -- Transferred
            WHEN v_char IN ('L', 'K')           THEN 9   -- Legal/Collection
            WHEN v_char IN ('0','1','2','3','4','5','6','7','8','9')
                                                THEN TO_NUMBER(v_char)
            ELSE -2  -- Invalid character
        END;

        -- Track worst (highest) status
        IF v_current_status > v_worst_status THEN
            v_worst_status := v_current_status;
        END IF;
    END LOOP;

    RETURN v_worst_status;
END;
/
