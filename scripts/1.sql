drop table mkr_data;
create table mkr_data as
select 
 qnb.mkr_id ID,
(select finreal from iibahramova.tmp_mkr_fin_random where qnb.fin = finrandom) FIN,

--to_date(qnb.date_of_birth,'dd.mm.yyyy') date_of_birth,--niye burda error vermir
--TO_CHAR(TO_DATE(qnb.date_of_birth,'DD/MM/YYYY'),'DD.MM.YYYY') AS date_of_birth,
   to_date( CASE 
        WHEN REGEXP_LIKE(qnb.date_of_birth, '^[0-9]{2}/[0-9]{2}/[0-9]{4}$') THEN 
            TO_DATE(qnb.date_of_birth, 'DD/MM/YYYY')
        WHEN REGEXP_LIKE(qnb.date_of_birth, '^[0-9]{2}-[A-Z]{3}-[0-9]{2}$') THEN 
            TO_DATE(qnb.date_of_birth, 'DD-MON-RR')
        ELSE NULL
    END,'dd.mm.yyyy') AS date_of_birth,

to_date(qnb.mkr_date,'dd.mm.yyyy') mkr_date, --niye burda error vermir 
(SELECT aa.name_az FROM dwmain.DG_CREDIT_TYPE aa WHERE aa.code = qnb.credit_type) AS cr_type_look,
(SELECT cc.name FROM dwmain.DG_COLLETERAL_TYPES cc WHERE qnb.collateral_code= cc.code) AS colletaral,
qnb.p_id ID_2,
qnb.bank_id BANK_ID,
qnb.Bank_Name BANK_NAME,
qnb.credit_type,
(SELECT aa.name_az FROM dwmain.DG_CREDIT_TYPE aa WHERE aa.code = qnb.credit_type) AS credittype_name,
qnb.org_type,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.granted_on, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY') granted_on,
qnb.initial_amount,
qnb.line_ammount,
qnb.days_interest_overdue,
qnb.days_main_sum_overdue,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.contract_due_on, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY') contract_due_on,
qnb.interest_rate,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.last_update_date, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY') last_update_date,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.last_payment_date, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY') last_payment_date,
qnb.outstanding_debt_main,
qnb.outstanding_debt_interest,
qnb.monthly_payment_amount,
qnb.prolongations,
qnb.l_credit_status,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.credit_status_close_date, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY') credit_status_close_date,
qnb.credit_purpose,
(SELECT e.name_az FROM dwmain.DG_LOAN_PURPOSE e WHERE e.code = qnb.credit_purpose) creditpurpose_name,
qnb.currency,
qnb.mkr_id,
qnb.collateral_code,
(SELECT c.name FROM dwmain.DG_COLLETERAL_TYPES c WHERE qnb.collateral_code = c.code) collateralt_type_name,
qnb.collateral_market_value,
qnb.collateral_registry_agency,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.collateral_registry_date, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY')collateral_registry_date,
qnb.collateral_registry_no,
qnb.collateral_any_info,
qnb.fk_person_liability liabilityid,
qnb.overdue_days,
--to_date(qnb.overdue_period,'dd.mm.yyyy') overdue_period,
TO_DATE(TO_CHAR(TO_DATE(qnb.overdue_period, 'YYYY-MM-DD'), 'DD.MM.YYYY'),'DD.MM.YYYY') overdue_period,

qnb.lh_credit_status CREDIT_STATUS,
--to_date(qnb.file_date,'dd.mm.yyyy') file_date,
TO_DATE(TO_CHAR(TO_DATE(SUBSTR(qnb.file_date, 1, 19), 'YYYY-MM-DD HH24:MI:SS'),'DD.MM.YYYY'),'DD.MM.YYYY')file_date,
to_date(qnb.mkr_date,'dd.mm.yyyy') req_date
from scoring.qnb_mkr_data_mba_backup qnb 
--where   trunc(qnb.mkr_date) between to_date('&t','dd.mm.yyyy') and to_date('&t1','dd.mm.yyyy');
WHERE trunc(to_date(qnb.mkr_date, 'dd.mm.yyyy')) BETWEEN to_date('&t','dd.mm.yyyy') AND to_date('&t1','dd.mm.yyyy');
--select * from mkr_data
