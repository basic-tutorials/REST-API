drop table mkr_data;
create table mkr_data as
select 
       a.id,
       a.request_type_id,
       a.client_code,
       a.querypurpose,
       d.fin FIN,
       d.dateofbith  DATE_OF_BIRTH,
       a.active,
       a.is_sended,
       trunc(a.request_date) MKR_DATE,
/*     (select p.name from ibs_rep.mv_acb_inquiry_purposes p
       where a.querypurpose=p.id) inq_purpose,*/
       (select aa.name from IBS_REP.MV_ACB_CREDIT_TYPES aa
         where aa.code = b.credittype) cr_type_look,
     /* (select e.name from ibs_rep.mv_acb_credit_purpose_types  e
         where e.code = b.creditpurpose) credit_purpose_,*/
       (select name
          from dwmain.DG_COLLETERAL_TYPES cc
         where b.collateralcode = cc.code) colletaral,
       b.req_id req_id_1,
       b.id ID_2,
       b.bankid BANK_ID,
       b.bankname BANK_NAME,
       b.accountno,
       b.credittype CREDIT_TYPE,
       b.credittypename,
       b.orgidtype ORG_TYPE,
       b.grantedon GRANTED_ON,
       b.initialamount INITIAL_AMOUNT,
       b.lineamount LINE_AMMOUNT,
       b.daysinterestoverdue DAYS_INTEREST_OVERDUE,
       b.daysmainsumoverdue DAYS_MAIN_SUM_OVERDUE,
       b.contractdueon CONTRACT_DUE_ON,
       b.interestrate INTEREST_RATE,
       b.lastupdateddate LAST_UPDATE_DATE,
       b.lastpaymentdate LAST_PAYMENT_DATE,
       b.outstandingdebtmain OUTSTANDING_DEBT_MAIN,
       b.outstandingdebtinterest OUTSTANDING_DEBT_INTEREST,
       b.monthlypaymentamount MONTHLY_PAYMENT_AMOUNT,
       b.prolongations PROLONGATIONS,
       b.creditstatus CREDIT_STATUS,
       b.creditstatusclosedate CREDIT_STATUS_CLOSE_DATE,
       b.creditpurpose CREDIT_PURPOSE,
       b.creditpurposename,
       b.currency CURRENCY,
       b.mkrid MKR_ID,
       b.collateralcode COLLATERAL_CODE,
       b.collateraltypename,
       b.collateralmarketvalue COLLATERAL_MARKET_VALUE,
       b.collateralregistryagency COLLATERAL_REGISTRY_AGENCY,
       b.collateralregistrydate COLLATERAL_REGISTRY_DATE,
       b.collateralregistryno COLLATERAL_REGISTRY_NO,
       b.collateralanyinfo COLLATERAL_ANY_INFO,
       c.req_id req_id_2,
       c.liabilityid,
       c.overduedays OVERDUE_DAYS,
       to_date(replace(reportingperiod, 'x', '.'),'mm.yyyy') OVERDUE_PERIOD,
       c.creditstatus,
       d.filedate FILE_DATE,
       trunc(a.request_date) req_date
       

  from IBS_REP.MV_ACB_REQUEST_DATA      a,
       IBS_REP.MV_ACB_LIABILITY_INFO    b,
       IBS_REP.MV_ACB_LIABILITY_HISTORY c,
       IBS_REP.MV_ACB_BORROWER_INFO    d--,
       --mabayramova.satis_02_and_05 e
      


where a.id = b.req_id
   --and a.fincode in ()
   and c.req_id = a.id

   and b.id = c.liabilityid
   
   and d.req_id=a.id
   --and b.req_id='1369720'
   and trunc(a.request_date) between to_date('29.01.2025','dd.mm.yyyy') and to_date('06.05.2025','dd.mm.yyyy');
