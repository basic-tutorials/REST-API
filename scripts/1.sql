drop table mkr_data;
create table mkr_data
   as
      select qnb.mkr_id id,
             (
                select finreal
                  from iibahramova.tmp_mkr_fin_random
                 where qnb.fin = finrandom
             ) fin,
             to_date(
                case
                   when regexp_like(qnb.date_of_birth,
                                    '^[0-9]{2}/[0-9]{2}/[0-9]{4}$') then
                      to_date(qnb.date_of_birth,
                              'DD/MM/YYYY')
                   when regexp_like(qnb.date_of_birth,
                                    '^[0-9]{2}-[A-Z]{3}-[0-9]{2}$') then
                      to_date(qnb.date_of_birth,
                              'DD-MON-RR')
                   else
                      null
                end,
                     'dd.mm.yyyy') as date_of_birth,
             to_date(qnb.mkr_date,
                     'dd.mm.yyyy') mkr_date,
             (
                select aa.name_az
                  from dwmain.dg_credit_type aa
                 where aa.code = qnb.credit_type
             ) as cr_type_look,
             (
                select cc.name
                  from dwmain.dg_colleteral_types cc
                 where qnb.collateral_code = cc.code
             ) as colletaral,
             qnb.p_id id_2,
             qnb.bank_id bank_id,
             qnb.bank_name bank_name,
             qnb.credit_type,
             (
                select aa.name_az
                  from dwmain.dg_credit_type aa
                 where aa.code = qnb.credit_type
             ) as credittype_name,
             qnb.org_type,
             to_date(to_char(
                to_date(substr(
                   qnb.granted_on,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') granted_on,
             qnb.initial_amount,
             qnb.line_ammount,
             qnb.days_interest_overdue,
             qnb.days_main_sum_overdue,
             to_date(to_char(
                to_date(substr(
                   qnb.contract_due_on,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') contract_due_on,
             qnb.interest_rate,
             to_date(to_char(
                to_date(substr(
                   qnb.last_update_date,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') last_update_date,
             to_date(to_char(
                to_date(substr(
                   qnb.last_payment_date,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') last_payment_date,
             qnb.outstanding_debt_main,
             qnb.outstanding_debt_interest,
             qnb.monthly_payment_amount,
             qnb.prolongations,
             qnb.l_credit_status,
             to_date(to_char(
                to_date(substr(
                   qnb.credit_status_close_date,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') credit_status_close_date,
             qnb.credit_purpose,
             (
                select e.name_az
                  from dwmain.dg_loan_purpose e
                 where e.code = qnb.credit_purpose
             ) creditpurpose_name,
             qnb.currency,
             qnb.mkr_id,
             qnb.collateral_code,
             (
                select c.name
                  from dwmain.dg_colleteral_types c
                 where qnb.collateral_code = c.code
             ) collateralt_type_name,
             qnb.collateral_market_value,
             qnb.collateral_registry_agency,
             to_date(to_char(
                to_date(substr(
                   qnb.collateral_registry_date,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') collateral_registry_date,
             qnb.collateral_registry_no,
             qnb.collateral_any_info,
             qnb.fk_person_liability liabilityid,
             qnb.overdue_days,
             to_date(to_char(
                to_date(qnb.overdue_period,
               'YYYY-MM-DD'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') overdue_period,
             qnb.lh_credit_status credit_status,
             to_date(to_char(
                to_date(substr(
                   qnb.file_date,
                   1,
                   19
                ),
               'YYYY-MM-DD HH24:MI:SS'),
                'DD.MM.YYYY'
             ),
                     'DD.MM.YYYY') file_date,
             to_date(qnb.mkr_date,
                     'dd.mm.yyyy') req_date
        from scoring.qnb_mkr_data_mba_backup qnb
       where trunc(to_date(qnb.mkr_date,
      'dd.mm.yyyy')) between to_date('01.01.2018','dd.mm.yyyy') and to_date('31.12.2021','dd.mm.yyyy');