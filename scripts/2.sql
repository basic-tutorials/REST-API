drop table yekun_for_datamart;

create table yekun_for_datamart
   as
      with a as (
         select distinct a.id,
                         a.fin fincode,
                         a.mkr_date request_date
           from scoring.mkr_data a
      ),b as (
         select a.*,
                d.t_partyid client_code
           from a a,
                dwmain.dpartcode_dbt d
          where a.fincode = d.t_code
            and d.t_codekind = 101
      ),c as (
         select b.*,
                d.t_code rskod
           from b b,
                dwmain.dpartcode_dbt d
          where b.client_code = d.t_partyid
            and d.t_codekind = 1
      ),d as (
         select fincode,
                id,
                l_work_sector_vat,
                sum(salary) salary
           from (
            select fincode,
                   id,
                   nvl(
                      salary,
                      0
                   ) salary,
                   l_work_sector_vat
              from (
               select distinct b.fincode,
                               b.id,
                               a.t_employer_voen voen,
                               trunc(a.t_insert_date) insert_date,
                               a.t_emp_salary salary,
                               a.t_id,
                               a.t_nn
                 from ibs.asan_finance_is_yeri_v2@ibs_ro a,
                      a b
                where a.t_fin_code = b.fincode
                  and a.t_id in (
                  select max(t_id)
                    from ibs.asan_finance_is_yeri_v2@ibs_ro k
                   where k.t_fin_code = a.t_fin_code
               )
                  and a.t_contract_status_desc is not null
            ) x
         )
          group by fincode,
                   id,
                   l_work_sector_vat
      )
      select d.fincode,
             d.id,
             nvl(
                ibs.api_scoring_camunda_main.calc_net_from_gross@ibs_ro(
                   d.salary,
                   l_work_sector_vat
                ),
                salary
             ) salary
        from d
        left join c
      on c.fincode = d.fincode
         and c.id = d.id;