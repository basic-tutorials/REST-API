DROP TABLE YEKUN_FOR_DATAMART;

CREATE TABLE YEKUN_FOR_DATAMART AS

with a as
(select distinct a.id,a.fin fincode,a.mkr_date request_date from scoring.mkr_data a),----CEDVELI DEYISH

b as 
(select a.*,d.t_partyid client_code from a a, dwmain.dpartcode_dbt d
where a.fincode=d.t_code
and d.t_codekind=101),

c as
(select b.*, d.t_code rskod from b b, dwmain.dpartcode_dbt d
where b.client_code=d.t_partyid
and d.t_codekind=1),


d as
(select 
fincode,
id,
l_work_sector_vat,
sum(salary) salary
from 
(select fincode,
id,
NVL(salary,0) salary,
  2--IBS.CONST_SCORING_CAMUNDA.SECTOR_OZEL
l_work_sector_vat --HAFIZA 11/27/2024
from 
(select
distinct
b.fincode,
b.id,
a.t_employer_voen voen,
trunc(a.t_insert_date) insert_date,
a.t_emp_salary salary,
a.t_id
,a.t_nn
from ibs.asan_finance_is_yeri_v2@ibs_ro a, a b
where a.t_fin_code=b.fincode
AND A.T_ID IN (SELECT MAX(T_ID) FROM ibs.asan_finance_is_yeri_v2@ibs_ro K WHERE K.T_FIN_CODE = A.T_FIN_CODE)
--and trunc(a.t_insert_date)=b.request_date
and a.t_contract_status_desc is not null) x )
group by fincode,id, l_work_sector_vat)




select d.fincode,d.id, nvl(ibs.api_scoring_camunda_main.calc_net_from_gross@ibs_ro(d.salary,L_WORK_SECTOR_VAT),salary) salary from d
left join c on c.fincode=d.fincode
and c.id=d.id; 
