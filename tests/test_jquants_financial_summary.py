"""Offline contract tests for the resumable J-Quants Financial Summary store."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import pytest
from jp_stock_analysis.cli import build_parser, main
from jp_stock_analysis.errors import DataValidationError, ProviderError
from jp_stock_analysis.providers.jquants import JQuantsProvider
from jp_stock_analysis.validation.jquants_financial_summary import *

RAW = {"Code":"72030","DiscDate":"2026-01-05","DiscTime":"15:00:00","DiscNo":"1","DocType":"FYFinancialStatements_Consolidated_JP","CurPerType":"FY","CurFYEn":"2026-03-31","Sales":100,"OP":2,"OdP":3,"NP":4,"EPS":5,"DEPS":6,"BPS":7,"TA":8,"Eq":9,"ShEq":10,"ROE":11,"EqAR":12,"CFO":13,"CFI":14,"CFF":15,"CashEq":16,"ShOutFY":17,"TrShFY":18,"AvgSh":19,"FSales":20,"FOP":21,"FOdP":22,"FNP":23,"FEPS":24,"FSales2Q":25,"FOP2Q":26,"FOdP2Q":27,"FNP2Q":28,"FEPS2Q":29,"NxFSales":30,"NxFOP":31,"NxFOdP":32,"NxFNp":33,"NxFEPS":34,"NxFSales2Q":35,"NxFOP2Q":36,"NxFOdP2Q":37,"NxFNP2Q":38,"NxFEPS2Q":39,"Div1Q":1,"Div2Q":2,"Div3Q":3,"DivFY":4,"DivAnn":5,"DivTotalAnn":6,"DivUnit":7,"PayoutRatioAnn":8,"FDiv1Q":9,"FDiv2Q":10,"FDiv3Q":11,"FDivFY":12,"FDivAnn":13,"FDivTotalAnn":14,"FDivUnit":15,"FPayoutRatioAnn":16,"NxFDiv1Q":17,"NxFDiv2Q":18,"NxFDiv3Q":19,"NxFDivFY":20,"NxFDivAnn":21,"NxFDivUnit":22,"NxFPayoutRatioAnn":23,"NCSales":24,"NCOP":25,"NCOdP":26,"NCNP":27,"NCEPS":28,"NCBPS":29,"NCTA":30,"NCEq":31,"NCShEq":32,"NCROE":33,"NCEqAR":34,"FNCSales":35,"FNCOP":36,"FNCOdP":37,"FNCNP":38,"FNCEPS":39,"NxFNCSales":40,"NxFNCOP":41,"NxFNCOdP":42,"NxFNCNP":43,"NxFNCEPS":44,"ChgAcEst":"a","ChgByASRev":"b","ChgNoASRev":"c","MatChgSub":"d","SigChgInC":"e","RetroRst":"f","FutureField":"kept","api_key":"actual-api-key"}
class Fake:
 def __init__(self, rows=None, errors=None): self.rows=rows or {}; self.errors=errors or {}; self.calls=[]
 def get_statement_rows(self,t):
  self.calls.append(t); value=self.errors.get(t)
  if isinstance(value,list): value=value.pop(0) if value else None
  if value: raise value
  return self.rows.get(t,[])
def uni(tmp_path,*tickers):
 p=tmp_path/'universe.txt'; p.write_text('\n'.join(tickers)+'\n',encoding='utf-8'); return p
def acquire(fake,u,store,**kw): return acquire_incremental(fake,universe_file=u,store_dir=store,allow_network=True,sleep_seconds=0,sleep_fn=lambda _:None,**kw)
def state(store): return json.loads((store/STATE_NAME).read_text())
def rows(store):
 with (store/CSV_NAME).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def test_01_v2_short_fields_map(): assert map_financial_summary_row(RAW)['operating_profit']==2
def test_02_five_digit_code_normalizes(): assert map_financial_summary_row(RAW)['ticker']=='7203'
def test_03_unmapped_source_field_preserved(): assert 'FutureField' in map_financial_summary_row(RAW)['source_fields_json']
def test_04_top_level_secret_removed(): assert 'actual-api-key' not in map_financial_summary_row(RAW)['source_fields_json']
def test_05_nested_dict_secret_removed(): assert sanitize_raw_row({'x':{'TOKEN':'x','ok':1}})=={'x':{'ok':1}}
def test_06_nested_list_secret_removed(): assert sanitize_raw_row([{'apiKEY':'x','v':1}])==[{'v':1}]
def test_07_hash_key_order_deterministic(): assert canonical_record_hash({'b':1,'a':2})==canonical_record_hash({'a':2,'b':1})
def test_08_hash_uses_complete_sanitized_row(): assert canonical_record_hash({'Code':'72030','Future':1}) != canonical_record_hash({'Code':'72030'})
def test_09_content_equivalent_rows_deduplicate(): assert len(merge_exact_duplicates([map_financial_summary_row(RAW),map_financial_summary_row(dict(reversed(list(RAW.items()))))])[0])==1
def test_10_changed_content_is_revision_not_duplicate(): assert len(merge_exact_duplicates([map_financial_summary_row(RAW),map_financial_summary_row({**RAW,'Sales':999})])[0])==2
def test_11_disclosure_date_preserved(): assert map_financial_summary_row(RAW)['disclosed_date']=='2026-01-05'
def test_12_disclosure_time_preserved(): assert map_financial_summary_row(RAW)['disclosed_time']=='15:00:00'
def test_13_disclosure_number_preserved(): assert map_financial_summary_row(RAW)['disclosure_number']=='1'
def test_14_document_type_preserved(): assert map_financial_summary_row(RAW)['document_type'].startswith('FY')
def test_15_actual_consolidated_fields_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('sales','operating_profit','ordinary_profit','net_profit','eps','diluted_eps','bps','total_assets','equity','shareholders_equity','roe','equity_ratio')]==list(range(100,101))+list(range(2,13))
def test_16_cashflow_fields_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('cash_flow_operating','cash_flow_investing','cash_flow_financing','cash_and_equivalents')]==[13,14,15,16]
def test_17_share_count_fields_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('shares_outstanding_fy','treasury_shares_fy','average_shares')]==[17,18,19]
def test_18_current_forecasts_and_2q_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('forecast_sales','forecast_operating_profit','forecast_ordinary_profit','forecast_net_profit','forecast_eps','forecast_sales_2q','forecast_operating_profit_2q','forecast_ordinary_profit_2q','forecast_net_profit_2q','forecast_eps_2q')]==list(range(20,30))
def test_19_next_forecasts_and_2q_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('next_forecast_sales','next_forecast_operating_profit','next_forecast_ordinary_profit','next_forecast_net_profit','next_forecast_eps','next_forecast_sales_2q','next_forecast_operating_profit_2q','next_forecast_ordinary_profit_2q','next_forecast_net_profit_2q','next_forecast_eps_2q')]==list(range(30,40))
def test_20_actual_dividend_fields_map(): assert map_financial_summary_row(RAW)['dividend_total_annual']==6
def test_21_current_dividend_fields_map(): assert map_financial_summary_row(RAW)['forecast_payout_ratio_annual']==16
def test_22_next_dividend_fields_map(): assert map_financial_summary_row(RAW)['next_forecast_payout_ratio_annual']==23
def test_23_nonconsolidated_actuals_map(): assert map_financial_summary_row(RAW)['non_consolidated_equity_ratio']==34
def test_24_nonconsolidated_current_forecasts_map(): assert map_financial_summary_row(RAW)['forecast_non_consolidated_eps']==39
def test_25_nonconsolidated_next_forecasts_map(): assert map_financial_summary_row(RAW)['next_forecast_non_consolidated_eps']==44
def test_26_revision_flags_map():
 r=map_financial_summary_row(RAW); assert [r[x] for x in ('change_actual_estimate','change_by_accounting_revision','change_no_accounting_revision','material_change_subsequent','significant_change_in_consolidation','retroactive_restatement')]==list('abcdef')
def test_27_empty_ticker_completed(tmp_path):
 u=uni(tmp_path,'7203'); s=acquire(Fake({'7203':[]}),u,tmp_path/'s'); assert s['completed_tickers']==['7203'] and s['empty_tickers']==['7203'] and json.loads((tmp_path/'s/raw/7203.json').read_text())==[]
def test_28_nonrate_provider_error_continues(tmp_path):
 u=uni(tmp_path,'7203','8306'); waits=[]; f=Fake({'8306':[RAW]},{'7203':ProviderError('bad request')})
 s=acquire_incremental(f,universe_file=u,store_dir=tmp_path/'s',allow_network=True,sleep_seconds=0,sleep_fn=waits.append); assert f.calls==['7203','8306'] and s['failed_tickers']=={'7203':'provider_error'} and not waits
def test_29_rate_limit_retries_same_ticker(tmp_path):
 u=uni(tmp_path,'7203'); f=Fake({'7203':[RAW]},{'7203':[ProviderError('429'),None]}); acquire(f,u,tmp_path/'s',rate_limit_cooldown_seconds=1); assert f.calls==['7203','7203']
def test_30_linear_rate_limit_cooldown(tmp_path):
 u=uni(tmp_path,'7203'); waits=[]; f=Fake({}, {'7203':ProviderError('429')}); acquire_incremental(f,universe_file=u,store_dir=tmp_path/'s',allow_network=True,sleep_seconds=0,rate_limit_cooldown_seconds=900,max_rate_limit_retries=3,sleep_fn=waits.append); assert waits==[900,1800,2700]
def test_31_rate_limit_exhaustion_failed(tmp_path):
 s=acquire(Fake({}, {'7203':ProviderError('429')}),uni(tmp_path,'7203'),tmp_path/'s',max_rate_limit_retries=0); assert s['failed_tickers']=={'7203':'rate_limit_exhausted'}
def test_32_successful_retry_completes(tmp_path):
 u=uni(tmp_path,'7203'); s=acquire(Fake({'7203':[RAW]},{'7203':[ProviderError('429'),None]}),u,tmp_path/'s'); assert s['completed_tickers']==['7203'] and not s['failed_tickers']
def test_33_resume_skips_valid_completed(tmp_path):
 u=uni(tmp_path,'7203'); f=Fake({'7203':[RAW]}); acquire(f,u,tmp_path/'s'); acquire(f,u,tmp_path/'s'); assert f.calls==['7203']
def test_34_missing_raw_invalidates_skip(tmp_path):
 u=uni(tmp_path,'7203'); f=Fake({'7203':[RAW]}); acquire(f,u,tmp_path/'s'); (tmp_path/'s/raw/7203.json').unlink(); acquire(f,u,tmp_path/'s'); assert f.calls==['7203','7203']
def test_35_malformed_raw_invalidates_skip(tmp_path):
 u=uni(tmp_path,'7203'); f=Fake({'7203':[RAW]}); acquire(f,u,tmp_path/'s'); (tmp_path/'s/raw/7203.json').write_text('{'); acquire(f,u,tmp_path/'s'); assert f.calls==['7203','7203']
def test_36_rerun_no_hash_duplicate(tmp_path):
 u=uni(tmp_path,'7203'); f=Fake({'7203':[RAW]}); acquire(f,u,tmp_path/'s'); (tmp_path/'s/raw/7203.json').unlink(); acquire(f,u,tmp_path/'s'); assert len(rows(tmp_path/'s'))==1
def test_37_completed_rows_survive_remaining_processing(tmp_path):
 u=uni(tmp_path,'7203','8306'); f=Fake({'7203':[RAW],'8306':[ {**RAW,'Code':'83060'}]}); acquire(f,u,tmp_path/'s'); assert {r['ticker'] for r in rows(tmp_path/'s')}=={'7203','8306'}
def test_38_raw_write_is_atomic_and_clean(tmp_path):
 s=tmp_path/'s'; acquire(Fake({'7203':[RAW]}),uni(tmp_path,'7203'),s); assert json.loads((s/'raw/7203.json').read_text()) and not list((s/'raw').glob('.tmp-*'))
def test_39_state_write_is_atomic_and_clean(tmp_path):
 s=tmp_path/'s'; acquire(Fake({'7203':[RAW]}),uni(tmp_path,'7203'),s); assert state(s)['completed_tickers']==['7203'] and not list(s.glob('.tmp-*'))
def test_40_csv_schema_atomic_and_clean(tmp_path):
 s=tmp_path/'s'; acquire(Fake({'7203':[RAW]}),uni(tmp_path,'7203'),s)
 with (s/CSV_NAME).open(encoding='utf-8',newline='') as f: assert tuple(next(csv.reader(f)))==COLUMNS
 assert not list(s.glob('.tmp-*'))
def test_41_state_has_no_api_key(tmp_path):
 s=tmp_path/'s'; acquire(Fake({'7203':[RAW]}),uni(tmp_path,'7203'),s); assert 'actual-api-key' not in (s/STATE_NAME).read_text()
def test_42_raw_has_no_api_key(tmp_path):
 s=tmp_path/'s'; acquire(Fake({'7203':[RAW]}),uni(tmp_path,'7203'),s); assert 'actual-api-key' not in (s/'raw/7203.json').read_text()
def test_43_csv_and_reports_have_no_api_key(tmp_path):
 s=tmp_path/'s'; u=uni(tmp_path,'7203'); acquire(Fake({'7203':[RAW]}),u,s); assert all('actual-api-key' not in (s/n).read_text() for n in (CSV_NAME,QUALITY_NAME,COVERAGE_NAME))
def test_44_verifier_rejects_corrupt_hash(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s); data=rows(s);data[0]['record_hash']='bad';write_csv(s/CSV_NAME,data)
 with pytest.raises(DataValidationError,match='record hash'): verify_financial_summary(store_dir=s,universe_file=u)
def test_45_verifier_rejects_unknown_ticker(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);data=rows(s);data[0]['ticker']='9999';write_csv(s/CSV_NAME,data)
 with pytest.raises(DataValidationError,match='unknown ticker'): verify_financial_summary(store_dir=s,universe_file=u)
def test_46_verifier_rejects_missing_completed_raw(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);(s/'raw/7203.json').unlink()
 with pytest.raises(DataValidationError,match='raw artifact'): verify_financial_summary(store_dir=s,universe_file=u)
def test_47_verifier_allows_revisions_and_counts_group(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW,{**RAW,'Sales':101}]}),u,s); assert verify_financial_summary(store_dir=s,universe_file=u)['revision_group_count']==1
def test_48_verifier_allows_empty_ticker(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[]}),u,s); assert verify_financial_summary(store_dir=s,universe_file=u)['empty_ticker_count']==1
def test_49_verifier_rejects_unaccounted_ticker(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203','8306');acquire(Fake({'7203':[RAW]},{'8306':ProviderError('no')}),u,s);data=state(s);data['failed_tickers']={};(s/STATE_NAME).write_text(json.dumps(data),encoding='utf-8')
 with pytest.raises(DataValidationError,match='missing processing'): verify_financial_summary(store_dir=s,universe_file=u)
def test_50_field_coverage_includes_every_canonical_and_future_source(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s); report=json.loads((s/COVERAGE_NAME).read_text()); assert set(report['canonical_columns'])==set(COLUMNS) and report['canonical_columns']['sales']['non_null_ratio']==1 and report['raw_source_field_presence_counts']['FutureField']==1
def test_51_universe_hash_mismatch_rejected(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);u.write_text('8306\n')
 with pytest.raises(DataValidationError,match='universe hash'): verify_financial_summary(store_dir=s,universe_file=u)
def test_52_verifier_rejects_residual_tmp(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);(s/'.tmp-stale').write_text('x')
 with pytest.raises(DataValidationError,match='temporary'): verify_financial_summary(store_dir=s,universe_file=u)
def test_53_failed_ticker_can_later_succeed(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({}, {'7203':ProviderError('no')}),u,s);result=acquire(Fake({'7203':[RAW]}),u,s);assert result['completed_tickers']==['7203'] and not result['failed_tickers']
def test_54_provider_raw_accessor_preserves_get_statements(tmp_path):
 cache=tmp_path/'cache/statements';cache.mkdir(parents=True);(cache/'7203.json').write_text(json.dumps([RAW])); p=JQuantsProvider(cache_dir=tmp_path/'cache'); assert p.get_statement_rows('7203')==[RAW] and p.get_statements('7203')[0].ticker=='7203'
def test_55_cli_parser_wires_commands():
 p=build_parser(); assert p.parse_args(['fetch-jquants-financial-summary-incremental','--universe-file','u','--store-dir','s']).command=='fetch-jquants-financial-summary-incremental' and p.parse_args(['verify-jquants-financial-summary','--universe-file','u','--store-dir','s']).command=='verify-jquants-financial-summary'
def test_56_cli_refuses_network_without_flag(tmp_path,capsys,monkeypatch):
 monkeypatch.setattr('jp_stock_analysis.cli.JQuantsProvider',lambda *a,**k: (_ for _ in ()).throw(AssertionError('provider created'))); assert main(['fetch-jquants-financial-summary-incremental','--universe-file',str(uni(tmp_path,'7203')),'--store-dir',str(tmp_path/'s')])==1 and '--allow-network' in capsys.readouterr().err
def test_57_verifier_rejects_unparseable_source_fields(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);data=rows(s);data[0]['source_fields_json']='{' ;write_csv(s/CSV_NAME,data)
 with pytest.raises(DataValidationError,match='source_fields_json'): verify_financial_summary(store_dir=s,universe_file=u)
def test_58_verifier_rejects_nested_secret_source_key(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);data=rows(s);raw=json.loads(data[0]['source_fields_json']);raw['nested']={'Token':'bad'};data[0]['source_fields_json']=json.dumps(raw);data[0]['record_hash']=canonical_record_hash(raw);write_csv(s/CSV_NAME,data)
 with pytest.raises(DataValidationError,match='secret source key'): verify_financial_summary(store_dir=s,universe_file=u)
def test_59_verifier_rejects_corrupt_completed_raw(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);(s/'raw/7203.json').write_text('{}')
 with pytest.raises(DataValidationError,match='raw artifact'): verify_financial_summary(store_dir=s,universe_file=u)
def test_60_verifier_rejects_inconsistent_bookkeeping(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s);data=state(s);data['empty_tickers']=['8306'];(s/STATE_NAME).write_text(json.dumps(data))
 with pytest.raises(DataValidationError,match='bookkeeping'): verify_financial_summary(store_dir=s,universe_file=u)
def test_61_atomic_reports_leave_no_tmp_artifacts(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s); assert not list(s.glob('.tmp-*'))
def test_62_field_coverage_reports_null_counts_for_all_columns(tmp_path):
 s=tmp_path/'s';u=uni(tmp_path,'7203');acquire(Fake({'7203':[RAW]}),u,s); report=json.loads((s/COVERAGE_NAME).read_text())
 assert all(set(report['canonical_columns'][column])=={'non_null_count','null_count','non_null_ratio'} for column in COLUMNS)
