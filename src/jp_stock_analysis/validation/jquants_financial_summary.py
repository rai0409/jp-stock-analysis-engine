"""Resumable, audit-preserving J-Quants V2 Financial Summary acquisition."""
from __future__ import annotations
import csv, hashlib, json, os, re, tempfile, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
from jp_stock_analysis.errors import DataValidationError, ProviderError

SCHEMA_VERSION="1.0"
CSV_NAME="financial_summary.csv"; STATE_NAME="financial_summary_state.json"
QUALITY_NAME="financial_summary_quality_report.json"; COVERAGE_NAME="financial_summary_field_coverage_report.json"
_SECRET=re.compile(r"api_key|apikey|secret|token",re.I)
_MAP={
"Code":"ticker","DiscDate":"disclosed_date","DiscTime":"disclosed_time","DiscNo":"disclosure_number","DocType":"document_type","CurPerType":"current_period_type","CurPerSt":"current_period_start","CurPerEn":"current_period_end","CurFYSt":"current_fiscal_year_start","CurFYEn":"current_fiscal_year_end","NxtFYSt":"next_fiscal_year_start","NxtFYEn":"next_fiscal_year_end",
"Sales":"sales","OP":"operating_profit","OdP":"ordinary_profit","NP":"net_profit","EPS":"eps","DEPS":"diluted_eps","BPS":"bps","TA":"total_assets","Eq":"equity","ShEq":"shareholders_equity","ROE":"roe","EqAR":"equity_ratio","CFO":"cash_flow_operating","CFI":"cash_flow_investing","CFF":"cash_flow_financing","CashEq":"cash_and_equivalents","ShOutFY":"shares_outstanding_fy","TrShFY":"treasury_shares_fy","AvgSh":"average_shares",
"FSales":"forecast_sales","FOP":"forecast_operating_profit","FOdP":"forecast_ordinary_profit","FNP":"forecast_net_profit","FEPS":"forecast_eps","FSales2Q":"forecast_sales_2q","FOP2Q":"forecast_operating_profit_2q","FOdP2Q":"forecast_ordinary_profit_2q","FNP2Q":"forecast_net_profit_2q","FEPS2Q":"forecast_eps_2q",
"NxFSales":"next_forecast_sales","NxFOP":"next_forecast_operating_profit","NxFOdP":"next_forecast_ordinary_profit","NxFNp":"next_forecast_net_profit","NxFEPS":"next_forecast_eps","NxFSales2Q":"next_forecast_sales_2q","NxFOP2Q":"next_forecast_operating_profit_2q","NxFOdP2Q":"next_forecast_ordinary_profit_2q","NxFNP2Q":"next_forecast_net_profit_2q","NxFEPS2Q":"next_forecast_eps_2q",
"Div1Q":"dividend_1q","Div2Q":"dividend_2q","Div3Q":"dividend_3q","DivFY":"dividend_fy","DivAnn":"dividend_annual","DivTotalAnn":"dividend_total_annual","DivUnit":"dividend_unit","PayoutRatioAnn":"payout_ratio_annual","FDiv1Q":"forecast_dividend_1q","FDiv2Q":"forecast_dividend_2q","FDiv3Q":"forecast_dividend_3q","FDivFY":"forecast_dividend_fy","FDivAnn":"forecast_dividend_annual","FDivTotalAnn":"forecast_dividend_total_annual","FDivUnit":"forecast_dividend_unit","FPayoutRatioAnn":"forecast_payout_ratio_annual","NxFDiv1Q":"next_forecast_dividend_1q","NxFDiv2Q":"next_forecast_dividend_2q","NxFDiv3Q":"next_forecast_dividend_3q","NxFDivFY":"next_forecast_dividend_fy","NxFDivAnn":"next_forecast_dividend_annual","NxFDivUnit":"next_forecast_dividend_unit","NxFPayoutRatioAnn":"next_forecast_payout_ratio_annual",
"ChgAcEst":"change_actual_estimate","ChgByASRev":"change_by_accounting_revision","ChgNoASRev":"change_no_accounting_revision","MatChgSub":"material_change_subsequent","SigChgInC":"significant_change_in_consolidation","RetroRst":"retroactive_restatement"}
for p,n in (("NC","non_consolidated_"),("FNC","forecast_non_consolidated_"),("NxFNC","next_forecast_non_consolidated_")):
 for src,dst in {"Sales":"sales","OP":"operating_profit","OdP":"ordinary_profit","NP":"net_profit","EPS":"eps","BPS":"bps","TA":"total_assets","Eq":"equity","ShEq":"shareholders_equity","ROE":"roe","EqAR":"equity_ratio","Sales2Q":"sales_2q","OP2Q":"operating_profit_2q","OdP2Q":"ordinary_profit_2q","NP2Q":"net_profit_2q","EPS2Q":"eps_2q"}.items(): _MAP[p+src]=n+dst
COLUMNS=tuple(["ticker","record_hash",*dict.fromkeys(_MAP.values()),"source_fields_json"])
class StatementRowsProvider(Protocol):
 def get_statement_rows(self,ticker:str)->list[dict[str,Any]]: ...
def normalize_ticker(value:Any)->str:
 s=str(value or "").strip().upper()
 if len(s)==5 and s.endswith("0"): s=s[:-1]
 if not re.fullmatch(r"[0-9][0-9A-Z]{3}",s): raise ValueError(f"invalid ticker: {value!r}")
 return s
def sanitize_raw_row(value:Any)->Any:
 if isinstance(value,dict): return {str(k):sanitize_raw_row(v) for k,v in value.items() if not _SECRET.search(str(k))}
 if isinstance(value,list): return [sanitize_raw_row(v) for v in value]
 return value
def _contains_secret_key(value:Any)->bool:
 if isinstance(value,dict): return any(_SECRET.search(str(k)) or _contains_secret_key(v) for k,v in value.items())
 if isinstance(value,list): return any(_contains_secret_key(v) for v in value)
 return False
def canonical_record_hash(row:dict[str,Any])->str:
 return hashlib.sha256(json.dumps(sanitize_raw_row(row),ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def map_financial_summary_row(raw:dict[str,Any])->dict[str,Any]:
 clean=sanitize_raw_row(raw); out={c:None for c in COLUMNS}; out["ticker"]=normalize_ticker(clean.get("Code")); out["record_hash"]=canonical_record_hash(clean)
 for src,dst in _MAP.items():
  if dst != "ticker": out[dst]=clean.get(src)
 out["source_fields_json"]=json.dumps(clean,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str); return out
def merge_exact_duplicates(rows:Sequence[dict[str,Any]])->tuple[list[dict[str,Any]],int]:
 seen=set(); result=[]
 for r in rows:
  if r["record_hash"] not in seen: seen.add(r["record_hash"]); result.append(r)
 return result,len(rows)-len(result)
def _atomic(path:Path,text:str)->None:
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".tmp-");
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text)
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def _json(path:Path,value:Any)->None:_atomic(path,json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,default=str)+"\n")
def load_csv(path:Path)->list[dict[str,Any]]:
 if not path.exists(): return []
 with path.open(encoding="utf-8",newline="") as f:
  r=csv.DictReader(f)
  if tuple(r.fieldnames or ())!=COLUMNS: raise DataValidationError("financial summary CSV schema mismatch")
  return list(r)
def write_csv(path:Path,rows:Sequence[dict[str,Any]])->None:
 import io
 b=io.StringIO(); w=csv.DictWriter(b,fieldnames=COLUMNS,lineterminator="\n");w.writeheader();w.writerows([{k:"" if r.get(k)is None else r.get(k) for k in COLUMNS} for r in sorted(rows,key=lambda r:(r["ticker"],r["record_hash"]))]);_atomic(path,b.getvalue())
def load_state(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
def _universe(path:Path)->list[str]:
 vals=[]
 for line in path.read_text(encoding="utf-8").splitlines():
  x=line.strip().split(",")[0]
  if x and x.lower() not in {"ticker","code"}: vals.append(normalize_ticker(x))
 return list(dict.fromkeys(vals))
def _state(*,ufile:Path, store:Path, rawdir:Path, universe:list[str], universe_hash:str, completed:set[str], empty:set[str], failed:dict[str,str], retries:list[dict[str,Any]], rows:list[dict[str,Any]])->dict[str,Any]:
 return {"schema_version":SCHEMA_VERSION,"generated_at":datetime.now(timezone.utc).isoformat(),"universe_file":str(ufile),"universe_sha256":universe_hash,"output_file":str(store/CSV_NAME),"raw_dir":str(rawdir),"universe_count":len(universe),"completed_tickers":sorted(completed),"empty_tickers":sorted(empty),"failed_tickers":dict(sorted(failed.items())),"retry_events":retries,"rows":len(rows),"tickers_with_rows":len({r['ticker'] for r in rows}),"status":"complete" if len(completed)+len(failed)==len(universe) else "partial","api_key_status":"not_recorded","secret_included":False}
def acquire_incremental(provider:StatementRowsProvider,*,universe_file:str|Path,store_dir:str|Path,allow_network:bool=False,sleep_seconds:float=90,rate_limit_cooldown_seconds:float=900,max_rate_limit_retries:int=6,sleep_fn:Callable[[float],None]=time.sleep)->dict[str,Any]:
 if not allow_network: raise DataValidationError("financial summary acquisition requires --allow-network")
 ufile=Path(universe_file); store=Path(store_dir); universe=_universe(ufile); uh=hashlib.sha256(ufile.read_bytes()).hexdigest(); rawdir=store/"raw"; statep=store/STATE_NAME; state=load_state(statep); completed=set(state.get("completed_tickers",[])); valid=set()
 for t in completed:
  p=rawdir/f"{t}.json"
  try:
   if isinstance(json.loads(p.read_text()),list):valid.add(t)
  except (OSError,json.JSONDecodeError): pass
 completed=valid; allrows=load_csv(store/CSV_NAME); retries=list(state.get("retry_events",[])); failed={k:v for k,v in state.get("failed_tickers",{}).items() if k not in completed}; empty=set(state.get("empty_tickers",[]))&completed
 for i,t in enumerate(universe):
  if t in completed: continue
  if i and sleep_seconds: sleep_fn(sleep_seconds)
  cycle=0
  while True:
   try: raw=provider.get_statement_rows(t); break
   except ProviderError as e:
    msg=str(e);
    if "429" in msg or "rate" in msg.lower():
     cycle+=1
     if cycle<=max_rate_limit_retries:
      cooldown=rate_limit_cooldown_seconds*cycle; retries.append({"ticker":t,"retry_cycle":cycle,"sleep_seconds":cooldown,"reason":"rate_limit"});sleep_fn(cooldown);continue
    failed[t]="rate_limit_exhausted" if cycle else "provider_error"; raw=None;break
  if raw is not None:
   clean=[sanitize_raw_row(x) for x in raw]; _json(rawdir/f"{t}.json",clean); completed.add(t); failed.pop(t,None)
   if not clean: empty.add(t)
   else: empty.discard(t); allrows.extend(map_financial_summary_row(x) for x in clean)
   allrows,_=merge_exact_duplicates(allrows)
  write_csv(store/CSV_NAME,allrows)
  state=_state(ufile=ufile,store=store,rawdir=rawdir,universe=universe,universe_hash=uh,completed=completed,empty=empty,failed=failed,retries=retries,rows=allrows);_json(statep,state)
 # A zero-ticker universe still has a stable, verifiable CSV and state artifact.
 if not statep.exists():
  write_csv(store/CSV_NAME,allrows)
  state=_state(ufile=ufile,store=store,rawdir=rawdir,universe=universe,universe_hash=uh,completed=completed,empty=empty,failed=failed,retries=retries,rows=allrows);_json(statep,state)
 write_quality_reports(store,ufile); return state
def write_quality_reports(store_dir:str|Path,universe_file:str|Path)->dict[str,Any]:
 store=Path(store_dir); rows=load_csv(store/CSV_NAME);state=load_state(store/STATE_NAME); universe=set(_universe(Path(universe_file))); hashes=[r['record_hash'] for r in rows]; periods=Counter(r['current_period_type'] for r in rows if r['current_period_type']); docs=Counter(r['document_type'] for r in rows if r['document_type']); groups=Counter((r['ticker'],r['current_fiscal_year_end'],r['current_period_type']) for r in rows); q={"rows":len(rows),"unique_record_hashes":len(set(hashes)),"exact_duplicate_count":len(rows)-len(set(hashes)),"universe_count":len(universe),"completed_ticker_count":len(state.get('completed_tickers',[])),"empty_ticker_count":len(state.get('empty_tickers',[])),"failed_ticker_count":len(state.get('failed_tickers',{})),"tickers_with_rows":len({r['ticker'] for r in rows}),"unknown_tickers":sorted({r['ticker'] for r in rows}-universe),"missing_processing_tickers":sorted(universe-set(state.get('completed_tickers',[]))-set(state.get('failed_tickers',{}))),"disclosed_date_min":min((r['disclosed_date'] for r in rows if r['disclosed_date']),default=None),"disclosed_date_max":max((r['disclosed_date'] for r in rows if r['disclosed_date']),default=None),"current_period_type_counts":dict(periods),"document_type_counts":dict(docs),"revision_group_count":sum(v>1 for v in groups.values()),"secret_included":False}; _json(store/QUALITY_NAME,q)
 rawcounts=Counter(); cov={}
 for c in COLUMNS: n=sum(bool(r.get(c)) for r in rows);cov[c]={"non_null_count":n,"null_count":len(rows)-n,"non_null_ratio":n/len(rows) if rows else 0}
 for r in rows:
  try: rawcounts.update(json.loads(r['source_fields_json']).keys())
  except Exception: pass
 report={"canonical_columns":cov,"raw_source_field_presence_counts":dict(sorted(rawcounts.items())),"secret_included":False};_json(store/COVERAGE_NAME,report);return q
def verify_financial_summary(*,store_dir:str|Path,universe_file:str|Path)->dict[str,Any]:
 store=Path(store_dir); rows=load_csv(store/CSV_NAME); state=load_state(store/STATE_NAME); universe=_universe(Path(universe_file)); errors=[]
 if state.get('universe_sha256')!=hashlib.sha256(Path(universe_file).read_bytes()).hexdigest(): errors.append('universe hash mismatch')
 if len({r['record_hash'] for r in rows})!=len(rows):errors.append('duplicate record hash')
 completed=set(state.get('completed_tickers',[])); empty=set(state.get('empty_tickers',[])); failed=set(state.get('failed_tickers',{}))
 if not completed <= set(universe) or not failed <= set(universe): errors.append('ticker outside universe')
 if empty - completed or completed & failed: errors.append('inconsistent ticker bookkeeping')
 if set(universe) - completed - failed: errors.append('missing processing tickers')
 for r in rows:
  try: raw=json.loads(r['source_fields_json']);
  except Exception:errors.append('invalid source_fields_json');continue
  if canonical_record_hash(raw)!=r['record_hash']:errors.append('record hash mismatch')
  if r['ticker'] not in universe:errors.append('unknown ticker')
  if _contains_secret_key(raw):errors.append('secret source key')
 for t in completed:
  try: json.loads((store/'raw'/f'{t}.json').read_text())
  except Exception: errors.append('missing completed raw artifact')
  else:
   try:
    artifact=json.loads((store/'raw'/f'{t}.json').read_text(encoding='utf-8'))
    if not isinstance(artifact,list) or _contains_secret_key(artifact): errors.append('invalid completed raw artifact')
    if t in empty and artifact: errors.append('inconsistent ticker bookkeeping')
   except Exception: pass
 if list(store.glob('.tmp-*')) or (store/'raw').exists() and list((store/'raw').glob('.tmp-*')): errors.append('residual temporary artifact')
 q=write_quality_reports(store,universe_file); errors += ['failed tickers'] if state.get('failed_tickers') else []
 if errors: raise DataValidationError('; '.join(sorted(set(errors))))
 return q
